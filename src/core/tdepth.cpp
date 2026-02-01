#include "tdepth.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <vector>
#include <string>
#include <bit>

// Existing T-depth estimator (safe upper bound)

// Estimate T-depth of a linear-phase circuit from an "odd" pattern in the
// punctured representation.
//
// At the moment this is a very simple bound: each odd monomial is assumed to require its own T-layer, so the depth equals the Hamming weight of the odd set.
// This is guaranteed to be an upper bound, the scheduler below can often do better.
int estimate_tdepth_from_punctured(const BitVec& odd_punct, int /*n*/) {
    // Upper bound: equals Hamming weight of the odd set
    return odd_punct.weight();
}

// DSATUR exact / color-improved scheduler
//
// Given a set of monomials, we build an interference graph where vertices
// are monomials and two vertices share an edge if their supports overlap
// (mi & mj) != 0.  A proper graph coloring corresponds to a layering of
// T gates where each color class is a set of monomials that can be
// implemented in parallel without ancillae.
//
// We then run DSATUR (a branch-and-bound graph coloring algorithm) to
// search for a minimal coloring, subject to soft caps on time and node
// count controlled by environment variables:
//
//   RM_TDEPTH_COLOR_NODES : max search nodes (default 500000)
//   RM_TDEPTH_COLOR_MS    : max search time in ms (default 50)
//
// If the exact search is cut off by these caps, we fall back to the
// greedy coloring, which is still valid but may not be optimal.

namespace {

static inline int ctz_u64(uint64_t x) {
    return std::countr_zero(x);
}

// Read an integer environment variable, with default and clamping to >= 0.
static int getenv_int(const char* key, int defv) {
    if (const char* s = std::getenv(key)) {
        try { return std::max(0, std::stoi(s)); }
        catch (...) {}
    }
    return defv;
}

// Simple adjacency representation for the conflict graph.
struct Graph {
    int N = 0;                      // number of vertices
    std::vector<uint64_t> adjBits;  // adjacency bitset per vertex (assume N <= 64)
    std::vector<int> degree;        // degree of each vertex
};

// Filter and normalize monomials:
//
//   * sort;
//   * remove duplicates;
//   * drop constant monomial (mask == 0).
static std::vector<int> filter_monoms(const std::vector<int>& monoms_raw) {
    std::vector<int> tmp = monoms_raw;
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    std::vector<int> out;
    out.reserve(tmp.size());
    for (int m : tmp) if (m != 0) out.push_back(m); // drop constants
    return out;
}

// Build the conflict graph from filtered monomials.
//
// Vertices: monoms[i]
// Edge (i,j) present iff monoms[i] and monoms[j] share at least one variable,
// i.e. (monoms[i] & monoms[j]) != 0.
static Graph build_graph_from_filtered(const std::vector<int>& monoms) {
    Graph G;
    G.N = static_cast<int>(monoms.size());
    G.adjBits.assign(G.N, 0);
    G.degree.assign(G.N, 0);
    for (int i = 0; i < G.N; ++i) {
        for (int j = i + 1; j < G.N; ++j) {
            if ((monoms[i] & monoms[j]) != 0) {
                G.adjBits[i] |= (1ull << j);
                G.adjBits[j] |= (1ull << i);
                G.degree[i]++; G.degree[j]++;
            }
        }
    }
    return G;
}

// DSATUR solver state.
struct DSAT {
    const Graph& G;
    const int N;
    const int color_cap_nodes;     // max visited nodes in search tree
    const int time_ms_cap;         // max time in milliseconds
    std::chrono::steady_clock::time_point t0;

    std::vector<int> color;        // -1 uncolored, else [0..best-1]
    std::vector<uint64_t> satMask; // colors used by neighbors (bitset)
    std::vector<int> order_deg;    // degrees (for tie-breaks)

    int nodes = 0;                              // number of DFS nodes visited
    int best = std::numeric_limits<int>::max(); // best (smallest) color count
    std::vector<int> bestColoring;              // coloring achieving `best`

    DSAT(const Graph& g, int cap_nodes, int time_ms)
        : G(g),
          N(g.N),
          color(N, -1),
          satMask(N, 0),
          order_deg(N, 0),
          color_cap_nodes(cap_nodes),
          time_ms_cap(time_ms),
          t0(std::chrono::steady_clock::now()) {
        for (int i = 0; i < N; ++i) order_deg[i] = G.degree[i];
    }

    // Check whether time budget is exhausted.
    bool time_up() const {
        if (time_ms_cap <= 0) return false;
        auto now = std::chrono::steady_clock::now();
        auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
        return ms >= time_ms_cap;
    }

    // Greedy coloring to obtain an initial upper bound.
    //
    // Vertices are ordered by decreasing degree; each vertex is assigned the
    // smallest color not used by its already-colored neighbors.
    int greedy_bound(std::vector<int>& greedyColor) const {
        greedyColor.assign(N, -1);
        std::vector<int> idx(N); for (int i=0;i<N;++i) idx[i]=i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){
            if (G.degree[a] != G.degree[b]) return G.degree[a] > G.degree[b];
            return a < b;
        });
        int used = 0;
        for (int v : idx) {
            uint64_t forbid = 0;
            uint64_t nb = G.adjBits[v];
            while (nb) {
                int u = ctz_u64(nb);
                nb &= (nb - 1);
                int cu = greedyColor[u];
                if (cu >= 0) forbid |= (1ull << cu);
            }
            int c = 0;
            while ( (forbid >> c) & 1ull ) ++c;
            greedyColor[v] = c;
            if (c + 1 > used) used = c + 1;
        }
        return used;
    }

    // Choose next vertex to color: maximum saturation degree, tie by degree.
    int pick_vertex() const {
        int bestV = -1, bestSat = -1, bestDeg = -1;
        for (int v = 0; v < N; ++v) if (color[v] < 0) {
            int sat = std::popcount(satMask[v]);
            int deg = G.degree[v];
            if (sat > bestSat || (sat == bestSat && deg > bestDeg)) {
                bestSat = sat; bestDeg = deg; bestV = v;
            }
        }
        return bestV;
    }

    // Assign color c to vertex v, updating saturation masks of neighbors.
    void assign_color(int v, int c, std::vector<int>& changed) {
        color[v] = c;
        uint64_t nb = G.adjBits[v];
        while (nb) {
            int u = ctz_u64(nb);
            nb &= (nb - 1);
            if ( (satMask[u] & (1ull << c)) == 0 ) {
                satMask[u] |= (1ull << c);
                changed.push_back(u);
            }
        }
    }

    // Undo color assignment at v and restore saturation masks for neighbors listed in `changed` by recomputing them.
    void undo_assign(int v, int /*c*/, const std::vector<int>& changed) {
        color[v] = -1;
        for (int u : changed) {
            // recompute satMask[u]
            uint64_t mask = 0;
            uint64_t nb = G.adjBits[u];
            while (nb) {
                int w = ctz_u64(nb);
                nb &= (nb - 1);
                int cw = color[w];
                if (cw >= 0) mask |= (1ull << cw);
            }
            satMask[u] = mask;
        }
    }

    // Depth-first search with branch-and-bound over color assignments.
    void dfs(int usedColors) {
        if (nodes++ >= color_cap_nodes || time_up()) return;
        if (usedColors >= best) return; // cannot beat current best

        int v = pick_vertex();
        if (v < 0) {
            // All vertices colored; update best if improved.
            best = usedColors;
            bestColoring = color;
            return;
        }

        // Colors already used by neighbors of v.
        uint64_t forbid = satMask[v];

        // Try existing colors 0..usedColors-1.
        for (int c = 0; c < usedColors; ++c) {
            if ( (forbid >> c) & 1ull ) continue;
            std::vector<int> changed;
            assign_color(v, c, changed);
            dfs(usedColors);
            undo_assign(v, c, changed);
            if (best <= usedColors) return; // optimal for this branch
            if (nodes >= color_cap_nodes || time_up()) return;
        }

        // Try a new color `usedColors` as last resort.
        {
            std::vector<int> changed;
            assign_color(v, usedColors, changed);
            dfs(usedColors + 1);
            undo_assign(v, usedColors, changed);
        }
    }
}; // struct DSAT

} // namespace

// Main entry point for exact / color-improved scheduling.
//
// monoms_raw : list of monomial masks (bitsets over n variables).
// n          : number of variables/qubits (used only for interpretation).
// budget     : optional target depth.  If > 0 and we find a coloring
//              with depth <= budget, we can early-stop.
// layers     : on return, contains `depth` layers, each a vector of monomial
//              masks that can be executed in parallel.
//
// Returns:
//   depth    : best depth found (≤ |monoms|), guaranteed ≤ greedy coloring.
int schedule_tdepth_from_monoms(const std::vector<int>& monoms_raw,
                                int /*n*/,
                                int budget,
                                std::vector<std::vector<int>>& layers)
{
    layers.clear();

    // Build filtered list (drop constants, deduplicate).
    std::vector<int> monoms = filter_monoms(monoms_raw);
    Graph G = build_graph_from_filtered(monoms);
    const int N = G.N;

    if (N <= 0) {
        return 0;
    }
    if (N == 1) {
        layers = { { monoms[0] } };
        return 1;
    }

    // Search caps from environment (with reasonable defaults).
    const int node_cap = getenv_int("RM_TDEPTH_COLOR_NODES", 500000);
    const int time_ms  = getenv_int("RM_TDEPTH_COLOR_MS",    50);

    DSAT solver(G, node_cap, time_ms);

    // Initial upper bound via greedy coloring.
    std::vector<int> greedyColor;
    int ub = solver.greedy_bound(greedyColor);
    solver.best = ub;
    solver.bestColoring = greedyColor;

    // Budget check: if greedy is already within budget, accept it directly.
    if (budget > 0 && ub <= budget) {
        layers.assign(ub, {});
        for (int i = 0; i < N; ++i) layers[greedyColor[i]].push_back(monoms[i]);
        return ub;
    }

    // Exact search (bounded by node and time caps).
    solver.dfs(/*usedColors*/ 0);
    int depth = solver.best;

    // Materialize layers from best coloring (may equal greedy if bounded out).
    layers.assign(depth, {});
    for (int i = 0; i < N; ++i) {
        int c = solver.bestColoring[i];
        if (c >= 0) layers[c].push_back(monoms[i]);
    }

    return depth;
}
