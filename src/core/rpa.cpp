#include "rpa.h"
#include <vector>
#include <algorithm>
#include <cassert>
#include <unordered_set>
#include "dumer_list.h"

// Insert a zero bit at position j into t.
//
// If t encodes an (n-1)-bit index with bits except position j, this returns the corresponding n-bit index where bit j is 0.
//
// Example (n=3, j=1):
//   t (2 bits) : b2 b0   (positions {2,0})
//   insert_zero_bit(t, 1) => b2 0 b0
static inline int insert_zero_bit(int t, int j){
    int low  = t & ((1<<j) - 1);
    int high = t >> j;
    return (high << (j+1)) | low;
}

// Project y along axis j to obtain two (n-1)-variable functions y0,y1.
//
// For each t ∈ {0,..,2^{n-1}-1} representing an assignment to all variables
// except x_j, we define:
//   b0 = insert_zero_bit(t, j)       (x_j = 0)
//   b1 = b0 | (1<<j)                 (x_j = 1)
// and set:
//
//   y0[t] = y[b0]
//   y1[t] = y[b1]
static inline void project_axis_j(const BitVec& y, int n, int j, BitVec& y0, BitVec& y1){
    const int N  = 1<<n;
    const int N2 = N>>1;
    assert(y.nbits == N);
    y0 = BitVec(N2);
    y1 = BitVec(N2);
    for (int t=0; t<N2; ++t){
        int b0 = insert_zero_bit(t, j);
        int b1 = b0 | (1<<j);
        if (y.get(b0)) y0.set1(t);
        if (y.get(b1)) y1.set1(t);
    }
}

// Compose a full-length codeword for axis j from u and v (each of length 2^{n-1}).
//
// For each t, let b0, b1 be as in project_axis_j.  We set:
//
//   c[b0] = u[t]
//   c[b1] = u[t] XOR v[t]
//
// This mirrors the Plotkin decomposition RM(r,n) = { (u, u+v) } but with coordinates rearranged to treat x_j as the last bit.
static inline BitVec compose_axis_j(const BitVec& u, const BitVec& v, int n, int j){
    const int N2 = u.nbits;
    const int N  = N2<<1;
    BitVec out(N);
    for (int t=0; t<N2; ++t){
        int b0 = insert_zero_bit(t, j);
        int b1 = b0 | (1<<j);
        bool ub = u.get(t);
        bool vb = v.get(t);
        if (ub) out.set1(b0);
        if (ub ^ vb) out.set1(b1);
    }
    return out;
}

// One RPA-1 iteration on the full truth table.
//
// For each coordinate axis j:
//
//   1. Project y to (y0, y1) along axis j.
//   2. Decode v̂ ∈ RM(r-1, n-1) from y0 XOR y1.
//   3. Decode û ∈ RM(r,   n-1) from y0 XOR v̂.
//   4. Compose a candidate codeword cand using compose_axis_j(û, v̂).
//
// All candidate codewords (one per axis) then vote bitwise.  For each index i,
// we count how many axis-candidates have a 1 at position i.  y_next[i] becomes:
//   * 1 if votes(i) > n/2,
//   * y[i] if votes(i) == n/2 (tie: keep original),
//   * 0 otherwise.
//
// This is a standard "projection-aggregation" heuristic for RM decoding.
static BitVec rpa1_iter(const BitVec& y, int r, int n){
    const int N  = 1<<n;
    std::vector<int> votes(N, 0);   // per-position vote count across axes

#if defined(RM_HAVE_TBB) || defined(RM_HAVE_OPENMP)
    if (rm_parallel_enabled() && n >= 2) {
        #if defined(RM_HAVE_TBB)
            // TBB: each thread gets its own vote array; we combine at the end.
            tbb::combinable<std::vector<int>> tls([&]{ return std::vector<int>(N, 0); });
            rm_par_for(0, (std::size_t)n, [&](std::size_t j){
                BitVec y0, y1; project_axis_j(y, n, (int)j, y0, y1);
                BitVec ysum = y0; ysum.xor_inplace(y1);
                BitVec vhat = dumer_decode_full(ysum, r-1, n-1);
                BitVec tmp  = y0; tmp.xor_inplace(vhat);
                BitVec uhat = dumer_decode_full(tmp,  r,   n-1);
                BitVec cand = compose_axis_j(uhat, vhat, n, (int)j);

                auto& local = tls.local();
                // Accumulate set bits by scanning words.
                const std::size_t W = cand.w.size();
                for (std::size_t w=0; w<W; ++w){
                    uint64_t x = cand.w[w];
                    if (!x) continue;
                    std::size_t base = w<<6;
                    while (x){
                        unsigned long idx;
                        #if defined(__GNUG__) || defined(__clang__)
                            idx = (unsigned long)__builtin_ctzll(x);
                        #else
                            // Portable ctzll: count trailing zeros manually.
                            idx = 0; uint64_t yx=x; while ((yx & 1ull)==0){ ++idx; yx >>= 1; }
                        #endif
                        std::size_t i = base + (std::size_t)idx;
                        if (i < (std::size_t)N) local[i] += 1;
                        x &= (x - 1);  // clear lowest set bit
                    }
                }
            });
            // Combine thread-local vote arrays.
            tls.combine_each([&](const std::vector<int>& v){
                const std::size_t M = v.size();
                for (std::size_t i=0;i<M;++i) votes[i] += v[i];
            });
        #else
            // OpenMP implementation.
            #pragma omp parallel
            {
                std::vector<int> local(N, 0);
                #pragma omp for schedule(dynamic)
                for (int j=0; j<n; ++j){
                    BitVec y0, y1; project_axis_j(y, n, j, y0, y1);
                    BitVec ysum = y0; ysum.xor_inplace(y1);
                    BitVec vhat = dumer_decode_full(ysum, r-1, n-1);
                    BitVec tmp  = y0; tmp.xor_inplace(vhat);
                    BitVec uhat = dumer_decode_full(tmp,  r,   n-1);
                    BitVec cand = compose_axis_j(uhat, vhat, n, j);

                    const std::size_t W = cand.w.size();
                    for (std::size_t w=0; w<W; ++w){
                        uint64_t x = cand.w[w];
                        if (!x) continue;
                        std::size_t base = w<<6;
                        while (x){
                            unsigned long idx;
                            #if defined(__GNUG__) || defined(__clang__)
                                idx = (unsigned long)__builtin_ctzll(x);
                            #else
                                idx = 0; uint64_t yx=x; while ((yx & 1ull)==0){ ++idx; yx >>= 1; }
                            #endif
                            std::size_t i = base + (std::size_t)idx;
                            if (i < (std::size_t)N) local[i] += 1;
                            x &= (x - 1);
                        }
                    }
                }
                // Merge local vote counts into global `votes`.
                #pragma omp critical
                {
                    for (int i=0;i<N;++i) votes[i] += local[i];
                }
            }
        #endif
    } else
#endif
    {
        // Sequential implementation of the same logic.
        for (int j=0; j<n; ++j){
            BitVec y0, y1; project_axis_j(y, n, j, y0, y1);
            BitVec ysum = y0; ysum.xor_inplace(y1);
            BitVec vhat = dumer_decode_full(ysum, r-1, n-1);
            BitVec tmp  = y0; tmp.xor_inplace(vhat);
            BitVec uhat = dumer_decode_full(tmp,  r,   n-1);
            BitVec cand = compose_axis_j(uhat, vhat, n, j);
            const std::size_t W = cand.w.size();
            for (std::size_t w=0; w<W; ++w){
                uint64_t x = cand.w[w];
                if (!x) continue;
                std::size_t base = w<<6;
                while (x){
                    unsigned long idx;
                    #if defined(__GNUG__) || defined(__clang__)
                        idx = (unsigned long)__builtin_ctzll(x);
                    #else
                        idx = 0; uint64_t yx=x; while ((yx & 1ull)==0){ ++idx; yx >>= 1; }
                    #endif
                    std::size_t i = base + (std::size_t)idx;
                    if (i < (std::size_t)N) votes[i] += 1;
                    x &= (x - 1);
                }
            }
        }
    }

    // Majority vote across axes, ties fall back to original y.
    BitVec y_next(N);
    for (int i=0;i<N;i++){
        int v = votes[i];
        if (v > n/2) {
            y_next.set1(i);
        } else if (v == n/2) {
            if (y.get(i)) y_next.set1(i);
        }
    }
    return y_next;
}

// RPA-1 seeding followed by (optional) list-based refinement.
//
// y                : input truth table of length 2^n.
// r, n             : RM(r,n) parameters.
// iters            : number of RPA-1 iterations to apply.
// final_list_size  : if 1, perform a single Dumer decode, if >1, run list decoders around both the RPA estimate and y and pick the best candidate by distance to y.
BitVec rpa1_seed_full(const BitVec& y, int r, int n, int iters, int final_list_size){
    const int N = 1<<n;
    assert(y.nbits == N);

    // RPA-1 seeding phase
    BitVec est = y;
    int I = std::max(1, iters);
    for (int t=0; t<I; ++t){
        est = rpa1_iter(est, r, n);
    }

    // Distance to ORIGINAL y (not to est).
    auto dist_to_y = [&](const BitVec& code)->int{
        BitVec diff = code; diff.xor_inplace(y);
        return diff.weight();
    };

    const int ls = std::max(1, final_list_size);

    if (ls == 1){
        // Simple case: decode once around est and once directly from y, then choose whichever is closer to y.
        BitVec c_est = dumer_decode_full(est, r, n);
        BitVec c_y   = dumer_decode_full(y,   r, n);
        return (dist_to_y(c_est) <= dist_to_y(c_y)) ? c_est : c_y;
    } else {
        // Beam around est and beam around y, pick best by distance to y.
        auto beam_est = dumer_list_decode_full(est, r, n, ls);
        auto beam_y   = dumer_list_decode_full(y,   r, n, ls);

        bool have = false;
        int best_d = 0;
        BitVec best_c(N);

        auto consider_pool = [&](const auto& pool){
            for (const auto& cand : pool){
                const BitVec& code = cand.code;   // works for DL_Candidate
                int d = dist_to_y(code);
                if (!have || d < best_d){
                    have = true; best_d = d; best_c = code;
                }
            }
        };

        consider_pool(beam_est);
        consider_pool(beam_y);

        if (!have){
            // Safety fallback (should not occur in practice).
            return dumer_decode_full(est, r, n);
        }
        return best_c;
    }
}
