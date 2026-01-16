#include "dumer_list.h"
#include <algorithm>
#include <tuple>
#include <cstdlib>     // getenv
#include "parallel.h"
#include "bitops.h"

// Query whether deterministic behaviour is requested.
//
// If the environment variable RM_DETERMINISTIC is set to a non‑zero / non‑false value, we break ties in a fully deterministic way (by hashing the candidate codeword).
static inline bool rm_deterministic() {
    const char* v = std::getenv("RM_DETERMINISTIC");
    if (!v) return false;
    return !(v[0]=='0' || v[0]=='f' || v[0]=='F' || v[0]=='n' || v[0]=='N');
}

// 64-bit FNV-1a hash of a BitVec, applied word-by-word.
// This is used only for deterministic tie-breaking, it has no cryptographic meaning.
static inline uint64_t hash_bitvec(const BitVec& b){
    // 64-bit FNV-1a over 64-bit words (little endian)
    uint64_t h = 1469598103934665603ull;
    for (std::size_t i=0;i<b.w.size();++i){
        uint64_t x = b.w[i];
        for (int k=0;k<8;++k){
            uint8_t byte = (uint8_t)((x >> (8*k)) & 0xffu);
            h ^= byte;
            h *= 1099511628211ull;
        }
    }
    h ^= (uint64_t)b.nbits;
    return h;
}

// Merge-and-trim utility: keep the best 'keep' candidates in v.
//
// When RM_DETERMINISTIC is not set, we simply select by distance and allow non-deterministic tie ordering (faster).
//
// When RM_DETERMINISTIC is set, we sort by (distance, hash(codeword_bytes)) to obtain a stable, deterministic order.
static void trim(std::vector<DL_Candidate>& v, int keep){
    if ((int)v.size() <= keep) return;

    if (!rm_deterministic()){
        // Fast non-deterministic path:
        //   1. Use nth_element to partition the smallest `keep` distances.
        //   2. Throw away the rest.
        //   3. Stably sort the survivors by distance for convenience.
        std::nth_element(v.begin(), v.begin()+keep, v.end(),
            [](const DL_Candidate& A, const DL_Candidate& B){ return A.dist < B.dist; });
        v.resize(keep);
        std::stable_sort(v.begin(), v.end(),
            [](const DL_Candidate& A, const DL_Candidate& B){ return A.dist < B.dist; });
        return;
    }

    // Deterministic path: build (dist, hash, index) keys and sort them.
    std::vector<std::tuple<int,uint64_t,std::size_t>> keys;
    keys.reserve(v.size());
    for (std::size_t i=0;i<v.size();++i){
        keys.emplace_back(v[i].dist, hash_bitvec(v[i].code), i);
    }
    std::stable_sort(keys.begin(), keys.end(),
        [](auto const& a, auto const& b){
            if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
            return std::get<1>(a) < std::get<1>(b);
        });

    std::vector<DL_Candidate> out;
    out.reserve(keep);
    const std::size_t K = std::min<std::size_t>(keep, keys.size());
    for (std::size_t j=0;j<K;++j){
        out.push_back(std::move(v[std::get<2>(keys[j])]));
    }
    v.swap(out);
}

// List version of the Dumer decoder for full RM(r,n).
//
// The structure mirrors dumer_decode_full(), but instead of propagating a single estimate (û,v̂) we propagate lists of candidates at each level.
// At the end we return up to list_size full-length codewords with the smallest Hamming distance to y.
std::vector<DL_Candidate>
dumer_list_decode_full(const BitVec& y, int r, int n, int list_size){
    const int N = 1<<n;

    // Base cases: mainly the same logic as in the single-path decoder, but now returning (small) lists.

    if (r < 0) {
        // RM(r,n) = {0} for r < 0
        return { DL_Candidate{ BitVec(N), y.weight() } };
    }
    if (r >= n) {
        // RM(r,n) = F2^N for r ≥ n; y is already a codeword.
        return { DL_Candidate{ y, 0 } };
    }
    if (n == 0){
        // Length-1 code: only codeword is 0 (since r < n here).
        int d0 = y.get(0) ? 1 : 0;
        return { DL_Candidate{ BitVec(1), d0 } };
    }
    if (r == 0){
        // RM(0,n) = {0^N, 1^N}.  Evaluate both and trim to list_size.
        DL_Candidate c0{ BitVec(N), y.weight() };
        BitVec ones(N); for (int i=0;i<N;i++) ones.set1(i);
        DL_Candidate c1{ ones, N - y.weight() };
        std::vector<DL_Candidate> out{c0, c1};
        trim(out, std::min(list_size, 2));
        return out;
    }

    // Recursive step using Plotkin decomposition: y = (y0, y1)
    const int N2 = N >> 1;
    BitVec y0 = y.slice(0, N2);
    BitVec y1 = y.slice(N2, N2);
    BitVec ysum = y0; ysum.xor_inplace(y1);  // observation for v

    // First, decode v candidates from y0 ⊕ y1 in RM(r-1, n-1).
    auto v_list = dumer_list_decode_full(ysum, r-1, n-1, list_size);

    // For each v candidate, decode u candidates and combine them to full codewords (u, u+v).  We accumulate them in `out`, then trim at the end.
    std::vector<DL_Candidate> out;
    out.reserve(std::max(4, list_size) * std::max(4, list_size));

#if defined(RM_HAVE_TBB) || defined(RM_HAVE_OPENMP)
    // If parallelism is enabled and we have enough v candidates, parallelize the outer loop over v_list.  Each worker builds partial results in thread-local storage and we merge them afterwards.
    if (rm_parallel_enabled() && v_list.size() >= 2) {
        #if defined(RM_HAVE_TBB)
            tbb::combinable<std::vector<DL_Candidate>> tls;
            rm_par_for(0, v_list.size(), [&](std::size_t i){
                const auto& vCand = v_list[i];
                BitVec tmp = y0; tmp.xor_inplace(vCand.code);
                auto u_list = dumer_list_decode_full(tmp, r, n-1, list_size);
                auto& local = tls.local();
                for (const auto& uCand : u_list){
                    BitVec right = uCand.code; right.xor_inplace(vCand.code);
                    BitVec code = BitVec::concat(uCand.code, right);
                    int dist = hamming_distance(y, code);
                    local.push_back(DL_Candidate{ std::move(code), dist });
                }
            });
            tls.combine_each([&](std::vector<DL_Candidate>& v){ out.insert(out.end(), v.begin(), v.end()); });
        #else
            #pragma omp parallel
            {
                std::vector<DL_Candidate> local;
                #pragma omp for schedule(dynamic)
                for (long long ii=0; ii<(long long)v_list.size(); ++ii){
                    const auto& vCand = v_list[(std::size_t)ii];
                    BitVec tmp = y0; tmp.xor_inplace(vCand.code);
                    auto u_list = dumer_list_decode_full(tmp, r, n-1, list_size);
                    for (const auto& uCand : u_list){
                        BitVec right = uCand.code; right.xor_inplace(vCand.code);
                        BitVec code = BitVec::concat(uCand.code, right);
                        int dist = hamming_distance(y, code);
                        local.push_back(DL_Candidate{ std::move(code), dist });
                    }
                }
                #pragma omp critical
                out.insert(out.end(), local.begin(), local.end());
            }
        #endif
    } else
#endif
    {
        // Sequential path: same logic as above but without threading.
        for (const auto& vCand : v_list){
            BitVec tmp = y0; tmp.xor_inplace(vCand.code);
            auto u_list = dumer_list_decode_full(tmp, r, n-1, list_size);
            for (const auto& uCand : u_list){
                BitVec right = uCand.code; right.xor_inplace(vCand.code);
                BitVec code = BitVec::concat(uCand.code, right);
                int dist = hamming_distance(y, code);
                out.push_back(DL_Candidate{ std::move(code), dist });
            }
        }
    }

    // Keep only the best list_size candidates according to our trimming rules.
    trim(out, list_size);
    return out;
}
