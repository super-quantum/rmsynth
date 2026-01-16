#include "dumer_list_chase.h"
#include <algorithm>
#include <tuple>
#include <cstdlib>

// Same RM_DETERMINISTIC helper as in dumer_list.cpp.
//
// When RM_DETERMINISTIC is set to something not starting with 0/f/F/n/N, we enable deterministic tie-breaking for reproducible runs.
static inline bool rm_deterministic() {
    const char* v = std::getenv("RM_DETERMINISTIC");
    if (!v) return false;
    return !(v[0]=='0' || v[0]=='f' || v[0]=='F' || v[0]=='n' || v[0]=='N');
}

// FNV-1a hash of a BitVec, used only for deterministic ordering.
static inline uint64_t hash_bitvec(const BitVec& b){
    uint64_t h = 1469598103934665603ull;
    for (std::size_t i=0;i<b.w.size();++i){
        uint64_t x = b.w[i];
        for (int k=0;k<8;++k){
            uint8_t byte = (uint8_t)((x >> (8*k)) & 0xffu);
            h ^= byte; h *= 1099511628211ull;
        }
    }
    h ^= (uint64_t)b.nbits;
    return h;
}

// Simple Hamming distance via XOR + weight.
// (Wrapper so this file does not depend on hamming_distance_words.)
static inline int hamming_dist(const BitVec& a, const BitVec& b){
    BitVec t = a; t.xor_inplace(b); return t.weight();
}

// Trim a list of candidates down to `keep` entries, using the same rules as in dumer_list.cpp.
static void trim(std::vector<DL_Candidate>& v, int keep){
    if ((int)v.size() <= keep) return;
    if (!rm_deterministic()){
        std::nth_element(v.begin(), v.begin()+keep, v.end(),
            [](const DL_Candidate& A, const DL_Candidate& B){ return A.dist < B.dist; });
        v.resize(keep);
        std::stable_sort(v.begin(), v.end(),
            [](const DL_Candidate& A, const DL_Candidate& B){ return A.dist < B.dist; });
        return;
    }
    // Deterministic path: sort by (distance, hash(code)) and retain best `keep`.
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

// Dumer-List + top-level Chase-style reliability flips on the full truth table.
//
// y            : input word of length 2^n (BitVec).
// r, n         : RM(r,n) parameters.
// list_size    : baseline beam size for dumer_list_decode_full.
// chase_t      : maximum size of flip patterns (currently 1 or 2 supported).
// chase_limit  : global cap on the number of flip patterns explored.
//
// Algorithm outline:
//   1. Run baseline Dumer list decoder to get `base` candidates.
//   2. Take the best baseline candidate base[0] and look at positions where
//      it disagrees with y; these are treated as "least reliable" positions.
//   3. Consider flipping patterns (subsets) of these positions of size 1..chase_t,
//      up to an overall limit of `chase_limit` patterns.
//   4. For each flipped pattern, run a 1-element Dumer list decoder and add
//      the single returned candidate to the pool.
//   5. Deduplicate equal codewords and keep the best `list_size` by distance.
std::vector<DL_Candidate>
dumer_list_chase_full(const BitVec& y, int r, int n,
                      int list_size, int chase_t, int chase_limit)
{
    // 1) Baseline beam via list decoder.
    auto base = dumer_list_decode_full(y, r, n, list_size);

    // If we cannot or do not want to chase, just return the baseline list.
    if (chase_t <= 0 || chase_limit <= 0 || base.empty()){
        return base;
    }

    // 2) Collect mismatch positions between y and the best baseline candidate.
    //    These positions are where we "suspect" flips might help.
    const DL_Candidate& best0 = base[0];
    std::vector<int> errs;
    errs.reserve(y.nbits);
    for (int i=0;i<y.nbits;i++){
        if (y.get(i) != best0.code.get(i)) errs.push_back(i);
    }
    if (errs.empty()){
        // Already matches exactly; no need for Chase extensions.
        return base;
    }

    // Limit the number of error positions considered for flips.
    // We will only consider the first K positions in `errs`.
    int K = std::min((int)errs.size(), chase_limit);

    // 3–4) Enumerate flip patterns of size 1..chase_t with a global cap.

    std::vector<DL_Candidate> cand = base;  // start with baseline candidates
    int emitted = 0;                        // number of flip patterns tried

    auto push_candidate = [&](const BitVec& code){
        int d = hamming_dist(y, code);
        cand.push_back(DL_Candidate{code, d});
    };

    // Helper: run list decoder in "1-candidate" mode on a modified input.
    auto decode_1 = [&](const BitVec& yflip){
        auto one = dumer_list_decode_full(yflip, r, n, 1);
        if (!one.empty()) push_candidate(one[0].code);
    };

    // Size-1 flips.
    for (int a=0; a<K && emitted<chase_limit; ++a){
        BitVec yf = y; yf.toggle(errs[a]);
        decode_1(yf); ++emitted;
    }

    // Size-2 flips.
    if (chase_t >= 2){
        for (int a=0; a<K && emitted<chase_limit; ++a){
            for (int b=a+1; b<K && emitted<chase_limit; ++b){
                BitVec yf = y; yf.toggle(errs[a]); yf.toggle(errs[b]);
                decode_1(yf); ++emitted;
            }
        }
    }

    // 5) Deduplicate by codeword (exact equality on bytes).
    std::vector<DL_Candidate> uniq;
    uniq.reserve(cand.size());
    std::vector<std::vector<uint8_t>> seen; seen.reserve(cand.size());
    for (auto &c : cand){
        auto by = c.code.to_bytes_le();
        bool dup=false;
        for (auto &s : seen){
            if (s == by){ dup=true; break; }
        }
        if (!dup){ seen.push_back(std::move(by)); uniq.push_back(std::move(c)); }
    }

    // Keep at most list_size best candidates (by distance, then hash).
    trim(uniq, list_size);
    return uniq;
}
