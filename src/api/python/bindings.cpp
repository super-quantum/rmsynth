#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <tuple>
#include <unordered_set>
#include <string>
#include <cstdint>
#include <vector>

#include "rm_code.h"
#include "decode_bruteforce.h"
#include "bitops.h"
#include "dumer.h"
#include "dumer_list.h"
#include "dumer_list_chase.h"
#include "rpa.h"
#include "puncture.h"
#include "anf.h"
#include "tdepth.h"   // depth estimator / scheduler

// Python bindings (rmcore) for the C++ Reed–Muller decoding primitives.
//
// The exposed functions are used by the higher-level Python package to optimize linear-phase / diagonal quantum circuits as described in the reference.  The general flow is:
//
//   1. Represent the quantum operation as a truth table / "odd" set.
//   2. Use various RM decoders (Dumer, list-decoding, RPA, etc.) to find a
//      nearby codeword, interpreted as a phase polynomial.
//   3. Convert the codeword to its ANF and schedule monomials to minimize
//      T-depth.
//
// All bitstrings are passed between Python and C++ as little-endian byte arrays, consistent with BitVec::from_bytes_le()/to_bytes_le().

namespace py = pybind11;

// helpers: Python bytes <-> BitVec

// Convert a Python bytes object (little-endian bit order) to a BitVec containing the first nbits bits.
static BitVec bytes_to_bitvec(py::bytes b, int nbits){
    std::string s = b; // py::bytes -> contiguous bytes
    std::vector<uint8_t> v(s.begin(), s.end());
    return BitVec::from_bytes_le(v, nbits);
}

// Convert a BitVec to a Python bytes object in little-endian order.
static py::bytes bitvec_to_bytes(const BitVec& bv){
    auto v = bv.to_bytes_le();
    return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
}

// Stable 64-bit hash for tie-breaking on codewords (FNV-1a over 64-bit words).
// This is only used to obtain deterministic orderings when multiple candidates have the same distance and estimated T-depth.
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

struct ListDecodeCand {
    int dist;
    int depth;
    uint64_t key;
    BitVec full;
    BitVec punct;
};

PYBIND11_MODULE(rmcore, m) {
    // Core RM code utilities

    // Dimension k = dim(RM(r,n))
    m.def("rm_dimension", &rm_dimension, py::arg("n"), py::arg("r"));

    // Generator matrix rows of RM(r,n) packed into uint64_t words.
    // The corresponding monomial masks are written into the `monoms` vector.
    m.def("rm_generator_rows", &rm_generator_rows,
          py::arg("n"), py::arg("r"), py::arg("monoms"));

    // Reference brute-force decoder (mostly for tests / small instances)
    py::class_<DecodeResult>(m, "DecodeResult")
        .def_readonly("codeword", &DecodeResult::codeword)
        .def_readonly("monomials", &DecodeResult::monomials)
        .def_readonly("ties", &DecodeResult::ties);

    m.def("decode_rm_bruteforce", &decode_rm_bruteforce,
          py::arg("w_bits"), py::arg("n"), py::arg("r"));

    // Dumer decoders on full (unpunctured) length N = 2^n

    // Single-path Dumer decoder on a full truth table.
    // y_full is a bytes object encoding N = 2^n bits.
    m.def("decode_rm_dumer_full",
          [](py::bytes y_full_bytes, int n, int r){
              BitVec y = bytes_to_bitvec(y_full_bytes, 1<<n);
              BitVec c = dumer_decode_full(y, r, n);
              return bitvec_to_bytes(c);
          }, py::arg("y_full"), py::arg("n"), py::arg("r"));


    // Punctured decoders (length L = 2^n - 1)
    //
    // In the optimizer we usually work with the punctured RM code obtained by
    // deleting the coordinate corresponding to input 0^n.  The helpers
    // embed_punctured_to_full() and puncture_full() convert between the
    // punctured representation (length L) and the full one (length 2^n).
    //
    // For each punctured decoder we try both possible missing bits b0 ∈ {0,1},
    // decode in the full domain, then score the resulting punctured codeword
    // using:
    //   1. Hamming distance in the punctured metric;
    //   2. Estimated T-depth of the odd set after optimization;
    //   3. A stable hash (for deterministic tie-breaking).
    //
    // The winner is also converted to ANF and returned as a list of monomials.

    // 1) Plain Dumer (single-path) + puncturing wrapper.
    m.def("decode_rm_dumer_punctured",
          [](py::bytes w_punct_bytes, int n, int r){
              const int L = (1<<n) - 1;
              BitVec w = bytes_to_bitvec(w_punct_bytes, L);

              // Best candidate across b0 ∈ {0,1}.
              struct Best {
                  int dist;   // punctured Hamming distance
                  int depth;  // estimated T-depth
                  uint64_t key; // hash for tie-breaking
                  BitVec full;
                  BitVec punct;
                  bool have=false;
              } best;

              // Try a specific choice of the missing bit b0.
              auto consider = [&](bool b0){
                  // Form full-length observation by inserting b0.
                  BitVec y_full = embed_punctured_to_full(w, b0);

                  // Decode in the full RM code.
                  BitVec c_full = dumer_decode_full(y_full, r, n);

                  // Score the resulting punctured codeword.
                  BitVec c_punct = puncture_full(c_full);
                  BitVec diff = c_punct; diff.xor_inplace(w);         // odd set after optimization
                  int dist  = diff.weight();
                  int depth = estimate_tdepth_from_punctured(diff, n);
                  uint64_t key = hash_bitvec(c_punct);

                  // Update best candidate if this one is better.
                  if (!best.have ||
                      dist < best.dist ||
                      (dist == best.dist && (depth < best.depth ||
                                             (depth == best.depth && key < best.key)))){
                      best.have  = true;
                      best.dist  = dist;
                      best.depth = depth;
                      best.key   = key;
                      best.full  = c_full;
                      best.punct = c_punct;
                  }
              };

              consider(false);
              consider(true);

              // Extract monomials of the chosen full codeword via its ANF.
              std::vector<uint8_t> coef = anf_from_truth(best.full, n);
              std::vector<int> monoms;
              for (int mask=0; mask<(1<<n); ++mask)
                  if (coef[mask] & 1) monoms.push_back(mask);

              return py::make_tuple(
                  bitvec_to_bytes(best.punct),  // best punctured codeword
                  monoms,                       // ANF monomials
                  best.dist                     // punctured distance to input
              );
          }, py::arg("w_punct"), py::arg("n"), py::arg("r"));

    // 2) Dumer-List (beam) + puncturing wrapper.
    //    We first perform list decoding in the full code and then pick the punctured candidate that scores best.
    m.def("decode_rm_dumer_list_punctured",
          [](py::bytes w_punct_bytes, int n, int r, int list_size){
              const int L = (1<<n) - 1;
              if (list_size < 1) list_size = 1;
              BitVec w = bytes_to_bitvec(w_punct_bytes, L);

              struct Best {
                  int dist;
                  int depth;
                  uint64_t key;
                  BitVec full;
                  BitVec punct;
                  bool have=false;
              } best;

              auto consider = [&](bool b0){
                  BitVec y_full = embed_punctured_to_full(w, b0);
                  auto cands = dumer_list_decode_full(y_full, r, n, list_size);
                  for (auto& cand : cands){
                      BitVec c_punct = puncture_full(cand.code);
                      BitVec diff = c_punct; diff.xor_inplace(w);
                      int dist  = diff.weight();
                      int depth = estimate_tdepth_from_punctured(diff, n);
                      uint64_t key = hash_bitvec(c_punct);

                      if (!best.have ||
                          dist < best.dist ||
                          (dist == best.dist && (depth < best.depth ||
                                                 (depth == best.depth && key < best.key)))){
                          best.have  = true;
                          best.dist  = dist;
                          best.depth = depth;
                          best.key   = key;
                          best.full  = cand.code;
                          best.punct = c_punct;
                      }
                  }
              };

              consider(false);
              consider(true);

              std::vector<uint8_t> coef = anf_from_truth(best.full, n);
              std::vector<int> monoms;
              for (int mask=0; mask<(1<<n); ++mask)
                  if (coef[mask] & 1) monoms.push_back(mask);

              return py::make_tuple(
                  bitvec_to_bytes(best.punct),
                  monoms,
                  best.dist
              );
          }, py::arg("w_punct"), py::arg("n"), py::arg("r"), py::arg("list_size"));

    // 2b) Dumer-List TOP-K (beam), return top-K punctured candidates ordered by (distance, depth, hash).
    // Only the punctured codewords are returned, Python code can re-score them with custom metrics.
    m.def("decode_rm_dumer_list_topk_punctured",
          [](py::bytes w_punct_bytes, int n, int r, int list_size, int top_k){
              const int L = (1<<n) - 1;
              if (list_size < 1) list_size = 1;
              if (top_k < 1) top_k = 1;
              if (top_k > list_size) top_k = list_size;

              BitVec w = bytes_to_bitvec(w_punct_bytes, L);

              std::vector<ListDecodeCand> pool;
              pool.reserve(2 * std::max(1, list_size));

              auto consider = [&](bool b0){
                  BitVec y_full = embed_punctured_to_full(w, b0);
                  auto cands = dumer_list_decode_full(y_full, r, n, list_size);
                  for (auto& cand : cands){
                      BitVec c_punct = puncture_full(cand.code);
                      BitVec diff = c_punct; diff.xor_inplace(w);
                      int dist  = diff.weight();
                      int depth = estimate_tdepth_from_punctured(diff, n);
                      uint64_t key = hash_bitvec(c_punct);
                      pool.push_back(ListDecodeCand{dist, depth, key, cand.code, c_punct});
                  }
              };

              consider(false);
              consider(true);

              // Deduplicate by the punctured bytes so that we never return the same candidate more than once, even if it arises from both choices of b0.
              std::unordered_set<std::string> seen;
              std::vector<ListDecodeCand> uniq;
              uniq.reserve(pool.size());
              for (auto &c : pool){
                  auto v = c.punct.to_bytes_le();
                  std::string s(reinterpret_cast<const char*>(v.data()), v.size());
                  if (seen.insert(s).second){
                      uniq.push_back(std::move(c));
                  }
              }

              // Sort by (distance, depth, hash) and keep the top_k.
              std::sort(uniq.begin(), uniq.end(),
                        [](const ListDecodeCand& A, const ListDecodeCand& B){
                            if (A.dist != B.dist) return A.dist < B.dist;
                            if (A.depth != B.depth) return A.depth < B.depth;
                            return A.key < B.key;
                        });

              if ((int)uniq.size() > top_k) uniq.resize(top_k);

              std::vector<py::bytes> out;
              out.reserve(uniq.size());
              for (auto &c : uniq){
                  out.push_back(bitvec_to_bytes(c.punct));
              }
              return out;
          }, py::arg("w_punct"), py::arg("n"), py::arg("r"),
             py::arg("list_size"), py::arg("top_k"));

    // 3) Dumer-List + Chase (beam + reliability flips), punctured wrapper.
    //
    // The "Chase" part performs additional flips of the least reliable bits (controlled by chase_t and chase_limit) around each list candidate.
    m.def("decode_rm_dumer_list_chase_punctured",
          [](py::bytes w_punct_bytes, int n, int r, int list_size,
             int chase_t, int chase_limit){
              const int L = (1<<n) - 1;
              if (list_size < 1) list_size = 1;
              if (chase_t < 0) chase_t = 0;
              if (chase_limit < 0) chase_limit = 0;

              BitVec w = bytes_to_bitvec(w_punct_bytes, L);

              struct Best {
                  int dist;
                  int depth;
                  uint64_t key;
                  BitVec full;
                  BitVec punct;
                  bool have=false;
              } best;

              auto consider = [&](bool b0){
                  BitVec y_full = embed_punctured_to_full(w, b0);
                  auto cands = dumer_list_chase_full(
                      y_full, r, n, list_size, chase_t, chase_limit);
                  for (auto& cand : cands){
                      BitVec c_punct = puncture_full(cand.code);
                      BitVec diff = c_punct; diff.xor_inplace(w);
                      int dist  = diff.weight();
                      int depth = estimate_tdepth_from_punctured(diff, n);
                      uint64_t key = hash_bitvec(c_punct);
                      if (!best.have ||
                          dist < best.dist ||
                          (dist == best.dist && (depth < best.depth ||
                                                 (depth == best.depth && key < best.key)))){
                          best.have  = true;
                          best.dist  = dist;
                          best.depth = depth;
                          best.key   = key;
                          best.full  = cand.code;
                          best.punct = c_punct;
                      }
                  }
              };

              consider(false);
              consider(true);

              std::vector<uint8_t> coef = anf_from_truth(best.full, n);
              std::vector<int> monoms;
              for (int mask=0; mask<(1<<n); ++mask)
                  if (coef[mask] & 1) monoms.push_back(mask);

              return py::make_tuple(
                  bitvec_to_bytes(best.punct),
                  monoms,
                  best.dist
              );
          }, py::arg("w_punct"), py::arg("n"), py::arg("r"),
             py::arg("list_size"), py::arg("chase_t"), py::arg("chase_limit"));

    // RPA-1 decoders (Recursive Projection-Aggregation, order-1)

    // RPA-1 on a full truth table.  This is a more advanced decoder that projects onto smaller subspaces, aggregates estimates, and then performs a final list decoding step of size final_list_size around the seed.
    m.def("decode_rm_rpa1_full",
          [](py::bytes y_full_bytes, int n, int r, int iters, int final_list_size){
              BitVec y = bytes_to_bitvec(y_full_bytes, 1<<n);
              BitVec c = rpa1_seed_full(y, r, n, iters, final_list_size);
              return bitvec_to_bytes(c);
          }, py::arg("y_full"), py::arg("n"), py::arg("r"),
             py::arg("iters"), py::arg("final_list_size"));

    // RPA-1: punctured wrapper.  For each b0 ∈ {0,1} we:
    //   1. Run RPA-1 around y_full.
    //   2. Run a baseline Dumer list decoder of the same list size.
    //   3. Score all resulting full codewords in the punctured metric.
    m.def("decode_rm_rpa1_punctured",
      [](py::bytes w_punct_bytes, int n, int r, int iters, int final_list_size){
          const int L = (1<<n) - 1;
          BitVec w = bytes_to_bitvec(w_punct_bytes, L);
          const int ls = std::max(1, final_list_size);

          struct Best {
              int dist;
              int depth;
              uint64_t key;
              BitVec full;
              BitVec punct;
              bool have=false;
          } best;

          // Score a full-length codeword in the punctured metric and update
          // the current best candidate.
          auto score_and_update = [&](const BitVec& c_full){
              BitVec c_punct = puncture_full(c_full);
              BitVec diff = c_punct; diff.xor_inplace(w);
              int dist  = diff.weight();
              int depth = estimate_tdepth_from_punctured(diff, n);
              uint64_t key = hash_bitvec(c_punct);
              if (!best.have ||
                  dist < best.dist ||
                  (dist == best.dist && (depth < best.depth ||
                                         (depth == best.depth && key < best.key)))){
                  best.have  = true;
                  best.dist  = dist;
                  best.depth = depth;
                  best.key   = key;
                  best.full  = c_full;
                  best.punct = c_punct;
              }
          };

          // Try a particular choice of the missing coordinate b0.
          auto consider = [&](bool b0){
              // Full-length observation for this puncturing.
              BitVec y_full = embed_punctured_to_full(w, b0);

              // RPA candidate (around the seeded estimate).
              BitVec c_rpa = rpa1_seed_full(y_full, r, n, iters, ls);
              score_and_update(c_rpa);

              // Baseline beam around y_full (superset guarantee in punctured metric).
              auto beam_y = dumer_list_decode_full(y_full, r, n, ls);
              for (const auto& cand : beam_y){
                  score_and_update(cand.code);
              }
          };

          consider(false);
          consider(true);

          // Convert the chosen full codeword to its ANF monomials.
          std::vector<uint8_t> coef = anf_from_truth(best.full, n);
          std::vector<int> monoms;
          for (int mask=0; mask<(1<<n); ++mask)
              if (coef[mask] & 1) monoms.push_back(mask);

          return py::make_tuple(
              bitvec_to_bytes(best.punct),
              monoms,
              best.dist
          );
      }, py::arg("w_punct"), py::arg("n"), py::arg("r"),
         py::arg("iters"), py::arg("final_list_size"));

    // T-depth helpers

    // Estimate the T-depth of a linear-phase circuit corresponding to the given odd set, represented in punctured form.
    m.def("tdepth_from_punctured",
          [](py::bytes odd_punct_bytes, int n){
              const int L = (1<<n) - 1;
              BitVec odd = bytes_to_bitvec(odd_punct_bytes, L);
              return estimate_tdepth_from_punctured(odd, n);
          }, py::arg("odd_punct"), py::arg("n"));

    // Exact / colour-improved scheduler from a set of monomials.
    //
    // Input:
    //   monoms - list of n-bit masks describing the monomials to implement.
    //   n      - number of variables/qubits.
    //   budget - optional cap on allowed depth (negative => no cap).
    //
    // Output:
    //   (depth, layers) where layers is a list of layers, each layer being
    //   a list of monomials that can be executed in parallel.
    m.def("tdepth_schedule_from_monoms",
          [](const std::vector<int>& monoms, int n, int budget){
              std::vector<std::vector<int>> layers;
              int depth = schedule_tdepth_from_monoms(monoms, n, budget, layers);
              return py::make_tuple(depth, layers);
          }, py::arg("monoms"), py::arg("n"), py::arg("budget") = -1);
}
