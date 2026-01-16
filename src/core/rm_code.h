#pragma once
#include <vector>
#include <cstdint>

// Generator rows for the (punctured) Reed–Muller code RM(r,n).
//
// The generator matrix has one row for each monomial of degree ≤ r in n variables.  Each row is returned as a uint64_t whose lower L = 2^n - 1 bits encode the values of that monomial on all nonzero inputs y ∈ {1, 2, ..., 2^n - 1}.
//
// Arguments:
//   n, r   : RM(r,n) parameters.
//   monoms : output vector that will be filled with the monomial masks
//            (integers in [0, 2^n)) corresponding to each row.
//
// Note: This representation is used together with decode_rm_bruteforce() and matches the punctured representation used by the optimizer.
std::vector<uint64_t> rm_generator_rows(int n, int r, std::vector<int>& monoms);

// Dimension of RM(r,n) = sum_{d=0}^r C(n, d), or 0 for r < 0.
int rm_dimension(int n, int r);
