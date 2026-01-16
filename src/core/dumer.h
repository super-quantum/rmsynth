#ifndef RM_SYNTH_DUMER_H
#define RM_SYNTH_DUMER_H

#pragma once
#include <vector>
#include <cstdint>
#include "bitops.h"


// Dumer's recursive hard-decision decoder for full RM(r,n) codes.
//
// The decoder operates directly on full truth tables of length N = 2^n, represented as BitVecs.  It is based on the standard Plotkin decomposition
// RM(r,n) = { (u, u+v) : u ∈ RM(r,n-1), v ∈ RM(r-1,n-1) }


// Dumer recursive hard-decision decoder for RM(r,n) over F2.
//
// Input:
//   y_full - received word / truth table of length N = 2^n (BitVec).
//   r      - RM order.
//   n      - number of variables (so the block length is 2^n).
//
// Output:
//   A BitVec of length N that is a codeword in RM(r,n) and is (heuristically) close to y_full in Hamming distance.
//
// Note: This is a classic deterministic decoder, it is not the list / beam variant (see dumer_list.* for that).
BitVec dumer_decode_full(const BitVec& y_full, int r, int n);

#endif //RM_SYNTH_DUMER_H
