#ifndef RM_SYNTH_DUMER_LIST_H
#define RM_SYNTH_DUMER_LIST_H

#pragma once
#include <vector>
#include "bitops.h"


// List (beam) decoder based on Dumer's recursion.
//
// Instead of returning a single estimate, the decoder keeps up to `list_size` candidates at each recursion level and returns the best few at the end.

// Candidate codeword for RM(r,n) together with its Hamming distance to the input word that was decoded.
struct DL_Candidate {
    BitVec code;  // length-N codeword (N = 2^n)
    int    dist;  // Hamming distance to the input word
};

// List decoder for RM(r,n) based on the Dumer recursion.
//
// Input:
//   y         - received word / truth table of length N = 2^n.
//   r, n      - RM parameters.
//   list_size - maximum number of candidates to keep.
//
// Output:
//   A vector of up to list_size candidates sorted by increasing distance.
//   Ties are broken deterministically when the environment variable
//   RM_DETERMINISTIC is set (see dumer_list.cpp for details).
std::vector<DL_Candidate>
dumer_list_decode_full(const BitVec& y, int r, int n, int list_size);

#endif //RM_SYNTH_DUMER_LIST_H
