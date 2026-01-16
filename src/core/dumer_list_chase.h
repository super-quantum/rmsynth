#ifndef RM_SYNTH_DUMER_LIST_CHASE_H
#define RM_SYNTH_DUMER_LIST_CHASE_H

#pragma once
#include <vector>
#include "bitops.h"
#include "dumer_list.h"

// Dumer-List decoder with a top-level Chase-style refinement.
//
// Starting from the baseline list produced by dumer_list_decode_full(), we identify positions where the best candidate disagrees with the input word and apply small "flip patterns" (size 1..chase_t) at those positions.
// For each modified input we run a single-candidate list decode, collect candidates, deduplicate them, and finally keep the best list_size by Hamming distance to the ORIGINAL input.
//
// Parameters:
//   y           : observed word / truth table (length 2^n).
//   r, n        : RM(r,n) parameters.
//   list_size   : final number of candidates to return.
//   chase_t     : maximum size of flip patterns (usually 1 or 2).
//   chase_limit : limit on how many flip patterns are explored in total.
std::vector<DL_Candidate>
dumer_list_chase_full(const BitVec& y, int r, int n,
                      int list_size, int chase_t, int chase_limit);

#endif //RM_SYNTH_DUMER_LIST_CHASE_H
