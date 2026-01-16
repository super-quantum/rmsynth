#ifndef RM_SYNTH_RPA_H
#define RM_SYNTH_RPA_H

#pragma once
#include "bitops.h"
#include "dumer.h"
#include "dumer_list.h"

// RPA-1 seeding (Recursive Projection–Aggregation, order 1) followed by a final Dumer or Dumer-List decoder.
//
// INPUT:
//   y    : noisy truth table of length 2^n.
//   r,n  : RM(r,n) parameters.
//   iters : number of RPA-1 iterations of projection/aggregation.
//   final_list_size :
//        = 1  -> final step is a single-path Dumer decoder.
//        > 1  -> final step uses Dumer-List of size final_list_size around both the RPA estimate and the original y, and picks the closest candidate to y.
//
// OUTPUT:
//   An RM(r,n) codeword (BitVec of length 2^n) that serves as a strong candidate for further processing.
BitVec rpa1_seed_full(const BitVec& y, int r, int n, int iters, int final_list_size);

#endif //RM_SYNTH_RPA_H
