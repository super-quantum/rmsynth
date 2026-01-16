#pragma once
#include <vector>
#include <cstdint>
#include <utility>
#include <limits>
#include "bitops.h"

// Simple T-depth estimator (backwards compatible with older versions).
//
// odd_punct : punctured odd-set mask (length 2^n - 1).
// n         : number of variables/qubits (currently unused).
//
// Returns an upper bound on the T-depth needed to implement the odd set.
int estimate_tdepth_from_punctured(const BitVec& odd_punct, int n);

/**
 * exact/color‑improved scheduler for parity gadgets (monomials)
 *
 * INPUT:
 *   monoms : list of monomial masks (bitset over n vars).  mask==0 (constant)
 *            is ignored, since constants do not require T gates.
 *   n      : number of variables/qubits.
 *   budget : optional target depth. If >0, the search may early‑stop on any
 *            solution with depth <= budget. If <=0, it attempts to find a
 *            minimum (subject to internal node/time caps).
 *
 * OUTPUT:
 *   layers : if non-empty on return, contains 'depth' color classes, each
 *            is a vector of monomial masks placed at that T layer.
 *
 * RETURNS:
 *   depth  : best depth found (<=|monoms|). Guaranteed <= greedy coloring.
 *
 * Notes:
 *   conflict  : two monomials conflict if their supports overlap, i.e.
 *               (mi & mj) != 0. This captures the no‑ancilla parity network
 *               layering constraint.
 *   algorithm : DSATUR with branch‑and‑bound (exact for our sizes), seeded
 *               by a greedy coloring.
 *   safety    : time/node caps (env‑configurable) ensure practical running
 *               time. If the exact search aborts, we return the greedy
 *               coloring (still a valid schedule).
 */
int schedule_tdepth_from_monoms(const std::vector<int>& monoms,
                                int n,
                                int budget,
                                std::vector<std::vector<int>>& layers);
