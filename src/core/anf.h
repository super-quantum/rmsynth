#ifndef RM_SYNTH_ANF_H
#define RM_SYNTH_ANF_H

#pragma once
#include <vector>
#include "bitops.h"

// ANF utilities
//
// A Boolean function f : {0,1}^n -> {0,1} can be written in its algebraic normal form (ANF)
//
//   f(x) = ⊕_mask a[mask] * ∏_{i : mask_i = 1} x_i , where ⊕ is XOR over F2, and each "mask" is an n‑bit index that selects which variables appear in the monomial.
//
// In this file we provide a helper for converting from a truth table representation to the vector of ANF coefficients a[mask].

// Compute ANF (algebraic normal form) coefficients for a Boolean function given by its full truth table.
//
// Input:
//   y  - truth table of f: {0,1}^n -> {0,1}, stored as a BitVec of length 2^n.
//        Entry y.get(i) is the function value on the input whose binary encoding is i (LSB first).
//   n  - number of input variables (so y.size() should equal 2^n).
//
// Output:
//   Vector a of length 2^n with entries in {0,1}.  a[mask] is the coefficient of the monomial ∏_{j : mask_j = 1} x_j in the ANF of f.
//
// Implementation detail:
//   The conversion is done in-place using a fast Möbius transform on the Boolean cube.

std::vector<uint8_t> anf_from_truth(const BitVec& y, int n);

#endif //RM_SYNTH_ANF_H
