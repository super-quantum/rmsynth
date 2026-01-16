#pragma once
#include <vector>
#include <cstdint>

// Brute-force decoder for small Reed–Muller codes RM(r,n)
//
// This is an exhaustive maximum-likelihood decoder that enumerates all codewords of RM(r,n) (via a generator matrix) and picks the closest one to a given received word of length N = 2^n.
//
// The implementation represents codewords packed into the low N bits of a uint64_t, so it is only intended for very small instances (e.g. n ≤ 6).
// It is mainly used for testing and as a simple reference decoder.

struct DecodeResult{
    // Closest codeword in RM(r,n), packed into the low 2^n bits of this word.
    // Bit ordering is consistent with rm_generator_rows().
    uint64_t codeword;

    // Monomials (as n-bit masks) corresponding to the chosen codeword.
    // These match the order of the generator rows returned by rm_generator_rows().
    std::vector<int> monomials;

    // Number of codewords that attain the minimum Hamming distance.
    // The decoder always picks a canonical one (lexicographically smallest packed codeword), but we keep this count for diagnostics.
    int ties;
};

// Brute-force maximum-likelihood decoding for RM(r,n) on up to 64 bits.
//
// Input:
//   w_bits - received word packed into the low 2^n bits of a uint64_t.
//   n      - number of variables (block length N = 2^n).
//   r      - order of the Reed–Muller code.
//
// Output:
//   A DecodeResult with the closest codeword, its generating monomials, and the number of ties.
//
// Complexity:
//   O(2^k * N) where k is the RM dimension.  This grows very quickly with n, so in practice this is only used for very small codes or unit tests.
DecodeResult decode_rm_bruteforce(uint64_t w_bits, int n, int r);
