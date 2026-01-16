#ifndef RM_SYNTH_BITOPS_H
#define RM_SYNTH_BITOPS_H

#pragma once
#include <vector>
#include <cstdint>
#include <cassert>
#include <algorithm>

// Lightweight bit-vector utilities.
//
// BitVec is a simple dynamically-sized bitset backed by a std::vector<uint64_t>.
// Bits are numbered from 0 to nbits-1.  Bit i lives in word w[i>>6] at position (i & 63).  All operations assume little-endian word ordering.
//
// These helpers are used throughout the Reed–Muller decoding code to represent truth tables and codewords of length up to 2^n.

struct BitVec {
    int nbits;                  // number of meaningful bits
    std::vector<uint64_t> w;    // storage (ceil(nbits / 64) words)

    BitVec(): nbits(0) {}

    explicit BitVec(int n): nbits(n), w((n+63)>>6, 0ull) {}

    // Number of bits in this vector.
    int size() const { return nbits; }

    // Set all bits to zero.
    void clear(){ std::fill(w.begin(), w.end(), 0ull); }

    // Construct a BitVec from a little-endian byte array.
    //
    // bytes[0] contains bits 0..7, bytes[1] contains bits 8..15, etc.
    // Only the first `nbits` bits are read (extra bits are ignored).
    static BitVec from_bytes_le(const std::vector<uint8_t>& bytes, int nbits){
        BitVec b(nbits);
        for (int i=0;i<nbits;i++){
            int byte = i>>3;
            int bit  = i & 7;
            if (byte < (int)bytes.size() && ((bytes[byte]>>bit)&1))
                b.set1(i);
        }
        return b;
    }

    // Export this BitVec as a little-endian byte array.
    //
    // The resulting vector has ceil(nbits / 8) bytes, with the same layout
    // convention as from_bytes_le().
    std::vector<uint8_t> to_bytes_le() const {
        std::vector<uint8_t> out((nbits+7)>>3, 0);
        for (int i=0;i<nbits;i++){
            if (get(i)) out[i>>3] |= (1u<<(i&7));
        }
        return out;
    }

    // Test bit i (0-based indexing).
    inline bool get(int i) const { return (w[i>>6] >> (i&63)) & 1ull; }

    // Set bit i to 1.
    inline void set1(int i){ w[i>>6] |= (1ull<<(i&63)); }

    // Set bit i to 0.
    inline void set0(int i){ w[i>>6] &= ~(1ull<<(i&63)); }

    // Flip bit i.
    inline void toggle(int i){ w[i>>6] ^= (1ull<<(i&63)); }

    // Hamming weight (number of 1 bits) of this vector.
    int weight() const {
        int s=0;
        const std::size_t M = w.size();
        #if defined(_OPENMP)
        #pragma omp simd reduction(+:s)
        #endif
        for (std::size_t i=0;i<M;++i){
            uint64_t x = w[i];
            #if defined(__GNUG__) || defined(__clang__)
                // Use builtin popcount when available.
                s += __builtin_popcountll(x);
            #else
                // Portable fallback: clear lowest set bit repeatedly.
                while (x){ x &= (x-1); ++s; }
            #endif
        }
        return s;
    }

    // In-place XOR with another BitVec of the same length.
    void xor_inplace(const BitVec& other){
        assert(nbits == other.nbits);
        const std::size_t M = w.size();
        #if defined(_OPENMP)
        #pragma omp simd
        #endif
        for (std::size_t i=0;i<M;++i) w[i] ^= other.w[i];
    }

    // Concatenate two BitVecs: out = a || b.
    static BitVec concat(const BitVec& a, const BitVec& b){
        BitVec out(a.nbits + b.nbits);
        for (int i=0;i<a.nbits;i++) if (a.get(i)) out.set1(i);
        for (int i=0;i<b.nbits;i++) if (b.get(i)) out.set1(a.nbits + i);
        return out;
    }

    // Slice out a contiguous range of bits [start, start+len).
    BitVec slice(int start, int len) const {
        BitVec out(len);
        for (int i=0;i<len;i++) if (get(start+i)) out.set1(i);
        return out;
    }
};


// Hamming distance utilities

// SIMD-friendly Hamming distance on 64-bit words.
//
// Given two arrays a[0..M-1], b[0..M-1], return the total number of differing bits.
// This is the core primitive used by the decoders to score candidates.
inline int hamming_distance_words(const uint64_t* a, const uint64_t* b, std::size_t M){
    int s=0;
    #if defined(_OPENMP)
    #pragma omp simd reduction(+:s)
    #endif
    for (std::size_t i=0;i<M;++i){
        uint64_t x = a[i] ^ b[i];
        #if defined(__GNUG__) || defined(__clang__)
            s += __builtin_popcountll(x);
        #else
            while (x){ x &= (x-1); ++s; }
        #endif
    }
    return s;
}

// Hamming distance between two BitVecs of the same length.
inline int hamming_distance(const BitVec& a, const BitVec& b){
    assert(a.nbits == b.nbits);
    return hamming_distance_words(a.w.data(), b.w.data(), a.w.size());
}

#endif //RM_SYNTH_BITOPS_H
