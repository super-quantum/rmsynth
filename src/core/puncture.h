#ifndef RM_SYNTH_PUNCTURE_H
#define RM_SYNTH_PUNCTURE_H

#pragma once
#include "bitops.h"

// Embed a punctured word (length 2^n - 1) into a full word (length 2^n) by inserting a bit b0 at position 0.
//
// Convention:
//   * Index 0 corresponds to input x = 0^n (all-zero input).
//   * The punctured representation omits this coordinate and stores only the remaining 2^n - 1 truth-table entries (indices 1..2^n-1) packed as positions 0..(2^n-2) in `punct`.
inline BitVec embed_punctured_to_full(const BitVec& punct, bool b0){
    BitVec full(punct.nbits + 1);
    if (b0) full.set1(0);
    for (int i=0;i<punct.nbits;i++){
        if (punct.get(i)) full.set1(i+1);
    }
    return full;
}

// Puncture a full word of length 2^n by dropping index 0 (the 0^n input).
//
// The result is a word of length 2^n - 1 where bit i corresponds to original bit (i+1).
inline BitVec puncture_full(const BitVec& full){
    BitVec punct(full.nbits - 1);
    for (int i=0;i<punct.nbits;i++){
        if (full.get(i+1)) punct.set1(i);
    }
    return punct;
}

#endif //RM_SYNTH_PUNCTURE_H
