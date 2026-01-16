#pragma once
#include <cstdint>

// Portable popcount for 64-bit words.
//
// On GCC/Clang we use the built-in 64-bit popcount, otherwise we fall back to a simple loop that repeatedly clears the lowest set bit.
inline int popcount64(uint64_t x){
#if defined(__GNUG__)
    return __builtin_popcountll(x);
#else
    int c=0; while(x){ x&=(x-1); ++c;} return c;
#endif
}
