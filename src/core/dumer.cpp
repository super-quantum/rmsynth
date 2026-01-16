#include "dumer.h"
#include <algorithm>

// Utility: return an N-bit vector filled with ones.
static inline BitVec all_ones(int N){
    BitVec b(N);
    for (int i=0;i<N;i++) b.set1(i);
    return b;
}

// Recursive Dumer decoder for full RM(r,n) using the Plotkin decomposition.
//
// At each level we view the length-N word y as a concatenation y = (y0, y1) and use the identity
//
//   RM(r,n) = { (u, u+v) : u ∈ RM(r,n-1), v ∈ RM(r-1,n-1) }.
//
// We first decode an estimate v̂ from y0 ⊕ y1 in RM(r-1,n-1), then decode û from y0 ⊕ v̂ in RM(r,n-1), and finally return (û, û ⊕ v̂).
BitVec dumer_decode_full(const BitVec& y, int r, int n){
    const int N = 1<<n;

    // Handle trivial parameter regimes explicitly.
    if (r < 0) return BitVec(N);              // RM(r,n) = {0} for r < 0
    if (r >= n) return y;                     // RM(r,n) = F2^N for r ≥ n
    if (n == 0) return BitVec(1);             // length-1 code, only 0 if r<n

    if (r == 0){
        // RM(0,n) contains only the two constant codewords 0^N and 1^N.
        // Choose whichever is closer in Hamming distance to y.
        int wt = y.weight();
        return (wt <= N - wt) ? BitVec(N) : all_ones(N);
    }

    // Plotkin decomposition:
    //   y = (y0, y1) with length N/2 halves
    //   y_sum = y0 ⊕ y1 is a noisy observation of v
    const int N2 = N>>1;
    BitVec y0   = y.slice(0,   N2);
    BitVec y1   = y.slice(N2,  N2);
    BitVec ysum = y0; ysum.xor_inplace(y1);

    // Step 1: decode v̂ ∈ RM(r-1, n-1) from the sum.
    BitVec v_hat = dumer_decode_full(ysum, r-1, n-1);

    // Step 2: decode û ∈ RM(r, n-1) from y0 ⊕ v̂.
    BitVec tmp = y0; tmp.xor_inplace(v_hat);
    BitVec u_hat = dumer_decode_full(tmp, r, n-1);

    // Step 3: assemble (û, û ⊕ v̂) ∈ RM(r,n).
    BitVec u_plus_v = u_hat; u_plus_v.xor_inplace(v_hat);
    return BitVec::concat(u_hat, u_plus_v);
}
