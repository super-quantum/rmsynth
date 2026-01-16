#include "anf.h"

// Compute ANF coefficients from the truth table using a fast Möbius transform.
// The algorithm is essentially the standard zeta transform over the Boolean lattice, specialized to F2.
std::vector<uint8_t> anf_from_truth(const BitVec& y, int n){
    const int N = 1<<n;

    // Initialize ANF coefficient array with the truth table values.
    // After the Möbius transform below, a[mask] will contain the ANF coefficient for monomial "mask".
    std::vector<uint8_t> a(N, 0);
    for (int i=0;i<N;i++) a[i] = y.get(i) ? 1 : 0;

    // Möbius transform over the Boolean cube:
    //
    // For each variable d, we propagate contributions from subsets that do not contain d to those that do.
    // Over F2 this turns point evaluations into ANF coefficients.
    //
    // After processing all dimensions, "a" holds the ANF coefficients.
    for (int d=0; d<n; ++d){
        for (int mask=0; mask<N; ++mask){
            if (mask & (1<<d)) a[mask] ^= a[mask ^ (1<<d)];
        }
    }
    return a;
}
