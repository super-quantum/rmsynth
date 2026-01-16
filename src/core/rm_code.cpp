#include "rm_code.h"
#include <vector>
#include <cstdint>

// Small helper: Hamming weight of a 32-bit integer.
static inline int weight(unsigned x){ return __builtin_popcount(x); }

// Build generator rows for (punctured) RM(r,n).
//
// The generator matrix corresponds to evaluating all monomials of degree ≤ r over the domain {1, 2, ..., 2^n - 1}, i.e., all nonzero inputs.
// This matches the punctured RM code used elsewhere in the project.
//
// Parameters:
//   n, r   : RM(r,n) parameters.
//   monoms : output; cleared and filled with the integer masks t labeling
//            each generator row (t encodes which variables appear).
//
// Returns:
//   A vector of uint64_t, one per generator row.  Row j has length
//   L = 2^n - 1 bits packed into the lower L bits of the uint64_t.
//   Bit (y-1) is the value of the monomial with mask `monoms[j]` on input y.
std::vector<uint64_t> rm_generator_rows(int n, int r, std::vector<int>& monoms){
    const int L = (1<<n) - 1;        // length of punctured truth table
    std::vector<uint64_t> rows;
    monoms.clear();

    // Constant monomial (degree 0), present for r >= 0.
    if (r >= 0){
        monoms.push_back(0);         // mask 0 => constant 1
        uint64_t bits=0;
        for(int j=0;j<L;++j) bits |= (1ull<<j);  // 1 on all nonzero inputs
        rows.push_back(bits);
    }

    // Non-constant monomials: masks t ∈ {1,..,2^n-1} of weight ≤ r.
    for(int t=1; t<(1<<n); ++t){
        if (weight(t) <= r){
            monoms.push_back(t);
            uint64_t bits=0;
            // For each nonzero input y ∈ {1,..,L}, the monomial evaluates to 1 iff all variables present in t are also 1 in y, i.e. (t & y) == t.
            for(int y=1; y<=L; ++y) if ((t & y) == t) bits |= (1ull<<(y-1));
            rows.push_back(bits);
        }
    }
    return rows;
}

// Compute the dimension of the Reed–Muller code RM(r,n).
//
// Dimension is sum_{d=0}^r C(n, d), with the convention that RM(r,n) is the zero code when r < 0.
int rm_dimension(int n, int r){
    if (r < 0) return 0;
    long long s=0, c=1;
    for(int d=0; d<=r; ++d){
        // Incrementally compute C(n, d) = C(n, d-1) * (n-d+1) / d.
        c = (d==0) ? 1 : c*(n-d+1)/d;
        s += c;
    }
    return (int)s;
}
