#include "decode_bruteforce.h"
#include "rm_code.h"
#include "util.h"
#include <limits>

// Exhaustive decoder for RM(r,n).  This walks through all 2^k possible coefficient vectors u (where k is the dimension of RM(r,n)), forms the
// corresponding codeword, and picks the one with minimum Hamming distance to the received word w_bits.
DecodeResult decode_rm_bruteforce(uint64_t w_bits, int n, int r){
    DecodeResult out; out.ties = 0;

    // For r < 0 the code contains only the all-zero word.
    if (r < 0){ out.codeword = 0; return out; }

    std::vector<int> monoms;

    // Get the generator matrix rows of RM(r,n) as packed 64-bit words.
    // "monoms" holds the monomial masks used for each row.
    auto rows = rm_generator_rows(n, r, monoms);
    const int m = (int)rows.size();  // generator matrix has m rows

    uint64_t best_word = 0;
    int      best_dist = std::numeric_limits<int>::max();
    int      best_u    = 0;          // coefficient vector that gave best_word

    // Enumerate all subsets of generator rows.  u is the vector of ANF coefficients in {0,1}^m, cw is the resulting codeword.
    for (int u=0; u < (1<<m); ++u){
        uint64_t cw = 0;
        for (int i=0;i<m;++i)
            if ((u>>i)&1) cw ^= rows[i];   // accumulate selected rows (XOR over F2)

        int dist = popcount64(w_bits ^ cw); // Hamming distance to received word

        // Keep the best candidate, with deterministic tie-breaking: we prefer smaller distance, and among equal-distance codewords we prefer the lexicographically smaller packed word.
        if (dist < best_dist || (dist == best_dist && cw < best_word)){
            out.ties = (dist == best_dist) ? (out.ties+1) : 1;
            best_dist = dist;
            best_word = cw;
            best_u    = u;
        }
    }

    out.codeword = best_word;

    // Decode the chosen coefficient vector back into a list of monomial masks.
    for (int i=0;i<m;++i)
        if ((best_u>>i)&1) out.monomials.push_back(monoms[i]);

    return out;
}
