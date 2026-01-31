# 9. Caching, contracts, and reports

## 9.1 Caching RM data (`rm_cache.py`)

Many parts of the Python layer repeatedly need the same Reed–Muller generator matrix and the list of monomial masks for a given pair $(n,r)$. Computing these from scratch every time would be wasteful, especially when decoders and SNAP routines are called in loops.

`rm_cache.py` provides a very small but important cache:

* `get_rows(n, r)`: returns the punctured generator rows for $\mathrm{RM}(r,n)$ as Python integers, where bit $i$ corresponds to column $i$ in punctured order. Internally it just calls `rm_generator_rows(n, r)` from `core.py`, but the result is wrapped in an `@lru_cache(maxsize=16)` so the same $(n,r)$ pair is only computed once.
* `get_monom_list(n, r)`: returns the monomial masks in the same order as the rows: `[0]` for the constant term (if $r \ge 0$), then all masks with `popcount(mask) <= r` in increasing numeric order.

This ordering is crucial. It must match both the Python generator and the C++ generator in `rm_code.cpp`. The decoder, SNAP, and OSD routines all rely on this alignment when mapping between “basis index”, “monomial mask”, and “generator row”.

The invariants for this module are simple but strict:
1.  For fixed $(n,r)$, `get_rows` and `get_monom_list` never change during the lifetime of the process.
2.  Index `i` in `get_monom_list(n,r)` is the monomial whose generator row is `get_rows(n,r)[i]`.

If either of these were violated, contracts and decoders would start failing in subtle ways. The cache ensures consistency and avoids recomputing expensive combinatorial structures.

## 9.2 Contracts (`contracts.py`)

The `contracts.py` module is a small “safety net” for the optimization pipeline. It encodes three conceptual contracts that any decoder must satisfy when plugged into `Optimizer`:



1.  The decoded monomials must lie in $\mathrm{RM}(r,n)$.
2.  The decoded codeword bits must equal the XOR of generator rows for those monomials.
3.  The reported punctured distance must match the T-count after lifting and re-synthesis.

The key helpers are:

`assert_in_rm(selected_monomials, n, r)`

For each mask `m` in `selected_monomials`, this recomputes `popcount(m)` and checks it is at most `r`. If any monomial has degree greater than `r`, an `AssertionError` is raised. This catches bugs where a decoder (or post‑processing step) accidentally produces an invalid RM basis vector.

`assert_code_consistency(code_bits, selected_monomials, n, r)`

This reconstructs the punctured codeword by XOR’ing generator rows corresponding to `selected_monomials`. It uses `get_rows` and `get_monom_list` to map masks to row indices, then compares the reconstructed bit list to the `code_bits` returned by the decoder. If they differ, it reports the first few mismatching indices to help debugging. This enforces that “basis coefficients” and “codeword” are consistent views of the same RM codeword.

`assert_distance_equals_tcount(vec_mod8, code_bits, n, r, reported_dist)`

Here `vec_mod8` is the original phase vector mod 8. The oddness of the original vector is `w_bits`, and after applying the decoded codeword we conceptually move from `w_bits` to `w_bits XOR code_bits`. Contract 3 is that the reported punctured distance equals the number of odd entries in the final coefficients, i.e. the T‑count after optimization. This helper recomputes `after_odd = w_bits ^ code_bits` and checks that `sum(after_odd) == reported_dist`. It deliberately stays in the “bit” world; it doesn’t try to re-synthesize a circuit, just checks the T-count arithmetic.

On top of these, there are two orchestration functions:

* `check_all(...)` runs all three contracts and raises an `AssertionError` immediately on failure.
* `maybe_check_all(...)` calls `check_all` only if `_checks_enabled()` returns `True`. That function reads the environment variable `RM_CHECKS`. When `RM_CHECKS` is unset or empty, checks are OFF by default; tests and debug runs can enable them by setting `RM_CHECKS=1` (or any truthy string), and production runs can leave them off.

The `Optimizer` uses `check_all` or `maybe_check_all` at the end of `optimize()`, so every optimization pass can be verified against the decoder output without changing the public API.

## 9.3 Reports (`report.py`)

Two simple dataclasses encapsulate the structured output of decoding and optimization:

`DecodeReport` is mostly used in tests and low-level experiments. It records:

* `n`, `r`, `length`, `dimension` – parameters of RM(r,n) and the punctured length.
* `distance` – Hamming distance between the input oddness and the decoded codeword.
* `selected_monomials` – list of monomial masks used in the candidate.
* `codeword_bits` – the codeword as a single integer (little-endian bit packing).
* `ties` – how many codewords tied for minimum distance in brute‑force decoding.

`OptimizeReport` is what `Optimizer.optimize()` returns together with the new circuit:

* `n`, `r` – problem size and RM order used.
* `before_t`, `after_t` – T-counts before and after optimization.
* `distance` – punctured distance of the chosen codeword.
* `bitlen` – length of the punctured vector ($2^n - 1$).
* `selected_monomials` – monomials actually used for the correction.
* `signature` – SHA‑256 hash (hex) of the optimized coefficient vector.

The `summary()` method produces a human-readable line like:
> `[rmsynth] n=6, r=2, length=63: T-count 27 -> 9 (distance=9). Signature=abcd1234ef567890...`

The CLI prints this summary for quick feedback, and tests use it to verify both the numerical improvement (T-count) and the bit-level signature when checking for regressions.