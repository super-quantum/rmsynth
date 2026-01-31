# 8. OSD module (`osd.py`)

The `osd.py` module implements a specialized Ordered Statistics Decoding (OSD) layer on top of punctured RM codes. It is used in two ways:
1.  as a stand‑alone decoder (`osd1`, `osd2`, `osd3` strategies in `decoders.py`), and
2.  as a local refinement around beam/RPA solutions (e.g. `beam-osd1`, or the OSD‑L1 refinement inside `"rpa-adv"`).

It is depth‑aware: when two candidates have the same Hamming distance, it uses the T‑depth estimator from `rmcore` to break ties in favor of shallower odd sets.



## 8.1 Bit packing helpers

The module works with both integer and list representations of punctured words, so it includes small helpers for moving between them:

* `_bits_to_int_le(bits)` packs a list of bits `[b0, b1, ..., b_{L-1}]` into an integer with bit `i` at position `i`.
* `_int_to_bits_le(x, L)` unpacks an integer into a length‑`L` list of bits in the same little‑endian convention.
* `_pack_bits_le(bits)` packs a list of bits into a `bytes` object, again little‑endian by bit position and containing `(L+7)//8` bytes. This is used when calling `rmcore.tdepth_from_punctured` on the residual odd set.

These conventions match what the C++ core and the rest of the Python library use, so you can reuse these helpers elsewhere if needed.

## 8.2 Building generator rows and columns

OSD needs both row‑wise and column‑wise views of the generator matrix.

`_rows_int(n, r)` simply calls `rm_cache.get_rows(n, r)` and returns a list of `K` row integers, each with `L` bits representing the punctured RM generator matrix $G$ for $\mathrm{RM}(r,n)^*$.

`_monoms(n, r)` returns the corresponding list of monomial masks in the same order, via `rm_cache.get_monom_list`.

`_columns_from_rows(rows, L, K)` converts the row‑oriented generator into a list of column vectors:

* each column is an integer with `K` bits, one bit per row,
* column `j`’s bit `i` is 1 exactly when `rows[i]` has bit `j` set.
  This column view is central for selecting information sets and for doing the linear algebra over $\mathbb{F}_2$ that OSD needs.

## 8.3 Information set selection

The function `_select_info_set(w_bits, c_bits, n, r, rows, cols)` is responsible for choosing an **information set**: a set of `K` columns that are linearly independent and “reliable” with respect to the received word and a baseline codeword.

For each column index `j`, a reliability triple is computed:

1.  `agree = 1` if `w_bits[j] == c_bits[j]`, 0 otherwise;
2.  `tiew = _column_weight_by_mask_weight(n, r, j)`, a monotone heuristic derived from the Hamming weight of the mask `m = j+1` and the RM order `r`;
3.  `-j` as a small tie‑breaker favoring lower indices.

These are sorted in descending order. Then a $\mathrm{GF}(2)$ MSB‑pivot linear basis is built incrementally with `_insert_vec_msb`, which tries to insert each column as a new basis vector by eliminating its most significant set bits against existing basis rows.

Columns are processed in descending reliability order; when `_insert_vec_msb(cols[j])` succeeds (meaning the column is linearly independent of the current basis), `j` is added to the information set. This continues until `K` columns are collected.
If the initial pass fails to reach `K`, a backup loop scans through all columns and attempts to fill the remaining spots while preserving independence. If even that fails, a `RuntimeError` is raised; for RM codes, this is not expected in practice.

The result is a list of `K` column indices `info` that are both reliable and form a full‑rank $K \times K$ submatrix when restricted to rows `rows`.

## 8.4 $K \times K$ inversion and base codeword

Once the information set is chosen, the decoder builds the $K \times K$ submatrix $A$ corresponding to those columns:
```python
A_rows = []
for i in range(K):
    row_bits = 0
    for j, col_idx in enumerate(info):
        if (rows[i] >> col_idx) & 1:
            row_bits |= (1 << j)
    A_rows.append(row_bits)
```
Here each `row_bits` is a `K`‑bit integer representing row `i` of $A$.

The helper `_gf2_inv(rows_kxk, K)` applies Gauss–Jordan elimination over $\mathbb{F}_2$ to invert this $K \times K$ matrix. It treats `rows_kxk` as the left part of an augmented $[A | I]$ matrix and performs row operations to obtain $[I | A^{-1}]$. The right half is returned as a list of `K` row integers, one per row of the inverse.

OSD then constructs the information‑set syndrome `s_bits`:
* for each `j`, `col_idx` in `info`, if `w_bits[col_idx]` is 1, bit `j` of `s_bits` is set.

The coefficient vector $a$ (in the RM basis) is recovered as:
```python
a_bits = 0
for j in range(K):
    if (Ainv_rows[j] & s_bits).bit_count() & 1:
        a_bits |= (1 << j)
```
so that bit `j` of `a_bits` is 1 iff the $j$-th row of $A^{-1}$ has odd overlap with `s_bits`. This gives the coefficients corresponding to the base codeword in the chosen information set.

The base codeword itself is constructed with `_encode_from_coeffs(rows, a_bits, K)`, which XORs generator rows where the corresponding bit in `a_bits` is 1. This is the OSD “order‑0” estimate around which higher‑order neighborhoods are explored.

## 8.5 OSD‑1/2/3 neighborhoods and depth‑aware tie‑breaking

The main function `osd_decode(w_bits, n, r, base_code_bits, order=1)` performs an OSD search of order 1, 2, or 3 around the baseline codeword defined by $a$.
It first recomputes the baseline codeword `cw0` from `a_bits` and `rows` (the generator). It then precomputes, for each position `j` in the information set:
1.  `Ainv_cols[j]`: the column vector of $A^{-1}$ corresponding to flipping the $j$-th information bit, encoded as `K` bits, and
2.  `delta_cw[j]`: the codeword delta obtained by applying that column of $A^{-1}$ as coefficients and encoding.

The received word is packed into `w_int` for Hamming distance computations, and `_depth_of(cw_int)` is defined as the T‑depth of the residual odd set:
```python
odd_bits = _int_to_bits_le(w_int ^ cw_int, L)
return rmcore.tdepth_from_punctured(_pack_bits_le(odd_bits), n)
```
This uses the C++ estimator to evaluate depth.

The algorithm maintains `best_cw`, `best_a`, `best_dist`, and `best_depth`, initialized to the baseline, and then explores neighborhoods:

* **Order 1 (OSD‑1):** for each `j`, flip `delta_cw[j]` to get a candidate codeword, compute its distance and depth, and update the incumbent when strictly better in distance, or equal in distance but shallower in depth.
* **Order 2 (OSD‑2):** if `order >= 2`, loops over all `i < j`, with an optional cap `RM_OSD2_MAX_PAIRS` on the number of checked pairs. Each candidate is `cw0 ^ delta_cw[i] ^ delta_cw[j]`, and the incumbent is updated in the same (distance, depth) lexicographic order.
* **Order 3 (OSD‑3):** similarly for triples `i < j < k`, with an optional cap `RM_OSD3_MAX_TRIPLES` on checked triples.

These environment variables allow you to contain the combinatorial explosion of higher‑order OSD neighborhoods while still benefitting from the most promising flips.

At the end, the best codeword is converted back to a bit list `code_bits`, and the corresponding coefficients `best_a` are used to select monomials from `_monoms(n, r)`. The function returns:
```python
(code_bits, selected_monomials, best_dist)
```
where `selected_monomials[i]` is the monomial mask for each 1 in `best_a`.

The **depth‑aware tie‑breaking** ensures that even when Hamming distance cannot be improved, OSD can still find candidates that reduce estimated T‑depth, which is valuable in the context of quantum circuit optimization.

## 8.6 Public API and relationship to standard OSD

Two functions are exposed for use by the rest of the library:

1.  `osd_decode(w_bits, n, r, base_code_bits, order)` is the full OSD routine described above. You typically call it with order 1, 2, or 3; higher orders are not implemented.
2.  `osd_refine_l1(w_bits, n, r, base_code_bits)` is a lightweight wrapper that simply calls `osd_decode(..., order=1)`. This is what `decoders.py` uses when it wants a cheap local refinement around an existing codeword without paying the cost of order‑2 or order‑3 neighborhoods.

From a coding‑theory perspective, this module is an instance of Ordered Statistics Decoding applied to punctured RM codes:
* reliability ordering is driven by agreement with a baseline codeword and a simple column weight heuristic,
* the information set is guaranteed to be full rank via a $\mathrm{GF}(2)$ basis check,
* coefficients are recovered in that information set and used to construct the base codeword, and
* small Hamming‑weight flips in the information set (order‑1, 2, 3) are explored to find better codewords.

The extra twist in this implementation is the depth‑aware scoring, which is specialized to the quantum setting: whenever Hamming distance ties, the candidate with lower estimated T‑depth wins. This makes OSD a natural local search primitive in the broader RM‑based circuit optimization pipeline.