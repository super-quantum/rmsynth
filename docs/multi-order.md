# 7. Multi‑order optimization (`rotk.py`)

`rotk.py` generalizes the RM‑based T‑count optimizer from pure $\mathbb{Z}_8$ (Clifford+T) to multi‑order phase polynomials, i.e. coefficients modulo $2^k$ or more generally modulo an integer $d$. This is the layer that lets you talk about optimizing $R_Z(2\pi/d)$ phases in a way that still looks like “decoding per bit‑plane”.



Conceptually, you can think of it as:

1.  splitting each coefficient into $k$ binary planes (for $2^k$),
2.  running a tailored per‑plane RM decoding (with some coupling heuristics and safety checks), and
3.  reassembling the planes without carries so that each plane is independently non‑worsening.

For composite $d$, the module adds a CRT‑style wrapper on top of that.

## 7.1 Basic helpers

The first group of helpers is about coercing inputs into a standard vector form and extracting or rebuilding bit‑planes.

`_mod_pow2(x, k)` is a tiny utility that reduces an integer modulo $2^k$ using bit‑masking:

```python
return x & ((1 << k) - 1)
```

This is used when you want to reinterpret an existing $\mathbb{Z}_8$ polynomial as modulo $2^k$ for some other $k$.

`_as_vec(a, n, k)` and `_to_dict(vec, n, k)` move between the two main representations used in the code:
1.  a dictionary `{mask -> coeff}` in the sense of `core.extract_phase_coeffs`, and
2.  a flat length‑$(2^n-1)$ list of integers (the “coefficient vector”).

`_as_vec` accepts either form and always returns a vector with entries reduced modulo $2^k$. When given a dict, it calls `coeffs_to_vec` first; when given a sequence, it just casts to int and masks.
`_to_dict` is the inverse for vectors: it calls `vec_to_coeffs` and filters out entries that are zero modulo $2^k$, so the result is a clean coefficient dictionary in the canonical monomial order.

**Bit‑plane extraction and reconstruction** are handled by `_plane_bits_from_vec` and `_rebuild_from_planes`.
Given a vector `a_vec` modulo $2^k$, `_plane_bits_from_vec(a_vec, n, k, ell)` returns the $\ell$-th bit‑plane as a punctured binary word:
* planes are indexed $\ell = 1, \dots, k$, with $\ell=1$ the most significant bit,
* for each coefficient, the function shifts down by `k-ell` and takes the least significant bit.
  So `plane[ell-1][j]` is the $\ell$-th bit of the $j$-th coefficient.

`_rebuild_from_planes(planes, k)` performs the inverse: starting from a list of planes `[plane_1, ..., plane_k]`, it rebuilds a vector modulo $2^k$ by setting the bit at position `k-ell` wherever the $\ell$-th plane has a 1. Importantly, this is **bitwise reconstruction with no carries**: planes are treated as independent binary layers.

Two further helpers round out the module’s “modulus plumbing”.
* `_v2(d)` computes the 2‑adic valuation $v_2(d)$: the largest $k$ such that $2^k$ divides $d$. It is used in the composite‑modulus wrapper to split $d = 2^k d_{\text{odd}}$.
* `_as_vec_mod(a, n, mod)` is a generalization of `_as_vec` when the modulus is an arbitrary integer `mod`. It again accepts either a dict or a vector, but reduces entries modulo `mod` instead of $2^k$. This is what you use when the target modulus is not a pure power of two.

## 7.2 Per‑plane weights

The function `weights_by_plane(a, n, k)` is the main observable that links the implementation back to the theory in the paper.
It takes either a dict or a vector of coefficients modulo $2^k$, and returns a list:

$$
[w_1, w_2, \dots, w_k],
$$

where $w_\ell$ is the Hamming weight of the $\ell$-th bit‑plane:

```python
a_vec = _as_vec(a, n, k)
return [ sum(_plane_bits_from_vec(a_vec, n, k, ell)) for ell in range(1, k + 1) ]
```

So $w_\ell$ tells you how many monomials have a 1 in the $\ell$-th bit of their coefficient. When $k=3$ ($\mathbb{Z}_8$), these are exactly the per‑plane counts for the T‑plane (LSB), S‑plane, and Z‑plane. For a general $R_Z(2\pi/2^k)$ phase, they correspond to the “parity” patterns across different bit‑significance levels.

The multi‑order optimizer aims to reduce these per‑plane weights, typically starting from the most significant plane and working downwards, while guaranteeing that each plane’s weight never increases.

## 7.3 Per‑plane candidate selection

Choosing how to update a single bit‑plane is encapsulated in `_choose_candidate_for_plane(...)`.
Given:
* `n`, `k`, and a plane index `ell`,
* a residual word `w_bits` for that plane (current 1s on bit‑plane $\ell$),
* a target RM code $\mathrm{RM}(r,n)^*$ with `r = n - ell - 1`,
* and a primary strategy name `strategy` (e.g. `"rpa-adv"`),

this helper builds and scores a small set of candidate codewords.

The first candidate is always the zero codeword, which corresponds to “do nothing on this plane”:
```python
zero_code = [0] * L
candidates.append((zero_code, [], before))
```
Here `before = sum(w_bits)` is the current plane weight; the distance of the zero codeword to `w_bits` is exactly `before`.

Then it calls the requested strategy via `decode_rm`:
```python
code_bits, selected, dist = decode_rm(w_bits, n, r, strategy=strategy, ...)
candidates.append((code_bits, selected, dist))
```
This is the “main” per‑plane candidate. To make tie‑breaking less brittle, a side strategy is also tried: if the primary is `"dumer-list"`, the side is `"rpa-adv"`, and vice versa. The side candidate is appended if the call succeeds.

Optionally, if `osd_per_plane` is enabled and the lightweight OSD wrapper `osd_refine_l1` is available, each candidate is refined by a small local OSD‑L1 search on that plane. For each base candidate `(cb, sm, d)`:
1.  `osd_refine_l1` is called to get a refined candidate `(rb, rsel, rd)`,
2.  if `rd < d`, the refined version is kept;
3.  if `rd == d`, the one with smaller support (fewer 1s) is preferred, as this tends to help later coupling;
4.  otherwise, the original is retained.
    This step stays within $\mathrm{RM}(r,n)$, because OSD operates on the same generator structure.

After per‑plane OSD, duplicates are removed by hashing the codeword bits; only unique `(code_bits, monomials, dist)` triples remain.

The final choice uses a scoring function that reflects three priorities:
1.  **Primary objective:** minimize the new plane weight `after = |w ^ code_bits|`. This is the main thing that should not get worse.
2.  **Optional plane weights:** if a vector of plane weights `weights` was provided, the score multiplies `after` by a plane‑specific factor, so some planes can be treated as more or less important.
3.  **Coupling proxy:** if `coupling != "none"` and there is at least one selected monomial, a simple overlap proxy is computed:
    * each monomial mask `t` is lifted to a parity vector via `lift_to_c([t], n)` and then reduced to its odd support,
    * the overlap between that support and the current residual `w_bits` is added up across monomials.

The score is essentially `(after, weighted_after, coupling_proxy)`. The candidate with the lexicographically smallest triple is chosen. Because `after` comes first, no candidate that increases the plane’s residual Hamming weight can ever win at this stage; and there is a separate safety check in the main loop that enforces this again.

## 7.4 Multi‑order bit‑plane algorithm

The heart of the module is `optimize_multiorder_bitplanes(a_in, n, k, ...)`, which takes coefficients modulo $2^k$ and returns an improved vector plus per‑plane statistics.
The algorithm proceeds in four conceptual steps.

### 1. Prepare planes and initial stats.
The input `a_in` is coerced into a vector modulo $2^k$ using `_as_vec`. For each plane $\ell$ from 1 to $k$, `_plane_bits_from_vec` is used to create a binary word `planes[ell-1]` representing the $\ell$-th bit of every coefficient.
A stats list is initialized; if `batch` is true, a `_RowCache` is constructed (it currently caches RM generator rows per `r`, ready for future coupling improvements).
GL(n,2) options, if present in `dec_kwargs` (`gl_trials`, `gl_family`), are extracted for per‑plane use.

### 2. For each plane $\ell = 1..k$: decode and possibly apply a change.
For each plane index `ell`:

* The RM order is `r = n - ell - 1`. This matches the hierarchical structure in the paper: higher planes use lower‑order RM codes.
* The current plane residual is `w = planes[ell-1]`, with weight `before = sum(w)`.
* If `r < 0` or `before == 0`, nothing is done: the plane is trivially zero or below the lowest RM order, and stats record no change.

Otherwise:

* `_choose_candidate_for_plane` is called to obtain a baseline candidate `(cb, sel, dist)` and its derived “after” weight `after_cb = |w ^ cb|`.
* Optionally, if `gl_trials` is positive, a per‑plane GL(n,2) preconditioning step is attempted: `decode_rm` is called with the same strategy and a non‑zero `gl_trials`. This internally tries several GL‑transformed versions of the plane, decodes each once, and maps back. The best GL candidate `(cb_gl, dist_gl)` is compared to the baseline; it is accepted only if it strictly reduces the new plane weight, or if it ties in new weight but has smaller distance.

The candidate selected after this GL comparison is `best_cb`. The provisional new plane is `w_new = w XOR best_cb`, with `after = |w_new|`.

To enforce the “never worsen this plane” guarantee, there is a final safety check:
```python
if after > before:
    w_new = w[:]        # revert
    after = before
    best_dist = before  # distance to zero-code fallback
```
This means that even if the decoder suggested a codeword with a higher residual than the original, the optimizer will simply ignore it and keep the old plane. Stats for that plane then reflect no improvement, and the distance is set to the cost of the zero candidate.
The updated plane `w_new` is written back to `planes[ell-1]`, and a stats entry is appended with fields `ell`, `r`, `before`, `after`, and the best distance.

### 3. Rebuild coefficients and normalize stats.
Once all $k$ planes have been processed, `_rebuild_from_planes(planes, k)` is used to reconstruct the optimized coefficient vector `a_out_vec` modulo $2^k$. Because no inter‑plane carries occur, this is literally the per‑bit recombination of the final plane patterns.

To make sure the stats align with the final vector, the code recomputes the per‑plane weights:
```python
w_final = weights_by_plane(a_out_vec, n, k)
for s in stats:
    s['after'] = int(w_final[s['ell'] - 1])
```
This is important because per‑plane decoding decisions can be influenced by GL search and coupling; it is easier and safer to recompute the final weights from the actual output.

## 7.5 Composite $m$ via CRT

`optimize_multiorder_d(a_in, n, d, ...)` lifts the multi‑order optimizer from pure $2^k$ moduli to arbitrary positive integers $d$. It starts by splitting $d$ into its dyadic and odd parts:

```python
k = _v2(d)       # largest k with 2^k | d
d_odd = d >> k   # the remaining odd factor
```

The input coefficients are coerced modulo $d$ via `_as_vec_mod`. If `k == 0` (i.e. $d$ is odd), nothing can be done with RM codes (which live in characteristic 2), so the function simply returns the original vector and an info dict with empty per‑plane fields.

For `k > 0`, the even part of the coefficients is extracted:
```python
m1 = 1 << k
a_even = [v % m1 for v in a_vec_d]
```
This is the component modulo $2^k$ that multi‑order RM can act on. The function then calls `optimize_multiorder_bitplanes(a_even, n, k, ...)` to improve this even part; all strategy and decoder knobs are forwarded.

Once the optimized even part `a_even_opt` is computed, the code merges it back with the original coefficients’ odd residues using `_crt_merge_even_odd`. For each coordinate, `_crt_merge_even_odd` solves

$$
x \equiv a_{\text{even,opt}} \pmod{2^k}, \quad x \equiv a_{\text{odd,orig}} \pmod{d_{\text{odd}}}
$$

and returns the unique solution modulo $d$. The modular inverse of $2^k$ modulo `d_odd` is computed once and reused; per‑coordinate CRT reconstruction is then cheap.

The info dictionary returned alongside the optimized vector includes:

* `k` and `d_odd`,
* `before_plane` and `after_plane` for the even part only (via `weights_by_plane`),
* and the full list of per‑plane stats from the $2^k$ pass.
  This makes it straightforward to inspect how much of the improvement came from the dyadic component when working with general $d$.

## 7.6 Circuit‑level wrapper

The convenience function `optimize_circuit_multiorder(circ, k, ...)` applies all of the above to a `Circuit`.
It starts by extracting the current coefficients (modulo 8) using `extract_phase_coeffs`. These are then reinterpreted modulo $2^k$:
```python
a0 = extract_phase_coeffs(circ)         # {mask -> k_mod8}
a0_k = {m: _mod_pow2(v, k) for m, v in a0.items()}
```
Per‑plane weights before optimization are computed with `weights_by_plane(a0_k, n, k)`.

The multi‑order bit‑plane optimizer is then run on `a0_k`:
```python
a1_vec, stats = optimize_multiorder_bitplanes(
    a0_k, n, k, strategy=strategy, list_size=list_size, **dec_kwargs
)
```
Afterwards, `weights_by_plane(a1_vec, n, k)` gives the per‑plane weights after optimization.

To produce a new circuit, `_to_dict` and `coeffs_to_vec` are used to turn `a1_vec` into a standard $\mathbb{Z}_8$‑like coefficient vector, and `synthesize_from_coeffs` is called to synthesize a circuit for those coefficients.

The function returns `new_circ` together with an info dictionary containing:
* `before_plane` and `after_plane` per‑plane weights,
* the full stats list (one entry per plane), and
* optionally, if `k == 3`, explicit `before_T` and `after_T` counts computed from the LSB (oddness) of the original and optimized coefficients.

In other words, `optimize_circuit_multiorder` is the multi‑order analogue of `Optimizer.optimize`, but focused entirely on per‑plane counts rather than on a full RM‑based T‑count report.