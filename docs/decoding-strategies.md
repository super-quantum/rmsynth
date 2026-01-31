# 5. Decoding strategies and policies (`decoders.py`)

The `decoders.py` module is the main “strategy layer” between the raw `rmcore` decoders and the higher‑level optimizer. It knows how to:

* talk to the C++ extension (`rmcore`),
* plug in fall‑back Python decoders when the extension is missing,
* combine different primitive decoders (Dumer, RPA‑1, OSD, GL(n,2) preconditioning),
* and apply distance/depth policies and local refinements like SNAP or OSD.

All public decoders ultimately return the same kind of object:

```python
code_bits: List[int]           # punctured RM(r,n) codeword, length (2^n - 1), little-endian
selected_monomials: List[int]  # masks of degree <= r
distance: int                  # Hamming distance to the input oddness
```

That structure is exactly what the optimizer and contracts expect.

## 5.1 Bit packing and `rmcore` loading

At the boundary with C++, the code needs to convert between Python lists of bits and the packed little‑endian byte strings that `rmcore` expects. Two small helpers do this:

```python
def pack_bits_le(bit_list: List[int]) -> bytes:
    ...

def unpack_bits_le(b: bytes, L: int) -> List[int]:
    ...
```

The convention is:
* `bit_list[i]` is bit `i` of the punctured word.
* `pack_bits_le` puts bit `i` into byte `i >> 3`, bit position `(i & 7)`.
* `unpack_bits_le` reverses that, up to a desired length `L`.

Everything that goes into or comes out of `rmcore` uses this format, so the representation is consistent across the entire library.

The `_load_rmcore()` function is responsible for actually finding and importing the compiled extension:

```python
def _load_rmcore():
    # 1) try rmsynth.rmcore
    # 2) try top-level rmcore
    # 3) scan sys.path for an rmcore*.so/pyd, possibly under a 'rmsynth' dir
```

It tries several locations:
1.  `from rmsynth import rmcore` – package‑local layout.
2.  `import rmcore` – flat layout where the extension is on `sys.path` directly.
3.  A manual scan through `sys.path` for shared libraries whose name starts with `rmcore` (optionally under a `rmsynth` directory). For each candidate file it uses `importlib.util.spec_from_file_location` and executes the module.

If everything fails, it returns `None`.

`best_available_decoder()` is a tiny helper built on top of this: if `_load_rmcore()` is available, it recommends `"dumer"`; otherwise it falls back to `"ml-exact"`, i.e. the pure‑Python brute‑force decoder.

## 5.2 Core helpers: integers and depth

Two small helpers convert between bit lists and Python integers in little‑endian order:

```python
def _bits_to_int(bits: List[int]) -> int:
    ...

def _int_to_bits(x: int, L: int) -> List[int]:
    ...
```

They are used everywhere inside local refinements (SNAP, OSD) where manipulating codewords as Python integers is simpler than keeping them as lists.

The more interesting helper is `_depth_from_solution`, which is the single source of truth for “how deep is this solution?”:

```python
def _depth_from_solution(
    w_bits: List[int], n: int,
    code_bits: List[int],
    monoms: Optional[List[int]]
) -> int:
    ...
```

Here `w_bits` is the received word (punctured oddness), `code_bits` is a candidate codeword, and `monoms` is an optional list of monomial masks that produced this codeword.

The function computes the residual odd set
```python
odd_bits = [(wb ^ cb) & 1 for wb, cb in zip(w_bits, code_bits)]
```
and then chooses between a fast estimator and an optional exact scheduler:

1.  If `rmcore` is missing, it simply returns `sum(odd_bits)`, i.e. the Hamming weight of the residual. This is the worst‑case T‑depth and works even without the extension.
2.  If `rmcore` is available, it reads `RM_DEPTH_MODE` from the environment. The default is `"est"`:
    * If `RM_DEPTH_MODE != "sched"`, it uses the fast estimator in C++:
        ```python
        return rmcore.tdepth_from_punctured(pack_bits_le(odd_bits), n)
        ```
      This is the path used for all inner‑loop work inside `decode_rm`: cheap, safe, and always an upper bound.
    * If `RM_DEPTH_MODE == "sched"` and a list of monomials is provided and `rmcore` has `tdepth_schedule_from_monoms`, it attempts an exact (bounded) DSATUR‑based scheduling:
        ```python
        budget_env = os.environ.get("RM_SCHED_BUDGET", "")
        budget = int(budget_env) if budget_env.strip() else 20000
        depth, _layers = rmcore.tdepth_schedule_from_monoms(monoms, n, budget)
        ```
      If this call fails for any reason, or if `monoms` is `None`, it falls back to the estimator.

This design ensures that:
* All inner policy decisions (which candidate is better?) are by default based on the estimator, so decoding remains fast and scalable.
* If you explicitly opt into exact scheduling via environment variables, you can get depth numbers that reflect the best layering the scheduler found for that candidate (subject to its internal caps).

## 5.3 Basic GF(2) linear algebra

Several decoders and refinements need to solve linear systems over $\mathrm{GF}(2)$ or manipulate invertible linear maps. `decoders.py` includes a small, self‑contained toolkit for this.

`_gf2_inv(rows_kxk, K)` performs Gauss–Jordan elimination on a $K \times K$ matrix over $\mathrm{GF}(2)$, where each row is stored as an integer bitmask. It keeps a copy of the left matrix and an identity matrix on the right:

```python
AugL = rows_kxk[:]
AugR = [1 << i for i in range(K)]
```

For each column, it finds a pivot row with a 1 in that column, swaps it into place, and eliminates that column from all other rows. If any column has no pivot, it raises `RuntimeError("Singular matrix in inversion.")`. Otherwise it returns the rows of $A^{-1}$ as bitmasks.

`_choose_info_columns(rows, K, L)` is a more specialized routine that selects a set of $K$ linearly independent columns in a $K \times L$ generator matrix. It runs Gauss–Jordan over columns instead of rows:
1.  It keeps a working copy `R` of the rows.
2.  It scans columns from left to right, and for each column finds a pivot row `r` that has a 1 in that column.
3.  After swapping that row into position `r`, it eliminates the column from all other rows.
4.  If it manages to find `K` such pivot columns, it returns their indices; otherwise it returns `None`.

This function is used to choose a robust “information set” of columns for monomial recovery.

For $\mathrm{GL}(n,2)$ work, there is a second pair:

* `_gf2_mat_inv_rows(rows, n)` inverts an $n \times n$ matrix given as row bitmasks, returning `None` if singular. This is used exclusively for constructing invertible linear maps in $\mathrm{GL}(n,2)$.
* `_gf2_matmul_rows(A_rows, B_rows, n)` multiplies two $n \times n$ matrices over $\mathrm{GF}(2)$ in row form. Row `i` of the product is the XOR of rows of `B` indicated by the 1‑bits in row `i` of `A`.

Once you have an invertible matrix $A$ over $\mathrm{GF}(2)$, `_apply_linear_rows_to_mask(mask, rows, n)` applies it to an $n$-bit column vector `v` encoded as `mask`:

```python
out = 0
for i in range(n):
    if ((rows[i] & mask).bit_count() & 1):
        out |= (1 << i)
```

This is then lifted to punctured words via `_permute_punctured_bits_gl(bits, n, A_rows)`, which maps each column index $j+1$ to $A \cdot (j+1)$ and writes bit `bits[j]` into the corresponding position `mp-1`. Because $A$ is invertible, this is a bijection on the nonzero masks $\{1, \dots, 2^n-1\}$.

## 5.4 Monomial recovery and GL(n,2) search

Once you have a valid punctured $\mathrm{RM}(r,n)$ codeword, you sometimes need to recover one coefficient vector (set of monomials) that generates it. `_recover_monoms_from_code_bits` does precisely this:

```python
def _recover_monoms_from_code_bits(code_bits: List[int], n: int, r: int) -> List[int]:
    ...
```

It proceeds as follows:
1.  It fetches the cached generator rows and monomial masks from `rm_cache`:
    ```python
    rows = get_rows(n, r)      # K rows, each L-bit int
    mons = get_monom_list(n, r)
    K = len(rows); L = len(code_bits)
    ```
2.  It calls `_choose_info_columns(rows, K, L)` to pick an information set of `K` columns such that the resulting $K \times K$ submatrix is invertible.
3.  It builds that submatrix $A$ as a list of `K` row bitmasks, one per monomial.
4.  It inverts $A$ with `_gf2_inv` to obtain $A^{-1}$.
5.  It builds a `K`‑bit “syndrome” vector `s_bits` of the codeword bits at the chosen columns.
6.  It computes the coefficient vector `a_bits = A^{-1} s` row‑wise:
    ```python
    if (Ainv_rows[j] & s_bits).bit_count() & 1:
        a_bits |= (1 << j)
    ```
7.  Finally, it returns the list of monomial masks `mons[i]` where `a_bits` has a 1.

This routine is only used on the Python side in places where the C++ decoder didn’t already provide monomials (e.g. after GL(n,2) preconditioning or when selecting among top‑K punctured candidates). It always yields a coefficient vector with degree $\le r$, and by construction its generator‑row sum equals the codeword.

The $\mathrm{GL}(n,2)$ machinery is driven by `_rand_gl_rows`:

```python
def _rand_gl_rows(n: int, family: str = "random") -> Tuple[List[int], List[int]]:
    ...
```

It returns a pair `(A_rows, Ainv_rows)` of $n \times n$ matrices in row‑bitmask form, guaranteeing that `A_rows` is invertible and `Ainv_rows` is its inverse. There are three families:
* `"perm"`: pure permutation matrices.
* `"triangular"`: product of a random lower‑triangular and random upper‑triangular matrix with ones on the diagonal.
* `"random"`: dense random matrices sampled until a nonsingular one is found.

Given such a matrix, `_permute_punctured_bits_gl(bits, n, A_rows)` applies the corresponding permutation on monomial masks to a punctured word.

The high‑level GL wrapper `_decode_rm_with_gl_search` wraps any base strategy:

```python
def _decode_rm_with_gl_search(
    w_bits_list, n, r, strategy,
    list_size, chase_t, chase_limit,
    snap, snap_t, snap_pool, rpa_iters, osd_top,
    snap_strong, snap_time_ms, snap_node_limit,
    depth_tradeoff, policy, policy_lambda,
    gl_trials, gl_family,
) -> Tuple[List[int], List[int], int]:
    ...
```

The logic is:
1.  Run the base `decode_rm` once on the original word, with `gl_trials=1` and `gl_family=None` to prevent recursion. This produces a baseline (`best_bits`, `best_sel`, `best_dist`) and a baseline depth via `_depth_from_solution`.
2.  For each of the remaining `gl_trials - 1` iterations:
    * Sample $A \in \mathrm{GL}(n,2)$ with `_rand_gl_rows`.
    * Permute the received word: `w_perm = _permute_punctured_bits_gl(w_bits_list, n, A_rows)`.
    * Decode once using the requested strategy and parameters, again with `gl_trials=1`.
    * Unpermute the candidate: `cand_bits = _permute_punctured_bits_gl(cand_bits_perm, n, Ainv_rows)`.
    * Recover a monomial set for the candidate via `_recover_monoms_from_code_bits`.
    * Compute its distance and depth.
    * Decide whether to accept it using `policy_decide`.
3.  Return the best candidate according to the policy.

The crucial point is that GL search never changes the code space itself; it only permutes coordinates, decodes in that permuted basis, and then unpermutes. The recovered monomials always correspond to a valid $\mathrm{RM}(r,n)$ codeword.

## 5.5 Cost models and policies

The module supports several ways to compare candidates: 

- pure distance (classical decoding),
- lexicographic distance + depth,
- and a guarded linear trade‑off between distance and depth.

`_lambda_from_policy(policy, policy_lambda, fallback)` translates a textual policy into a numeric `depth_tradeoff` ($k$):

* `policy` is `None` → returns `fallback`.
* `"distance+depth"`, `"dist+depth"`, `"linear"`, `"lin"` → returns `int(policy_lambda)` if given, otherwise `fallback`.
* `"depth-only"` or `"min-depth"` → returns 0, which is treated as “depth dominates, but guards still apply”.
* `"min-distance"` or `"distance"` → returns `None`, i.e. ignore depth in the main cost.
* Any unknown string → `fallback`.

`_distance_depth_cost` computes a triple `(dist, depth, cost)` for a given candidate:

```python
dist = sum((wb ^ cb) & 1 for wb, cb in zip(w_bits, code_bits))
if k is None:
    depth = _depth_from_solution(w_bits, n, code_bits, monoms=None)
    cost  = dist
else:
    depth = _depth_from_solution(w_bits, n, code_bits, monoms)
    cost  = k*dist + depth
```

This triple can then be used to sort candidates or to implement “distance first, depth as tie‑breaker”.

`_choose_by_tradeoff(cands, w_bits, n, k)` is used in the `"dumer-list"` path when `depth_tradeoff` is set and top‑K candidates are available. `cands` is a list of `(code_bits, monoms_or_None, dist)` tuples. The function computes `_distance_depth_cost` for each candidate, then chooses:

* `(cost, dist, depth)` as the primary sorting key when `k` is not None,
* `(dist, depth)` otherwise.
  It returns the best `(code_bits, monoms, dist)` triple.

The public policy function is `policy_decide`:

```python
def policy_decide(best_dist: int,
                  best_depth: int,
                  cand_dist: int,
                  cand_depth: int,
                  policy: str | None = None,
                  policy_lambda: float | None = None,
                  depth_tradeoff: int | None = None) -> bool:
    ...
```

It answers: “Should we accept the candidate over the current best?” There are two regimes:

### Guarded trade‑off (when `depth_tradeoff` is not `None`)
Let $k = \text{depth tradeoff}$, $\Delta d = \text{cand dist} - \text{best dist}$, and $\Delta h = \text{cand depth} - \text{best depth}$. The rules are:
1.  If the candidate has worse distance ($\Delta d > 0$), it is only accepted if the depth improves by strictly more than $k \cdot \Delta d$:
    `(-dh) > k * dd`.
2.  If the candidate has better distance ($\Delta d < 0$):
    * If depth is better or equal (`dh <= 0`), always accept.
    * If depth is worse (`dh > 0`), only accept if the distance improvement is strictly greater than $k \cdot \Delta h$: `(-dd) > k * dh`.
3.  If the distances are equal ($\Delta d == 0$), accept only on strict depth improvement (`dh < 0`).

This guard logic is designed so that depth can never “cheat” you into accepting a candidate with a significantly worse distance unless you explicitly allow a trade at the chosen rate ($k$).

### Ungarded policies (when `depth_tradeoff` is `None`)
If `policy` is `"distance+depth"` (case insensitive), it uses a linear cost:

$$
\text{dist} + \lambda \cdot \text{depth}
$$

with `lam = float(policy_lambda or 0.0)` and accepts the candidate if its cost is strictly smaller.
Otherwise it falls back to pure lexicographic comparison:

* accept if candidate distance is smaller;
* if distances are equal, accept only when candidate depth is smaller.

All high‑level strategies use `policy_decide` to compare their own internal candidate pools.

## 5.6 Local refinements: SNAP and branch‑and‑bound

The SNAP routines do local exact search around a given codeword but always stay inside the $\mathrm{RM}(r,n)$ code by toggling generator rows.

`snap_local_exact_bits` is the light‑weight version:

```python
def snap_local_exact_bits(
    w_bits: List[int], n: int, r: int,
    code_bits: List[int], monoms: List[int],
    snap_t: int = 2, pool_max: int = 16, comb_limit: int = 200
) -> Tuple[List[int], List[int], int]:
    ...
```

The steps are:

1.  Convert the received word and codeword to integers `w` and `cw`, and compute the residual `res = w ^ cw`.
    * If the residual is zero, return immediately – we are at zero distance.
2.  Build a pool of generator rows with strong overlap with the residual:
    * Fetch `rows = get_rows(n, r)` and the aligned `monom_list`.
    * For each row, compute `score = popcount(res & row)` and keep those with `score > 0`.
    * Sort them in descending order of score and keep the top `pool_max`.
3.  Initialize `best_cw = cw` and `best_dist = popcount(w ^ cw)`.
4.  Try flipping each row in the pool (size‑1 subsets), up to `comb_limit` candidates. For each:
    * Compute the candidate codeword `cand = cw ^ pool_rows[i]`.
    * Measure its distance `d = popcount(w ^ cand)`.
    * Keep the best candidate seen so far.
5.  If `snap_t >= 2` and the combination budget has not been exhausted, do the same for all row pairs in the pool (size‑2 subsets), again up to `comb_limit`.
6.  If no improvement was found, return the original `(code_bits, monoms, base_dist)`.
7.  Otherwise, derive the set of generator rows toggled (`best_subset`) and update the monomial set accordingly:
    * Maintain a `mono_set` from the original `monoms`.
    * For each toggled monomial `t`, remove it if present, or add it if absent.
    * Sort and return the new list of monomials and the new codeword bits (converted from `best_cw`).

Because the only operations on the codeword are XORs with generator rows, SNAP never leaves the $\mathrm{RM}(r,n)$ code; it just explores a small neighborhood in the coset defined by the received word.

`snap_branch_and_bound_bits` is a more powerful branch‑and‑bound version:

```python
def snap_branch_and_bound_bits(
    w_bits: List[int], n: int, r: int,
    code_bits: List[int], monoms: List[int],
    pool_max: int = 24,
    time_ms: int = 15,
    node_limit: int = 100_000
) -> Tuple[List[int], List[int], int]:
    ...
```

It follows the same basic pipeline:

1.  Build `res0 = w ^ cw` and a pool of rows by overlap.
2.  Reorder the pool by a “gain” heuristic ($2 \cdot \text{overlap} - \text{weight(row)}$) so that rows that are likely to reduce the residual are tried early.
3.  Seed an incumbent solution using all size‑1 and size‑2 subsets, as in the light SNAP.
4.  Then it runs a recursive DFS with pruning:
    * The recursion state is `(idx, curRes, chosen_mask)`, where `idx` is the pool index, `curRes` is the current residual, and `chosen_mask` is a bitmask of which rows have been toggled.
    * A `lower_bound(curRes, idx)` function estimates the best achievable distance from this state by summing positive gains; it gives a lower bound on the final residual weight.
    * The DFS updates the global best whenever it finds a smaller residual weight. It also reconstructs the corresponding `best_cw_int` by xoring rows indicated in `chosen_mask`.
    * It prunes branches where:
        * the lower bound is already $\ge$ the best distance,
        * the time limit `time_ms` has been exceeded,
        * or the node count `node_limit` has been reached.
5.  At the end, if `best_dist < base_dist`, it applies exactly the same “toggle monomials” logic as in the light SNAP to produce a new monomial set and codeword. Otherwise, it returns the original.

In `decode_rm`, the strong SNAP is only used in `rpa-adv` and `rpa2` when `snap_strong` is set (either explicitly or via effort presets). Even then, its output is guarded by `_decide`, so you only keep it if it is consistent with your chosen policy.

## 5.7 RPA‑2 permutations

RPA‑2 adds permutations of the physical qubits on top of RPA‑1. On the punctured side this shows up as permutations of the monomial masks, which are handled entirely in Python.
The basic helpers are:

* `_perm_inv(perm)` – compute the inverse of a permutation list.
* `_apply_perm_to_mask(mask, n, perm)` – apply a qubit permutation to a monomial mask: for each bit set in mask, set the corresponding bit at `perm[b]`.
* `_permute_punctured_bits(bits, n, perm)` – apply the permutation to column indices: $y = j+1$ gets mapped to `mp = _apply_perm_to_mask(y, n, perm)`, and the bit at `j` is placed at `mp-1`.
* `_unpermute_punctured_bits` uses the inverse permutation to restore the original ordering.

`_gen_rpa2_perms(n, max_perms)` constructs a small family of qubit permutations without trying to enumerate all ($n!$) elements:

1.  It always includes the identity permutation.
2.  It adds permutations that swap qubit 0 with each other qubit.
3.  It then adds permutations that swap `(0,i)` and `(1,j)` for $i<j$, until `max_perms` is reached, if specified.
    This gives a reasonably rich set of “axes” to project along without exploding in number.

The main RPA‑2 loop in `decode_rm` looks like this:

1.  Compute a reference beam candidate with `decode_rm_dumer_list_punctured` (baseline).
2.  Generate permutations via `_gen_rpa2_perms`.
3.  For each permutation `p`:
    * Permute the received word: `w_perm`.
    * Decode once with `rmcore.decode_rm_rpa1_punctured` on `w_perm` to get a full RPA‑seeded candidate.
    * Unpermute the candidate codeword: `code_un = _unpermute_punctured_bits(code_p, n, p)`.
    * Transform the monomial masks using the inverse permutation `inv = _perm_inv(p)` and `_apply_perm_to_mask`.
    * Compute distance and depth via `_depth_from_solution`.
    * Track the best candidate by `(dist, depth)`.
4.  At the end, it compares the best RPA‑2 candidate against the original beam candidate using `_decide` (i.e. under your configured policy), and keeps the winner. If you requested `strategy="rpa2"`, this winner is then passed through SNAP and optional OSD refinement exactly as in `rpa-adv`.

## 5.8 The main dispatcher: `decode_rm(...)`

The `decode_rm` function is the central entry point used by the optimizer and the benchmarking tools:

```python
def decode_rm(
    w_bits_list: List[int],
    n: int,
    r: int,
    strategy: str,
    list_size: int | None = None,
    chase_t: int | None = None,
    chase_limit: int | None = None,
    snap: bool = False,
    snap_t: int | None = None,
    snap_pool: int | None = None,
    rpa_iters: int | None = None,
    osd_top: int | None = None,
    snap_strong: bool | None = None,
    snap_time_ms: int | None = None,
    snap_node_limit: int | None = None,
    depth_tradeoff: int | None = None,
    policy: Optional[str] = None,
    policy_lambda: Optional[int] = None,
    gl_trials: Optional[int] = None,
    gl_family: Optional[str] = None,
):
    ...
```

The call always uses a punctured oddness word `w_bits_list` of length $L = (1 \ll n) - 1$. The behavior depends on the strategy and on whether the C++ extension is available.

At the very top, it converts `policy/policy_lambda` into a numeric `depth_tradeoff` via `_lambda_from_policy`. Then:

* If `gl_trials` is not `None` and greater than 1, it delegates entirely to `_decode_rm_with_gl_search`, which wraps any base strategy with GL(n,2) preconditioning and `policy_decide`. This wrapper calls back into `decode_rm` with `gl_trials=1` to avoid recursion.
* Otherwise, it defines `_decide(...)` as a small closure that calls `_depth_from_solution` and `policy_decide`.

The strategy dispatch is:

**Pure Python brute‑force – `"ml-exact"`**
Uses `core.decode_rm_bruteforce` over Python ints (no `rmcore` needed). This is exponential and only intended for small $n$, or as a correctness reference.

**No `rmcore` present**
If `_load_rmcore()` returns `None`, any non‑`"ml-exact"` strategy falls back to `decode_rm(..., "ml-exact")`. This makes sure the API still works (slowly) even if you haven’t built the extension.

**C++ “simple” strategies**
After computing `w_bytes = pack_bits_le(w_bits_list)`, the following delegations go directly to the extension:

* `"dumer"` → `rmcore.decode_rm_dumer_punctured`.
* `"dumer-list"`:
    * If `depth_tradeoff` is set and `rmcore` exposes `decode_rm_dumer_list_topk_punctured`, the code pulls a top‑K list of punctured codewords, scores them via `_choose_by_tradeoff`, recovers monomials with `_recover_monoms_from_code_bits`, and returns the chosen candidate.
    * Otherwise it calls `rmcore.decode_rm_dumer_list_punctured`.
* `"dumer-list-chase"` → `rmcore.decode_rm_dumer_list_chase_punctured`.
  In each case the result is converted back to a `List[int]` using `unpack_bits_le`.

**RPA‑1 based strategies – `"rpa-seed-beam"`, `"rpa-adv"`**
These always combine two candidates:
1.  A beam candidate from `rmcore.decode_rm_dumer_list_punctured`.
2.  An RPA‑seeded candidate from `rmcore.decode_rm_rpa1_punctured` with the given number of iterations and list size.
    After unpacking the codewords, `_decide` chooses between them under the current policy. For `"rpa-adv"`, the winning candidate is then refined by:
* light SNAP (`snap_local_exact_bits`), always;
* strong SNAP (`snap_branch_and_bound_bits`) if `snap_strong` is set;
* OSD‑L1 (`osd_refine_l1`) as a final, policy‑guarded step.

**RPA‑2 strategies – `"rpa2-seed-beam"`, `"rpa2"`**
These run the multi‑axis RPA‑1 loop described above, scoring candidates by distance and depth and then guarding the final RPA‑2 candidate against the original beam using `_decide`. The `"rpa2"` variant then applies SNAP and OSD in the same way as `"rpa-adv"`.

**OSD families – `"osd1"`, `"osd2"`, `"osd3"` and `"beam-osd1/2/3"`**
These decorate a beam decoder with OSD around it:
* For `"osd1/2/3"`, the pipeline is:
    * beam baseline via `rmcore.decode_rm_dumer_list_punctured`,
    * refinement by `osd_decode` with order 1, 2, or 3,
    * `_decide` chooses between the baseline and the refined candidate.
* For `"beam-osd1/2/3"`, the beam part itself is explored more thoroughly:
    * baseline beam via `decode_rm_dumer_list_punctured`,
    * plus optional top‑K beam seeds via `decode_rm_dumer_list_topk_punctured`,
    * each seed is refined by OSD,
    * the best refined candidate is then compared against the baseline via `_decide`.

**RPA alias – `"rpa"`**
This is just a convenient alias to the advanced RPA decoder:
```python
return decode_rm(w_bits_list, n, r, "rpa-adv", ...)
```
with the same SNAP and OSD behavior, but easier to spell.

For all strategies, the contract is the same: `decode_rm` returns punctured bits, a list of monomials with degree $\le r$, and a distance which equals the Hamming distance between `w_bits_list` and `code_bits`. The `contracts.py` module uses these outputs to check that:

1.  every monomial lies in $\mathrm{RM}(r,n)$,
2.  the code bits match the XOR of the generator rows for `selected_monomials`,
3.  and the reported distance matches the implied T‑count after lifting and re‑synthesis.

That makes `decode_rm` the central, policy‑aware oracle for “what codeword should we correct to?” in the entire library.

