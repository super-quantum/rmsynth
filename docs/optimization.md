# 6. Optimization pipeline (`optimizer.py`)

The `Optimizer` class is the user‑facing entry point for “take a Clifford+phase circuit, and try to reduce its T‑count (and maybe depth) using Reed–Muller decoding.” It sits on top of:

* the circuit/phase‑polynomial primitives in `core.py`,
* the decoding strategies and policies in `decoders.py`, and
* the autotuner in `autotune.py`.



At a high level, the pipeline is:

1.  Extract the phase polynomial from a circuit.
2.  Turn its $\mathbb{Z}_8$ coefficients into a punctured binary word (oddness).
3.  Choose a decoding strategy and parameters.
4.  Decode in $\mathrm{RM}(r, n)$ to get a codeword and monomial set.
5.  Lift that correction back to $\mathbb{Z}_8$, apply it to the coefficients, and re‑synthesize.
6.  Optionally check consistency contracts and return an `OptimizeReport`.

The rest of this section unpacks how that is wired up.

## 6.1 Effort and parameter mapping

The library tries to make “how hard should you try?” a single knob. Internally, several helper functions interpret that knob into concrete decoder parameters.

`_beam_from_effort(effort)` turns an integer effort level into a beam/list size. If `effort` is `None`, the default is a beam of 8. Otherwise the value is clamped to the range 1–5 and mapped to:

* effort 1 → beam 2
* effort 2 → beam 4
* effort 3 → beam 8
* effort 4 → beam 16
* effort 5 → beam 32

The pattern is simply `1 << effort`, with the clamping handling out‑of‑range values.

`_rpa_iters_from_effort(e)` similarly maps an effort level into the number of RPA iterations:

* 1 → 1 iteration
* 2 → 2 iterations
* 3 → 2 iterations
* 4 → 3 iterations
* 5 → 3 iterations

So mid‑range efforts already get a couple of RPA passes; going to the very highest efforts gives you more iterations but doesn’t grow unbounded.

For SNAP (local refinement), `_snap_params_from_effort(e)` chooses:

* a local search radius `snap_t` (whether to consider size‑1 or size‑2 or size‑3 subsets),
* a pool size `snap_pool`, and
* whether strong (branch‑and‑bound) SNAP should be considered a natural default.

When `e` is `None`, it returns `(2, 16, False)`, i.e. size‑2 SNAP on a pool of 16 rows with no strong SNP. For explicit effort values:

* effort 1 → `(1, 8, False)` – very light, only size‑1.
* effort 2 → `(2, 12, False)`.
* effort 3 → `(2, 16, False)`.
* effort 4 → `(2, 24, True)` – larger pool and strong SNAP is the default.
* effort 5 → `(3, 24, True)` – allow size‑3 local moves and strong SNAP.

Those helpers are used in `_choose` to avoid duplicating “effort → parameters” logic for each decoder flavor.

Finally, there’s a special parsing helper for one specific use case: “I don’t care about effort, just make it run in about X milliseconds.”

```python
def _parse_auto_latency(effort: Optional[object]) -> Optional[float]:
    ...
```

If `effort` is a string of the form `"auto-latency-<Xms>"` (whitespace ignored, `ms` suffix optional), this returns the latency budget `X` as a `float`. For anything else (`None`, integers, other strings), it returns `None`.

This is the hook that lets you write:
```python
Optimizer(effort="auto-latency-3ms", ...)
```
and have the optimizer delegate parameter choice to the autotuner rather than to fixed hard‑coded tables.

## 6.2 Cost model placeholder (`CostModel`)

There is a small `CostModel` dataclass:

```python
@dataclass
class CostModel:
    bitplane_cost: Dict[int, float] = None
    def __post_init__(self):
        if self.bitplane_cost is None:
            self.bitplane_cost = {0: 1.0}
```

Right now it is a placeholder for future work. The idea is that in multi‑order and bitplane‑aware optimizations, you might want to assign different relative costs to different bitplanes or layers (e.g., T vs S vs Z planes, or different architectural penalties). For the current Clifford+T pipeline, this is not yet wired into the decoder decisions, so `bitplane_cost` defaults to a trivial `{0: 1.0}`.

If we were to extend the library, this is where per‑plane or per‑bit cost models would be introduced, and then threaded into `decoders`’ policy logic or into `rotk.py`’s multi‑order routines.

## 6.3 The `Optimizer` class

`Optimizer` is what you construct in user code:

```python
from rmsynth import Optimizer

opt = Optimizer(decoder="rpa", effort=3, policy="distance+depth", policy_lambda=5)
new_circ, rep = opt.optimize(circ)
```

The constructor takes a fairly rich set of knobs, but most have sensible defaults.

* **`decoder`**: which top‑level strategy to use. If you pass `None`, the constructor will choose `"auto"` when the C++ extension is available, and `"ml-exact"` otherwise.

* **`effort`**: an integer 1–5 or a string `"auto-latency-<Xms>"`.
  Integer effort is interpreted via `_beam_from_effort`, `_rpa_iters_from_effort`, `_snap_params_from_effort`. The special string is reserved for autotuning (see §6.4).

* **`list_size`**: explicit beam size overriding the mapping from `effort`. If you set this, `_beam_from_effort` is only used when `list_size` is `None`.

* **`chase_t`, `chase_limit`**: parameters for `"dumer-list-adv"` (Dumer‑List‑Chase). These control how many reliability flips and how deep the chase search may go.

* **SNAP / RPA parameters**:
    * `snap_effort`: separate effort level for SNAP; if omitted, the main `effort` is reused.
    * `snap_t`, `snap_pool`: explicit overrides for the local SNAP parameters; if left `None`, they are derived from `snap_effort`.
    * `snap_strong`: whether strong branch‑and‑bound SNAP is a default; if `None`, the default comes from `_snap_params_from_effort`.
    * `snap_time_ms`, `snap_node_limit`: soft limits that are passed down to `snap_branch_and_bound_bits`.
    * `rpa_iters`: how many RPA iterations to use (if you don’t want the effort mapping).

* **OSD options**:
    * `osd_top`: how many beam candidates to feed into `beam-osd*` strategies; defaults to `min(list_size, 8)` when not specified.

* **Policy knobs**:
    * `policy`: textual policy name to balance distance vs depth, understood by `_lambda_from_policy` and `policy_decide` (see §5.5).
    * `policy_lambda`: numeric weight used for `"distance+depth"` style policies.
    * `depth_tradeoff`: legacy explicit guard parameter; still supported and forwarded directly into `decode_rm`. If you set both `policy` and `depth_tradeoff`, the mapping logic in `decoders` will combine them.

* **Autotuner knobs**:
    * `autotune_selector`: controls which selector the autotuner uses (`"quality-under-target"` vs `"pareto"`, matching `autotune.calibrate_single`).
    * `autotune_pareto_dist`, `autotune_pareto_lat`: Pareto slack and latency factor for the autotuner when using the Pareto selector.

* **`check_contracts`**: if `True`, the optimizer will run strict consistency checks from `contracts.py` after every optimization; if `False`, it will call the cheaper `maybe_check_all`.

The constructor stores all of these as `self.*` attributes and also initializes two introspection fields:

* `self.last_decoder_used`: the final strategy string (`"dumer"`, `"rpa-adv"`, etc.) chosen in the last call to `optimize`.
* `self.last_params_used`: the exact keyword dictionary passed into `decode_rm` (list size, RPA iterations, SNAP knobs, policy parameters).

The CLI uses these to print human‑readable summaries of what actually ran; you can also use them in tests or notebooks to inspect the optimizer’s decisions.

## 6.4 Strategy selection: `_choose(...)`

The method `Optimizer._choose(self, n: int, before_t: int) -> (strategy, kwargs)` encapsulates all of the high‑level decision making: given the number of qubits `n` and the *current* T‑count `before_t`, it decides which decoder strategy to call and with which parameters.

The docstring says it explicitly:

Select a decoder strategy + kwargs, honoring:

- explicit decoder choices first,
- auto‑latency (if requested) → rpa‑adv with autotuned params,
- 'auto' policy: small n → dumer; medium n (n=6) → dumer‑list; large n or high T → rpa‑adv.



First it builds a `policy_kwargs` dictionary. If either `policy` or `depth_tradeoff` was set on the optimizer, it includes:

```python
policy_kwargs = {
    "policy": self.policy,
    "policy_lambda": self.policy_lambda,
    "depth_tradeoff": self.depth_tradeoff,
}
```

Those are then threaded into any strategy that calls `decode_rm`, so all internal comparisons share the same distance–depth semantics.

### Auto‑latency path

Before looking at `self.decoder`, `_choose` checks whether `effort` encodes a latency target:

```python
auto_ms = _parse_auto_latency(self.effort)
if auto_ms is not None and _load_rmcore() is not None:
    params = suggest_params(..., target_ms=auto_ms, ...)
    ...
    return "rpa-adv", kwargs
```

If `_parse_auto_latency` returns a time budget and the C++ extension is available, `_choose` delegates to the autotuner:

1.  It calls `autotune.suggest_params(n, before_t, auto_ms, ...)`, passing through the same policy and selector knobs you gave to the optimizer.
2.  `suggest_params` returns an `EffortParams` dataclass with fields like `beam`, `rpa_iters`, `snap_pool`, etc.
3.  `_choose` then builds a `kwargs` dict for `"rpa-adv"`:
    ```python
    kwargs = {
        "list_size": params.beam,
        "rpa_iters": params.rpa_iters,
        "snap": True,
        "snap_t": params.snap_t,
        "snap_pool": params.snap_pool,
        "snap_strong": params.snap_strong,
        **policy_kwargs
    }
    ```

In this mode, `effort` is essentially an instruction to autotune for a target latency, not a discrete level. The choice of decoder is fixed to `"rpa-adv"`; beam and SNAP parameters are derived from real timing measurements cached in `autotune.py`.

### Explicit decoders

If the auto‑latency path didn’t trigger, `_choose` honours your `decoder` string more or less literally.

* `"ml-exact"` → returns `("ml-exact", {})`. `decode_rm` will use the brute‑force ML decoder from `core.py`.
* `"dumer"` → `("dumer", {})`.
* `"dumer-list"` → `("dumer-list", {"list_size": ls, **policy_kwargs})`, where `ls` is either your explicit `list_size` or derived from `_beam_from_effort(self.effort)`.
* `"dumer-list-adv"` → maps to `("dumer-list-chase", {list_size, chase_t, chase_limit, **policy_kwargs})`. This is the more advanced Dumer‑List‑Chase strategy.
* `"osd1"`, `"osd2"`, `"osd3"` → return the same string as strategy, plus a `list_size`. These are handled in `decode_rm` by calling `osd_decode` on top of a Dumer‑List baseline.
* `"beam-osd1"`, `"beam-osd2"`, `"beam-osd3"` → similar, but also include an `osd_top` parameter (defaulting to `min(ls, 8)`) so `decode_rm` knows how many beam seeds to refine with OSD.

For the RPA families, `_choose` sets up everything needed for `"rpa-adv"` and `"rpa2"` based on effort and SNAP parameters:

* `"rpa"`:
    1.  Compute `ls` from `list_size` or effort.
    2.  Compute `it` from `rpa_iters` or `_rpa_iters_from_effort(self.effort)`.
    3.  Compute `(st, sp, strong_default)` from `_snap_params_from_effort`, using `snap_effort` if set, otherwise `effort`.
    4.  Override `st` and `sp` with explicitly provided `snap_t` and `snap_pool` if any.
    5.  Decide `strong` based on `snap_strong` or the `strong_default`.

  Then return:
    ```python
    ("rpa-adv",
     {"list_size": ls, "rpa_iters": it, "snap": True,
      "snap_t": st, "snap_pool": sp, "snap_strong": strong,
      "snap_time_ms": ..., "snap_node_limit": ...,
      **policy_kwargs})
    ```
* `"rpa2"` follows exactly the same mapping, but returns `"rpa2"` as the strategy instead of `"rpa-adv"`.

So `"rpa"` is really just a shorthand for “RPA‑1 with SNAP and possibly strong SNAP, using effort‑based defaults.”

### `"auto"` decoder policy

If you set `decoder="auto"`, `_choose` inspects both the dimension `n` and the current T‑count `before_t` and follows a simple three‑regime heuristic (assuming the C++ extension is present):

1.  If `_load_rmcore()` fails, `"auto"` degrades to `"ml-exact"` because no fast decoders are available.
2.  For larger or more complex circuits (`n >= 7` or `before_t >= 24`), it chooses `"rpa-adv"` with parameters derived from effort and SNAP settings. This is the “go heavy” regime: use RPA and SNAP because the search space is large and big improvements are likely.
3.  For medium sized instances (`n >= 6` or `before_t >= 16` but not in the previous case), it chooses `"dumer-list"` with an effort‑derived `list_size`. The idea is that Dumer‑List alone is strong enough here and cheaper than running full RPA.
4.  For very small circuits, it uses plain `"dumer"` with no list decoding and no RPA.

If `decoder` is something else that `_choose` does not recognize, it falls back to returning `(self.decoder, {})`, letting `decode_rm` raise a `ValueError` if the strategy string is truly invalid. This makes it easy to experiment with new strategies by adding them first to `decoders.py` and then passing the name through `Optimizer`.

## 6.5 Main entrypoint: `optimize(circ)`

The method `Optimizer.optimize(self, circ: Circuit) -> (Circuit, OptimizeReport)` is the core of the user‑visible pipeline. It takes a `Circuit` (as defined in `core.py`) and returns:
* a new optimized `Circuit`, and
* an `OptimizeReport` summarizing what happened.

Let’s walk through it.

1.  **Extract phase coefficients and T‑count.**
    ```python
    n   = circ.n
    a   = extract_phase_coeffs(circ)
    vec = coeffs_to_vec(a, n)
    before_t = t_count_of_coeffs(vec)
    ```
    `extract_phase_coeffs` tracks how CNOTs redistribute phase forms and collects all phase‑gate exponents (mod 8) into a dict `{mask -> k}`. `coeffs_to_vec` then maps that dict into a fixed‑order $\mathbb{Z}_8$ vector `vec` of length $(2^n-1)$. `t_count_of_coeffs` counts how many entries in `vec` are odd; that is exactly the T‑count of the circuit.

2.  **Determine RM parameters and build the “word” to decode.**
    The code sets `r = n - 4`. This is the RM code order used throughout the library: for `n` qubits, we work in $\mathrm{RM}(n-4, n)$ as in the reference construction. It also notes the length `L = len(vec)` and builds the punctured binary word:
    ```python
    w_bits = [1 if (v & 1) else 0 for v in vec]
    ```
    This is the oddness pattern you would like to correct using an RM decoder.

3.  **Choose a strategy.**
    `_choose` is called with `(n, before_t)`:
    ```python
    strategy, kwargs = self._choose(n, before_t)
    self.last_decoder_used = strategy
    self.last_params_used  = kwargs
    ```
    At this point, `strategy` is a string understood by `decode_rm` (`"dumer"`, `"rpa-adv"`, `"auto-latency"`→`"rpa-adv"`, etc.), and `kwargs` contains the list size, number of RPA iterations, SNAP knobs, and policy settings.

4.  **Decode in $\mathrm{RM}(r, n)$ (or handle trivial $r < 0$).**
    If `r < 0` (which only happens for very small `n`, where $\mathrm{RM}(n-4,n)$ is trivial), the code skips decoding:
    ```python
    if r < 0:
        code_bits = [0] * ((1 << n) - 1)
        selected  = []
        dist      = sum(w_bits)
    else:
        code_bits, selected, dist = decode_rm(w_bits, n, r, strategy, **kwargs)
    ```
    In the nontrivial case, `decode_rm` performs all the heavy lifting described in §5: Dumer, RPA, OSD, SNAP, GL search, etc. The result is a punctured RM codeword `code_bits`, a list of selected monomials `selected` (compatible with `get_monom_list(n,r)`), and the punctured Hamming distance `dist`.

5.  **Lift the correction and apply it to the coefficients.**
    The selected monomials describe a correction vector `c` in the phase‑polynomial domain. `lift_to_c` constructs the $\mathbb{Z}_8$ vector of that correction:
    ```python
    c_vec   = lift_to_c(selected, n)
    vec_opt = add_mod8_vec(vec, c_vec)
    ```
    Here, `lift_to_c` adds 1 (mod 8) along every position where a chosen monomial evaluates to 1, and `add_mod8_vec` adds that correction pointwise modulo 8.

    The key invariant is that `vec_opt` has odd coefficients exactly at the positions that correspond to $w_{bits} \oplus code_{bits}$; the T‑count is therefore:
    ```python
    after_t = t_count_of_coeffs(vec_opt)
    ```
    By construction, `after_t` is either `<= before_t` or (in rare cases) slightly worse only if you asked for a distance‑depth trade‑off that allows it (and even then this is bounded by the policy).

6.  **Re‑synthesize the optimized circuit.**
    The new phase polynomial `vec_opt` is turned back into a circuit:
    ```python
    new_circ = synthesize_from_coeffs(vec_opt, n)
    ```
    This uses the same monomial‑to‑CNOT‑gadget construction described in §3.4. If you want depth‑aware synthesis, you can call `synthesize_with_schedule` at a higher level with the same `vec_opt`.

7.  **Run consistency contracts (optional).**
    If you constructed the optimizer with `check_contracts=True`, it will run the strict contract checks:
    ```python
    from .contracts import check_all
    check_all(vec, n, r, code_bits, selected, dist, strict=True)
    ```
    Otherwise, it calls `maybe_check_all(...)`, which only runs the contracts when the `RM_CHECKS` environment variable is set. The contracts verify three things:
    * all `selected_monomials` have degree $\le r$ (they lie in $\mathrm{RM}(r,n)$),
    * `code_bits` agrees with the XOR of generator rows corresponding to `selected_monomials`, and
    * the reported `dist` agrees with the T‑count change implied by `code_bits` and the original coefficients.
      If any of these fail in strict mode, an `AssertionError` is raised. This is extremely useful for testing and for catching subtle decoder bugs.

8.  **Compute a signature and assemble the report.**
    Finally, the optimized coefficient vector is hashed:
    ```python
    import hashlib
    sig = hashlib.sha256(bytes(vec_opt)).hexdigest()
    ```
    and used to construct an `OptimizeReport`:
    ```python
    rep = OptimizeReport(
        n=n,
        before_t=before_t,
        after_t=after_t,
        distance=dist,
        r=r,
        bitlen=L,
        selected_monomials=selected,
        signature=sig,
    )
    ```
    `OptimizeReport.summary()` prints a compact one‑line summary of the form:
    > `[rmsynth] n=6, r=2, length=63: T-count 32 -> 18 (distance=14). Signature=abcd1234...`

    The SHA‑256 signature acts as a stable identifier for the optimized phase polynomial: if you run the same optimizer settings and get the same signature, you know you got the same coefficients, even if the internal decoding path used different ties or permutations.

Putting it all together, `Optimizer.optimize` guarantees:
* the output circuit implements the same unitary as the input,
* its phase polynomial lies in the original affine space but shifted by a valid $\mathrm{RM}(r,n)$ codeword,
* all RM contracts are satisfied (when enabled), and
* the T‑count and/or T‑depth are improved according to the policy you specified, subject to well‑defined guardrails.