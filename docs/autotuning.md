# 10. Autotuning and benchmarking

## 10.1 Autotuner (`autotune.py`)

For RPA-based decoders, there are several knobs that directly impact runtime and quality:

1.  Beam size for Dumer-list (`list_size` / “beam”).
2.  Number of RPA iterations.
3.  Parameters for local SNAP refinement.

Picking them manually for each system and workload is tedious. The `autotune.py` module automates this by running small synthetic experiments and caching the results.



### Effort parameter model

The autotuner uses two small dataclasses:

`EffortParams` collects the knobs for a particular configuration:

* `beam` – list size for Dumer-list (and RPA beam).
* `chase_limit` – reserved for potential chase-based variants.
* `rpa_iters` – number of RPA-1 iterations.
* `snap_pool`, `snap_t`, `snap_strong` – SNAP neighborhood size and whether to enable branch-and-bound “strong SNAP”.

`MappingEntry` records the measured performance for a specific bucket:

* `params` – the chosen `EffortParams`.
* `median_ms`, `mean_ms` – timing statistics over a small sample.
* `trials` – how many sample words were used.

These are stored as plain JSON via `asdict()` so the cache is human-readable.

### Cache format and bucketing

The cache lives in a single JSON file:

* Path is either `RM_AUTOTUNE_CACHE` (if set) or `~/.rmsynth_autotune.json`.
* `_load_cache()` and `_save_cache()` read/write that file; errors are swallowed so autotuning never crashes the main program.

Each calibration result is keyed by a string of the form:
`n{n}/pre{lo}-{hi}/sel:{selector}/pd:{pareto_dist_slack}/pl:{pareto_latency_factor}`

where:

* `n` is the number of qubits.
* `pre{lo}-{hi}` is a bucket for the pre-optimization T-count (odd weight), grouped in bins of 8 and capped at $L = 2^n - 1$.
* `selector` is the selection strategy (`quality-under-target` or `pareto`).
* `pd` and `pl` encode the Pareto parameters so different tradeoff policies don’t collide.

This design ensures that if you ask for the same problem size, pre‑T range, and policy, you get consistent `EffortParams` across runs.

### Candidate grid and measurement

The core of autotuning builds a fixed “grid” of candidate `EffortParams`:

`_candidate_grid(target_ms)` enumerates beams in `{4, 8, 16, 32}`, RPA iterations in `{1, 2, 3}`, and SNAP pool sizes in `{8, 12, 16, 24}`, always with `chase_limit=16` and `snap_t=2`.

The grid is sorted differently depending on `target_ms` so that very tight latency budgets prioritize small beams and few iterations first.

Synthetic test words are created by `_make_sample_words(n, flips, L, trials, seed)`:

* For each trial, it chooses `flips` random positions in `[0..L-1]` and sets those bits to 1, others to 0.
* This models oddness vectors with a given sparsity.

`_measure_config(...)` runs a full decode loop for each synthetic word with a particular `EffortParams`:

1.  It uses `decode_rm(..., "rpa-adv", list_size=beam, rpa_iters=rpa_iters, snap=..., ...)`.
2.  First call is a warm‑up (not timed) to amortize import and JIT overhead.
3.  It then records the per-trial latency (in milliseconds) and the resulting Hamming distances, and returns `(median_ms, mean_ms, mean_dist)`.

### Selection strategies

`calibrate_single(...)` ties everything together for one bucket:

1.  It checks that `rmcore` is available; otherwise it returns a dummy entry marking the configuration as unusable (very large median, zero trials).
2.  It reads selector and Pareto parameters from arguments or environment:
    * `RM_AUTOTUNE_SELECTOR` (default `"quality-under-target"`).
    * `RM_AUTOTUNE_PARETO_DIST` (default 1).
    * `RM_AUTOTUNE_PARETO_LAT` (default 1.10).

Two selection modes are supported:

**`"quality-under-target"` (default)**
For all configs with `median_ms <= target_ms`, it picks the one with smallest `mean_dist`, breaking ties by `(median_ms, mean_ms)`. If no config meets the target, it simply picks the fastest median.

**`"pareto"`**
Let `f` be the fastest median across all configs. The tuner defines a latency bound `lat_ref = min(target_ms, f * pareto_latency_factor)`. It then:

1.  Keeps only configs with `median_ms <= lat_ref`.
2.  Finds the best `mean_dist` in this pool.
3.  Keeps configurations within `pareto_dist_slack` of this best distance.
4.  Among those, picks the smallest median.

This gives a simple latency–quality Pareto selection while avoiding extreme outliers.
Finally, `calibrate_single` stores the chosen configuration into the cache with all relevant metadata and returns a `MappingEntry`.

`suggest_params(...)` is the user-facing entry point:
It computes the bucket key, looks up the cache, and if present, reconstructs an `EffortParams` object. On cache miss, it calls `calibrate_single` using environment defaults for number of trials (`RM_AUTOTUNE_TRIALS`, default 4) and RNG seed (`RM_AUTOTUNE_SEED`, default 123), then returns the calibrated parameters.
The `Optimizer` uses this through its “auto-latency” mode.

## 10.2 Benchmarks (`bench.py`)

The `bench.py` module provides a small synthetic benchmarking harness that can be used from Python (or tests) without installing the CLI.

### Instance generators
All instance generators produce a length $L = 2^n - 1$ vector over $\mathbb{Z}_8$:

* `gen_near_constant_one(n, flips, rng, randomize_positions=True)`: Starts from the all-ones oddness vector (every coefficient $\equiv 1 \pmod 2$) and flips `flips` random positions to 0. This models “near-identity” circuits. It then encodes this oddness into a $\mathbb{Z}_8$ vector where only the least significant bit is used.
* `gen_xor_two_triples(n, rng)`: Requires $n \ge 6$. It constructs an oddness pattern equal to the XOR of two degree‑3 monomials: $(x_0 \wedge x_1 \wedge x_2) \oplus (x_3 \wedge x_4 \wedge x_5)$. The even part is zero. This is useful to probe behavior on low‑order structured polynomials.
* `gen_random_sparse(n, density, rng)`: Draws odd bits i.i.d. `Bernoulli(density)`, with even bits zero.
* `gen_random_z8(n, density_odd, rng)`: Draws each coefficient as: odd bit `Bernoulli(density_odd)`, plus an even part chosen uniformly from $\{0,2,4,6\}$. This is closer to a generic phase‑polynomial workload.

The dispatcher `_gen_instance(cfg, rng)` chooses among these based on `BenchConfig.gen`.

### Benchmark configuration and result types

`BenchConfig` is a dataclass describing one benchmark run:

* Problem parameters: `n`, `trials`.
* Instance generator and its parameters: `gen`, `flips`, `density`.
* Strategies to test: `strategies` (default: a mix of Dumer, Dumer-list, Chase, RPA beam, and RPA-adv).
* Mode: `"decode"` (default) or `"pipeline"`.
* Effort and tuning: `effort`, `list_size`, `chase_t`, `chase_limit`, `rpa_iters`, `snap_t`, `snap_pool`.
* RNG seed.

`TrialResult` records, for each strategy and each instance:

* `strategy` – name used in `decode_rm` or `Optimizer`.
* `dist` – distance (decode mode) or after‑T (pipeline mode).
* `before_t`, `after_t` – T-count-like quantities before and after the step.
* `time_ms` – runtime in milliseconds.

`Aggregate` holds per-strategy summary statistics:

* Mean, median, stdev, min, max of `dist`.
* Mean `after_t`.
* Mean latency in milliseconds.

### Modes: decode vs pipeline

`run_benchmark(cfg)` orchestrates everything:

It first checks that `rmcore` is available if `cfg.mode != "ml-exact"`. For $n \ge 6$, a pure brute‑force ML decoder would be unacceptably slow, so the benchmark refuses to run without the C++ extension.

For each trial:

1.  It samples a `vec` according to `cfg.gen`.
2.  It sets `r = n - 4`, consistent with the main optimizer.
3.  It loops over `cfg.strategies`.

**In `"decode"` mode:**
It calls `_decode_once(vec, n, r, strategy, cfg)`, which:

* Extracts `w_bits` as the oddness pattern.
* Maps the strategy string to an appropriate `decode_rm` call.
* Measures latency and returns `(dist, elapsed_ms)`.
  It records `before_t` = number of odd entries in `vec`, and `after_t = dist`, so distance is interpreted directly as T-count reduction.

**In `"pipeline"` mode:**
It calls `_pipeline_once(vec, n, strategy, cfg)`, which:

* Synthesizes a circuit from `vec`.
* Wraps it in an `Optimizer` configured with the given strategy.
* Runs `opt.optimize(circ)` and records `before_t`, `after_t` from the report.
  It records those T-counts and the full pipeline latency.

At the end, `run_benchmark`:

1.  Groups `TrialResult` objects by strategy.
2.  Computes `Aggregate` for each group.
3.  Sorts aggregates by `(mean_dist, mean_time_ms)` so “better and faster” strategies rise to the top.

This makes it easy to compare decoders both as raw decoders and as full T-count optimizers.