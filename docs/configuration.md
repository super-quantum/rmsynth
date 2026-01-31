# 12. Environment variables and configuration

The library exposes a number of environment variables to tune performance, tie‑breaking, and debugging without changing code.

## 12.1 Depth and scheduling

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RM_DEPTH_MODE` | `est` | Controls depth method used in `decoders._depth_from_solution`. <br>• `"est"`: use fast estimator `rmcore.tdepth_from_punctured`. <br>• `"sched"`: try exact T-depth scheduler. |
| `RM_SCHED_BUDGET` | `20000` | Search budget (nodes) for the T-depth scheduler when `RM_DEPTH_MODE="sched"`. Higher values yield better colorings but take longer. |
| `RM_TDEPTH_COLOR_NODES`| `500000` | Max nodes explored by DSATUR in `tdepth.cpp`. If limit hit, falls back to greedy. |
| `RM_TDEPTH_COLOR_MS` | `50` | Max wall-clock time (ms) for DSATUR. |

## 12.2 Parallelism and determinism

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RM_PARALLEL` | `ON` | If unset, parallelism (TBB/OpenMP) is enabled where available. Set to `"0"`, `"false"`, or `"no"` to force sequential execution. |
| `RM_DETERMINISTIC` | `OFF` | If set to a truthy string, list decoders break ties using a deterministic hash key (distance + hash) instead of `std::nth_element`. |

## 12.3 Autotuner

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RM_AUTOTUNE_CACHE` | `~/.rmsynth_autotune.json` | Path to the JSON cache file. |
| `RM_AUTOTUNE_SELECTOR` | `quality-under-target` | Strategy for `calibrate_single`. Options: `"quality-under-target"`, `"pareto"`. |
| `RM_AUTOTUNE_PARETO_DIST`| `1` | Pareto distance slack. |
| `RM_AUTOTUNE_PARETO_LAT` | `1.10` | Pareto latency factor. |
| `RM_AUTOTUNE_TRIALS` | `4` | Number of synthetic words sampled per configuration. |
| `RM_AUTOTUNE_SEED` | `123` | RNG seed for autotuning. |

## 12.4 Contracts and checks

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RM_CHECKS` | `OFF` | If set to a truthy value, enables contract checking in `Optimizer`, asserting invariants on every run. |

## 12.5 SNAP, RPA, GL, OSD

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RM_SNAP_BB_MS` | `15` | Max time (ms) for branch‑and‑bound "strong SNAP". |
| `RM_SNAP_BB_NODES` | `100000` | Max nodes for branch‑and‑bound SNAP. |
| `RM_SNAP_BB_POOL` | `snap_pool` | Pool size for branch-and-bound SNAP (defaults to the `snap_pool` argument). |
| `RM_RPA2_MAX_PERMS` | `unset` | Limits permutations in RPA-2 strategies. If unset, uses a modest default set. |
| `RM_OSD2_MAX_PAIRS` | `unset` | Caps number of pairs checked in OSD-2. |
| `RM_OSD3_MAX_TRIPLES` | `unset` | Caps number of triples checked in OSD-3. |