from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional
import json, os, time, random, math

from .decoders import decode_rm, _load_rmcore

# Data model for autotuning

@dataclass
class EffortParams:
    """
    Decoder effort knobs that we autotune.

    beam        : list_size (beam width) for Dumer-list / RPA
    chase_limit : max number of Chase flip patterns (currently fixed in grid)
    rpa_iters   : number of RPA-1 iterations
    snap_pool   : pool size for SNAP neighborhood (number of rows to consider)
    snap_t      : SNAP subset size (1 or 2 in current grid)
    snap_strong : whether to enable strong branch-and-bound SNAP
    """
    beam: int
    chase_limit: int
    rpa_iters: int
    snap_pool: int
    snap_t: int = 2
    snap_strong: bool = False

@dataclass
class MappingEntry:
    """
    Cached result of a calibration run for a particular bucket.
    """
    params: EffortParams
    median_ms: float   # median runtime over calibration trials
    mean_ms: float     # mean runtime
    trials: int        # number of calibration trials

# Cache path / helpers

def _cache_path() -> str:
    """
    Pick a JSON file for autotune results.

    RM_AUTOTUNE_CACHE, if set, overrides the default:
      $HOME/.rmsynth_autotune.json
    """
    p = os.environ.get("RM_AUTOTUNE_CACHE", "").strip()
    if p:
        return p
    home = os.path.expanduser("~")
    return os.path.join(home, ".rmsynth_autotune.json")

def _load_cache() -> Dict[str, dict]:
    """
    Load the autotune cache from disk, returning an empty dict on failure.
    """
    fn = _cache_path()
    if not os.path.exists(fn):
        return {}
    try:
        with open(fn, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(d: Dict[str, dict]) -> None:
    """
    Save the autotune cache best-effort, ignore I/O errors.
    """
    fn = _cache_path()
    try:
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, sort_keys=True)
    except Exception:
        pass

def clear_cache() -> None:
    """
    Public helper to nuke the autotune cache on disk.
    """
    fn = _cache_path()
    try:
        if os.path.exists(fn):
            os.remove(fn)
    except Exception:
        pass

# Bucketing: group instances by "complexity"

def _bucket_key(n: int, pre_t: int, L: int,
                selector: str, pareto_dist_slack: int, pareto_latency_factor: float) -> str:
    """
    Map (n, pre_t, selector, pareto params) to a cache key.

    - n:          number of qubits
    - pre_t:      pre-optimization T-count (odd positions)
    - L:          length of punctured vector (2^n - 1), used to cap buckets
    - selector:   strategy used to pick the winner config
    - pareto_*:   selector-specific knobs

    We bucket pre_t coarsely in bins of size 8 to recycle calibration
    across instances of similar T-weight.
    """
    # bucket pre_t in bins of 8 (cap at L)
    lo = (max(0, pre_t) // 8) * 8
    hi = min(L, lo + 7)
    # round latency factor to 3 dp in key to avoid float noise in filenames
    lat_tag = f"{pareto_latency_factor:.3f}"
    return f"n{n}/pre{lo}-{hi}/sel:{selector}/pd:{pareto_dist_slack}/pl:{lat_tag}"

# Candidate grid (search space over effort parameters)

def _candidate_grid(target_ms: float) -> List[EffortParams]:
    """
    Build a small grid of EffortParams to sweep over for calibration.

    The grid trades off beam width, RPA iterations, and SNAP pool size.
    The ordering of the grid is biased depending on the latency target:
      - For very tight targets, we try "cheaper" configs first.
      - For looser targets, we explore higher-quality configs earlier.
    """
    beams = [4, 8, 16, 32]
    iters = [1, 2, 3]
    pools = [8, 12, 16, 24]

    grid: List[EffortParams] = []
    for b in beams:
        for it in iters:
            for sp in pools:
                grid.append(EffortParams(beam=b, chase_limit=16, rpa_iters=it, snap_pool=sp, snap_t=2, snap_strong=False))

    if target_ms <= 2.0:
        grid.sort(key=lambda p: (p.beam, p.rpa_iters, p.snap_pool))
    else:
        grid.sort(key=lambda p: (p.rpa_iters, p.beam, p.snap_pool))
    return grid

# Sampling synthetic workloads for calibration

def _make_sample_words(n: int, flips: int, L: int, trials: int, seed: int) -> List[List[int]]:
    """
    Generate synthetic punctured oddness vectors for calibration.

    Each sample is a length-L bit list with exactly 'flips' ones (clipped
    to [0, L]), chosen uniformly at random.  This approximates the
    distribution of odd positions near the given pre-T count.
    """
    flips = max(0, min(flips, L))
    rng = random.Random(seed)
    words: List[List[int]] = []
    base = list(range(L))
    for _ in range(trials):
        idxs = rng.sample(base, flips) if flips > 0 else []
        bits = [0]*L
        for i in idxs:
            bits[i] = 1
        words.append(bits)
    return words

# Measuring the runtime + quality of a single configuration

def _measure_config(n: int, r: int, L: int, words: List[List[int]], cfg: EffortParams,
                    policy: Optional[str], policy_lambda: Optional[int]) -> Tuple[float, float, float]:
    """
    Run the decoder on a batch of synthetic words and measure:

      - median_ms : median runtime in milliseconds
      - mean_ms   : mean runtime in milliseconds
      - mean_dist : mean Hamming distance (proxy for "quality")

    This uses the "rpa-adv" strategy with the given EffortParams.
    """
    import statistics as st
    times: List[float] = []
    dists: List[int] = []

    # One warm-up call to avoid counting import / JIT / cache effects.
    if words:
        decode_rm(
            words[0], n, r, "rpa-adv",
            list_size=cfg.beam,
            rpa_iters=cfg.rpa_iters,
            snap=True, snap_t=cfg.snap_t, snap_pool=cfg.snap_pool,
            snap_strong=cfg.snap_strong,
            policy=policy, policy_lambda=policy_lambda,
        )

    # Timed calls.
    for w in words:
        t0 = time.perf_counter()
        _code_bits, _monoms, dist = decode_rm(
            w, n, r, "rpa-adv",
            list_size=cfg.beam,
            rpa_iters=cfg.rpa_iters,
            snap=True, snap_t=cfg.snap_t, snap_pool=cfg.snap_pool,
            snap_strong=cfg.snap_strong,
            policy=policy, policy_lambda=policy_lambda,
        )
        t1 = time.perf_counter()
        times.append((t1 - t0)*1000.0)
        dists.append(dist)

    median_ms = float(st.median(times)) if times else 0.0
    mean_ms = float(st.mean(times)) if times else 0.0
    mean_dist = float(st.mean(dists)) if dists else 0.0
    return median_ms, mean_ms, mean_dist

# Public API

def calibrate_single(n: int, pre_t: int, target_ms: float,
                     trials: int = 4,
                     seed: int = 123,
                     policy: Optional[str] = None,
                     policy_lambda: Optional[int] = None,
                     selector: Optional[str] = None,               # "quality-under-target" | "pareto"
                     pareto_dist_slack: Optional[int] = None,      # e.g., 1
                     pareto_latency_factor: Optional[float] = None # e.g., 1.10
                     ) -> MappingEntry:
    """
    Calibrate a single (n, pre_t bucket) to hit <= target_ms.

    selector == "quality-under-target" (default):
        Among configs with median <= target_ms, pick lowest mean distance;
        if none meet the target, pick the fastest median overall.

    selector == "pareto":
        Let f = fastest median across the grid. Consider pool with
        median <= min(target_ms, f*pareto_latency_factor).
        In that pool, find best mean distance d*, then keep configs with
        mean distance <= d* + pareto_dist_slack; choose the one with
        smallest median among them. On empty pool, fall back to fastest.

    The final MappingEntry is cached for reuse by suggest_params().
    """
    if _load_rmcore() is None:
        # If the native extension is missing, decoding will be extremely
        # slow and autotuning is not meaningful. Return a sentinel entry.
        best = MappingEntry(
            params=EffortParams(beam=4, chase_limit=16, rpa_iters=1, snap_pool=8, snap_t=2, snap_strong=False),
            median_ms=1e9, mean_ms=1e9, trials=0
        )
        return best

    # defaults / env overrides
    sel = (selector or os.environ.get("RM_AUTOTUNE_SELECTOR", "quality-under-target")).strip().lower()
    p_dist = int(pareto_dist_slack if pareto_dist_slack is not None else int(os.environ.get("RM_AUTOTUNE_PARETO_DIST", "1")))
    p_lat  = float(pareto_latency_factor if pareto_latency_factor is not None else float(os.environ.get("RM_AUTOTUNE_PARETO_LAT", "1.10")))

    L = (1 << n) - 1
    r = n - 4
    key = _bucket_key(n, pre_t, L, sel, p_dist, p_lat)
    cache = _load_cache()

    words = _make_sample_words(n, pre_t, L, trials, seed)
    grid = _candidate_grid(target_ms)

    # records: list of (median, mean, mean_dist, EffortParams)
    records: List[Tuple[float,float,float,EffortParams]] = []
    fastest: Tuple[float,float,float,EffortParams] | None = None
    met_target: List[Tuple[float,float,float,EffortParams]] = []

    # Sweep over candidate configurations and measure performance.
    for cfg in grid:
        med, mean, mdist = _measure_config(n, r, L, words, cfg, policy, policy_lambda)
        rec = (med, mean, mdist, cfg)
        records.append(rec)
        if fastest is None or med < fastest[0]:
            fastest = rec
        if med <= target_ms:
            met_target.append(rec)

    assert fastest is not None
    winner = fastest  # default winner is the fastest config

    # Apply selector policy.
    if sel == "quality-under-target":
        if met_target:
            # Rank configs that meet target by (mean distance, median, mean).
            met_target.sort(key=lambda t: (t[2], t[0], t[1]))
            winner = met_target[0]
    elif sel == "pareto":
        # Pareto-like pool: restrict latency, then relax on distance.
        lat_ref = min(target_ms, fastest[0] * p_lat)
        pool = [rec for rec in records if rec[0] <= lat_ref]
        if pool:
            best_dist = min(rec[2] for rec in pool)
            # keep within slack on distance
            pool2 = [rec for rec in pool if rec[2] <= best_dist + p_dist]
            # from those, pick smallest median (tie-break by distance, mean)
            pool2.sort(key=lambda t: (t[0], t[2], t[1]))
            winner = pool2[0]
        # else: keep fastest

    med, mean, mdist, cfg = winner
    entry = MappingEntry(params=cfg, median_ms=med, mean_ms=mean, trials=trials)

    # Store the calibration result in the cache.
    cache[key] = {
        "params": asdict(cfg),
        "median_ms": entry.median_ms,
        "mean_ms": entry.mean_ms,
        "trials": entry.trials,
        "target_ms": target_ms,
        "policy": policy,
        "policy_lambda": policy_lambda,
        "selector": sel,
        "pareto_dist_slack": p_dist,
        "pareto_latency_factor": p_lat,
    }
    _save_cache(cache)
    return entry

def suggest_params(n: int, pre_t: int, target_ms: float,
                   policy: Optional[str] = None,
                   policy_lambda: Optional[int] = None,
                   selector: Optional[str] = None,
                   pareto_dist_slack: Optional[int] = None,
                   pareto_latency_factor: Optional[float] = None) -> EffortParams:
    """
    Return EffortParams for the (n, pre_t) bucket, calibrating if needed.

    On cache hit:
        - the saved params are returned immediately.

    On cache miss:
        - calibrate_single(...) is invoked to populate the cache.
    """
    sel = (selector or os.environ.get("RM_AUTOTUNE_SELECTOR", "quality-under-target")).strip().lower()
    p_dist = int(pareto_dist_slack if pareto_dist_slack is not None else int(os.environ.get("RM_AUTOTUNE_PARETO_DIST", "1")))
    p_lat  = float(pareto_latency_factor if pareto_latency_factor is not None else float(os.environ.get("RM_AUTOTUNE_PARETO_LAT", "1.10")))

    L = (1 << n) - 1
    key = _bucket_key(n, pre_t, L, sel, p_dist, p_lat)
    cache = _load_cache()

    if key in cache:
        # Rehydrate EffortParams from the cached dict, using sensible defaults
        # if some keys are missing (for forward compatibility).
        p = cache[key].get("params", {})
        return EffortParams(
            beam=int(p.get("beam", 8)),
            chase_limit=int(p.get("chase_limit", 16)),
            rpa_iters=int(p.get("rpa_iters", 2)),
            snap_pool=int(p.get("snap_pool", 16)),
            snap_t=int(p.get("snap_t", 2)),
            snap_strong=bool(p.get("snap_strong", False)),
        )

    # Cache miss -> run calibration once.
    entry = calibrate_single(
        n=n, pre_t=pre_t, target_ms=target_ms,
        trials=int(os.environ.get("RM_AUTOTUNE_TRIALS", "4")),
        seed=int(os.environ.get("RM_AUTOTUNE_SEED", "123")),
        policy=policy, policy_lambda=policy_lambda,
        selector=sel, pareto_dist_slack=p_dist, pareto_latency_factor=p_lat,
    )
    return entry.params
