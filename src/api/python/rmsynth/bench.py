from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import time
import random
import statistics as stats

from .decoders import decode_rm, _load_rmcore
from .core import synthesize_from_coeffs, t_count_of_coeffs

# Instance generators (all return Z₈ vectors of length 2^n - 1)

def gen_near_constant_one(n: int, flips: int, rng: random.Random, randomize_positions: bool = True) -> List[int]:
    """
    Z8 vector where the oddness corresponds to a word that is "almost all ones":
      - Start from w = 1^L.
      - Flip 'flips' positions to 0 (either at random or deterministically).
      - Use w as the LSB and zero even part, so entries are in {0,1}.
    """
    L = (1 << n) - 1
    w = [1] * L
    if randomize_positions:
        idx = list(range(L))
        rng.shuffle(idx)
        for k in idx[: min(flips, L)]:
            w[k] = 0
    else:
        for k in range(min(flips, L)):
            w[k] = 0
    # oddness==w, even part 0 -> Z8 vector with LSB=w
    return [1 if b else 0 for b in w]

def gen_xor_two_triples(n: int, rng: random.Random) -> List[int]:
    """Oddness = (x0&x1&x2) XOR (x3&x4&x5); even padding is 0."""
    assert n >= 6
    L = (1 << n) - 1
    out = []
    for i in range(L):
        y = i + 1
        t1 = ((y & 1) and (y & 2) and (y & 4))
        t2 = ((y & 8) and (y & 16) and (y & 32))
        odd = (1 if t1 else 0) ^ (1 if t2 else 0)
        out.append(1 if odd else 0)
    return out

def gen_random_sparse(n: int, density: float, rng: random.Random) -> List[int]:
    """Oddness drawn i.i.d. Bernoulli(density); even padding 0."""
    L = (1 << n) - 1
    w = [1 if rng.random() < density else 0 for _ in range(L)]
    return [1 if b else 0 for b in w]

def gen_random_z8(n: int, density_odd: float, rng: random.Random) -> List[int]:
    """
    Z8 vector where:
      - LSB is Bernoulli(density_odd) for the odd part,
      - even bits are chosen uniformly from {0,2,4,6}.
    """
    L = (1 << n) - 1
    vec = []
    for _ in range(L):
        odd = 1 if rng.random() < density_odd else 0
        even = 2 * rng.choice([0, 1, 2, 3])  # 0,2,4,6
        vec.append((even | odd) % 8)
    return vec

# Benchmark config & result containers

@dataclass
class BenchConfig:
    """
    Parameters controlling a benchmark run.

    n          : number of qubits
    trials     : number of random instances to generate
    gen        : which generator to use ('near1','xor3','rand_sparse','rand_z8')
    flips      : used by 'near1'
    density    : used by 'rand_sparse' / 'rand_z8'
    strategies : list of decoder / pipeline strategy names to evaluate
    mode       : 'decode' (decode layer only) or 'pipeline' (full optimizer)
    effort     : maps to a default list_size via _beam_from_effort, unless overridden
    list_size  : optional explicit list_size
    chase_t    : max Chase subset size (for dumer-list-chase)
    chase_limit: global cap on Chase patterns
    rpa_iters  : number of RPA iterations
    snap_t     : SNAP subset size
    snap_pool  : SNAP pool size
    seed       : RNG seed for reproducibility
    """
    n: int = 6
    trials: int = 20
    gen: str = "near1"              # near1 | xor3 | rand_sparse | rand_z8
    flips: int = 10                 # for near1
    density: float = 0.25           # for rand_sparse/rand_z8
    strategies: List[str] = None    # ['dumer','dumer-list',...]
    mode: str = "decode"            # decode | pipeline
    effort: int = 3                 # 2^effort -> beam size if list_size not set
    list_size: Optional[int] = None
    chase_t: int = 2
    chase_limit: int = 16
    rpa_iters: int = 2
    snap_t: int = 2
    snap_pool: int = 16
    seed: int = 123

    def __post_init__(self):
        if self.strategies is None:
            self.strategies = ["dumer", "dumer-list", "dumer-list-chase", "rpa-seed-beam", "rpa-adv"]
        assert self.mode in ("decode", "pipeline"), "mode must be 'decode' or 'pipeline'"

@dataclass
class TrialResult:
    """
    Result for a single (strategy, instance) pair.
    """
    strategy: str
    dist: int         # distance after decode, or after-T in pipeline mode
    before_t: int     # T-count before optimization
    after_t: int      # T-count after optimization
    time_ms: float    # runtime in ms

@dataclass
class Aggregate:
    """
    Aggregate statistics over multiple TrialResult objects for the same strategy.
    """
    strategy: str
    trials: int
    mean_dist: float
    median_dist: float
    stdev_dist: float
    min_dist: int
    max_dist: int
    mean_after_t: float
    mean_time_ms: float

# Helper mapping from effort -> list_size

def _beam_from_effort(effort: int | None) -> int:
    """
    Map an "effort level" (1..5) to a beam width (list_size).
    """
    if effort is None: return 8
    e = max(1, min(int(effort), 5))
    return 1 << e   # 2,4,8,16,32

# Core benchmark helpers

def _gen_instance(cfg: BenchConfig, rng: random.Random) -> List[int]:
    """
    Dispatch to the appropriate instance generator according to cfg.gen.
    """
    if cfg.gen == "near1":
        return gen_near_constant_one(cfg.n, cfg.flips, rng, randomize_positions=True)
    if cfg.gen == "xor3":
        return gen_xor_two_triples(cfg.n, rng)
    if cfg.gen == "rand_sparse":
        return gen_random_sparse(cfg.n, cfg.density, rng)
    if cfg.gen == "rand_z8":
        return gen_random_z8(cfg.n, cfg.density, rng)
    raise ValueError(f"unknown generator: {cfg.gen}")

def _decode_once(vec: List[int], n: int, r: int, strategy: str, cfg: BenchConfig) -> Tuple[int, float]:
    """
    Run the decode layer only and return (distance, elapsed_ms).

    The input vec is a Z8 vector; its odd part (LSB) is interpreted as
    the punctured word for RM decoding.
    """
    from .decoders import decode_rm
    w_bits = [1 if (v & 1) else 0 for v in vec]

    # Map strategy name to decode_rm() invocation.
    kwargs: Dict[str, Any] = {}
    if strategy == "dumer":
        call = lambda: decode_rm(w_bits, n, r, "dumer")
    elif strategy == "dumer-list":
        ls = cfg.list_size or _beam_from_effort(cfg.effort)
        call = lambda: decode_rm(w_bits, n, r, "dumer-list", list_size=ls)
    elif strategy == "dumer-list-chase":
        ls = cfg.list_size or _beam_from_effort(cfg.effort)
        call = lambda: decode_rm(w_bits, n, r, "dumer-list-chase", list_size=ls, chase_t=cfg.chase_t, chase_limit=cfg.chase_limit)
    elif strategy == "rpa-seed-beam":
        ls = cfg.list_size or _beam_from_effort(cfg.effort)
        call = lambda: decode_rm(w_bits, n, r, "rpa-seed-beam", list_size=ls, rpa_iters=cfg.rpa_iters)
    elif strategy == "rpa-adv":
        ls = cfg.list_size or _beam_from_effort(cfg.effort)
        call = lambda: decode_rm(w_bits, n, r, "rpa-adv", list_size=ls, rpa_iters=cfg.rpa_iters, snap_t=cfg.snap_t, snap_pool=cfg.snap_pool)
    elif strategy == "ml-exact":
        call = lambda: decode_rm(w_bits, n, r, "ml-exact")
    elif strategy == "rpa":
        ls = cfg.list_size or _beam_from_effort(cfg.effort)
        call = lambda: decode_rm(w_bits, n, r, "rpa", list_size=ls, rpa_iters=cfg.rpa_iters,
                                 snap_t=cfg.snap_t, snap_pool=cfg.snap_pool)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    t0 = time.perf_counter()
    _, _, dist = call()
    t1 = time.perf_counter()
    return dist, (t1 - t0) * 1e3

def _pipeline_once(vec: List[int], n: int, strategy: str, cfg: BenchConfig) -> Tuple[int, int, float]:
    """
    Run the full optimization pipeline once and return:
        (before_t, after_t, elapsed_ms).
    """
    from .. import Optimizer
    if strategy == "dumer":
        opt = Optimizer(decoder="dumer")
    elif strategy == "dumer-list":
        opt = Optimizer(decoder="dumer-list", effort=cfg.effort, list_size=cfg.list_size or None)
    elif strategy == "dumer-list-chase":
        opt = Optimizer(decoder="dumer-list-chase", effort=cfg.effort, list_size=cfg.list_size or None,
                        chase_t=cfg.chase_t, chase_limit=cfg.chase_limit)
    elif strategy == "rpa-seed-beam":
        # guarded inside decoders
        opt = Optimizer(decoder="rpa-seed-beam", effort=cfg.effort, list_size=cfg.list_size or None,
                        rpa_iters=cfg.rpa_iters)
    elif strategy == "rpa-adv":
        opt = Optimizer(decoder="rpa-adv", effort=cfg.effort, list_size=cfg.list_size or None,
                        rpa_iters=cfg.rpa_iters, snap_t=cfg.snap_t, snap_pool=cfg.snap_pool)
    elif strategy == "auto":
        opt = Optimizer(decoder="auto", effort=cfg.effort, rpa_iters=cfg.rpa_iters)
    elif strategy == "ml-exact":
        opt = Optimizer(decoder="ml-exact")
    elif strategy == "rpa":
        opt = Optimizer(decoder="rpa", effort=cfg.effort, list_size=cfg.list_size or None,
                        rpa_iters=cfg.rpa_iters, snap_t=cfg.snap_t, snap_pool=cfg.snap_pool)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    circ = synthesize_from_coeffs(vec, n)
    t0 = time.perf_counter()
    _, rep = opt.optimize(circ)
    t1 = time.perf_counter()
    return rep.before_t, rep.after_t, (t1 - t0) * 1e3

def run_benchmark(cfg: BenchConfig) -> Tuple[List[TrialResult], List[Aggregate]]:
    """
    Run the benchmark and return (per_trial_results, aggregates).

    - In 'decode' mode, we only time the decoder layer (decode_rm).
      dist = decoder distance, before_t = initial odd count, after_t = dist.

    - In 'pipeline' mode, we run the full optimizer and record T-count
      before and after optimization.
    """
    if _load_rmcore() is None and cfg.mode != "ml-exact":
        # The decode layer will fall back to ml-exact if extension missing,
        # but that's too slow for n>=6. Be explicit to the caller.
        raise RuntimeError("rmcore extension not found; build the C++ extension first.")

    rng = random.Random(cfg.seed)
    results: List[TrialResult] = []

    for t in range(cfg.trials):
        vec = _gen_instance(cfg, rng)
        n = cfg.n
        r = n - 4

        for strat in cfg.strategies:
            if cfg.mode == "decode":
                dist, ms = _decode_once(vec, n, r, strat, cfg)
                before_t = sum(1 for v in vec if (v & 1))
                after_t = dist
                results.append(TrialResult(strat, dist, before_t, after_t, ms))
            else:
                before_t, after_t, ms = _pipeline_once(vec, n, strat, cfg)
                results.append(TrialResult(strat, after_t, before_t, after_t, ms))

    # Aggregate per strategy.
    by: Dict[str, List[TrialResult]] = {}
    for tr in results:
        by.setdefault(tr.strategy, []).append(tr)

    aggs: List[Aggregate] = []
    for strat, arr in by.items():
        dists = [x.dist for x in arr]
        afters = [x.after_t for x in arr]
        mean_d = float(stats.mean(dists))
        median_d = float(stats.median(dists))
        stdev_d = float(stats.pstdev(dists)) if len(dists) > 1 else 0.0
        mean_t = float(stats.mean(x.time_ms for x in arr))
        aggs.append(Aggregate(
            strategy=strat,
            trials=len(arr),
            mean_dist=mean_d,
            median_dist=median_d,
            stdev_dist=stdev_d,
            min_dist=min(dists),
            max_dist=max(dists),
            mean_after_t=float(stats.mean(afters)),
            mean_time_ms=mean_t,
        ))
    # Sort aggregates by mean distance then mean time.
    aggs.sort(key=lambda a: (a.mean_dist, a.mean_time_ms))
    return results, aggs
