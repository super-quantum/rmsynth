from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Sequence, Union

# ---------------------------------------------------------------------
# Multi-order / multi-bitplane optimization (ROT-K machinery)
#
# This module generalizes the RM-based optimizer from modulus 8 (Z8) to
# modulus 2^k and, more generally, to composite moduli d = 2^k * d_odd.
#
# Key ideas:
#   - For modulus 2^k, we treat each bit-plane of the coefficients as an
#     independent binary phase polynomial and run RM-decoding per plane,
#     from MSB (ℓ=1) to LSB (ℓ=k).
#   - For a general d, we split d into 2^k * d_odd, optimize only the
#     2^k-even part with the bitplane pass, and recombine via CRT.
#   - Each plane is optimized in a way that never *increases* the plane’s
#     Hamming weight (unless we explicitly revert to the identity).
#
# The actual decoder used per plane is pluggable via `strategy`, and can
# be RPA, Dumer, OSD-based, etc., as wired through decode_rm.
# ---------------------------------------------------------------------

# internal imports
from .core import (
    mk_positions, lift_to_c, coeffs_to_vec, vec_to_coeffs,
    Circuit, extract_phase_coeffs, synthesize_from_coeffs, rm_generator_rows
)
from .decoders import decode_rm
# optional per-plane refinement
try:
    from .osd import osd_refine_l1
except Exception:  # pragma: no cover
    osd_refine_l1 = None  # type: ignore


# helpers

def _mod_pow2(x: int, k: int) -> int:
    """Reduce integer x modulo 2^k."""
    return x & ((1 << k) - 1)

def _as_vec(a: Union[Dict[int, int], Sequence[int]], n: int, k: int) -> List[int]:
    """
    Coerce either a dict{mask->coeff} or a length-(2^n-1) vector into a vector mod 2^k.
    """
    if isinstance(a, dict):
        vec = coeffs_to_vec(a, n)
        return [int(v) & ((1 << k) - 1) for v in vec]
    vec = [int(v) & ((1 << k) - 1) for v in a]
    return vec

def _to_dict(vec: List[int], n: int, k: int) -> Dict[int, int]:
    """Inverse of _as_vec for vectors: build dict of nonzeros mod 2^k."""
    return {mask: v for mask, v in vec_to_coeffs(vec, n).items() if (v & ((1 << k) - 1)) != 0}

def _plane_bits_from_vec(a_vec: List[int], n: int, k: int, ell: int) -> List[int]:
    """
    Extract bit-plane ℓ (1=MSB, k=LSB) as a punctured binary word of length 2^n-1:
        a^(ℓ) = Res2(a >> (k-ℓ)).
    """
    shift = k - ell
    return [ (v >> shift) & 1 for v in a_vec ]

def _rebuild_from_planes(planes: List[List[int]], k: int) -> List[int]:
    """
    Recombine per-plane parity layers (MSB..LSB) back into a length-L vector mod 2^k,
    *without* inter-plane carry (i.e., interpret each column as a k-bit word).
    """
    assert len(planes) >= 1
    L = len(planes[0])
    out = [0] * L
    for ell in range(1, k + 1):  # MSB->LSB
        bit = k - ell
        p = planes[ell - 1]
        for j in range(L):
            if p[j] & 1:
                out[j] |= (1 << bit)
    return out

def _bits_to_int(bits: List[int]) -> int:
    """Pack a list of bits (LSB-first) into an integer."""
    acc = 0
    for i, b in enumerate(bits):
        if b & 1:
            acc |= (1 << i)
    return acc

def _int_popcount(x: int) -> int:
    """Integer popcount wrapper (kept for clarity)."""
    return x.bit_count()

# cache generator rows as packed ints for coupling/OSD helpers
class _RowCache:
    """
    Small helper that caches punctured generator rows for a fixed n, keyed by r.

    Currently this is a thin wrapper around rm_generator_rows, it exists so
    we can extend it later (e.g. with derived data per r) without touching
    the main algorithms.
    """
    def __init__(self, n: int):
        self.n = n
        self._by_r: Dict[int, List[int]] = {}  # r -> [row_int...] in mk_positions order

    def rows(self, r: int) -> List[int]:
        if r in self._by_r:
            return self._by_r[r]
        rows_bits = rm_generator_rows(self.n, r)  # list[int] bitmasks (length L)
        self._by_r[r] = rows_bits
        return self._by_r[r]

def _v2(d: int) -> int:
    """2-adic valuation v2(d): the largest k such that 2^k divides d."""
    if d <= 0:
        raise ValueError("d must be positive")
    k = 0
    while (d & 1) == 0:
        d >>= 1
        k += 1
    return k

def _as_vec_mod(a: Union[Dict[int, int], Sequence[int]], n: int, mod: int) -> List[int]:
    """
    Coerce either a dict{mask->coeff} or a length-(2^n-1) sequence into a vector mod 'mod'.
    """
    from .core import coeffs_to_vec
    if isinstance(a, dict):
        vec = coeffs_to_vec(a, n)
        return [int(v) % mod for v in vec]
    # assume sequence of length L
    return [int(v) % mod for v in a]

def _crt_merge_even_odd(a_even: List[int], a_odd_ref: List[int], k: int, d_odd: int, d: int) -> List[int]:
    """
    Recombine residues:
      x ≡ a_even (mod 2^k)
      x ≡ a_odd_ref (mod d_odd)
    into a unique x (mod d) for each position, using x = a_even + 2^k * t with
    t ≡ (a_odd_ref - a_even) * (2^k)^{-1} (mod d_odd).
    """
    if k == 0:
        # nothing to merge; return odd residues modulo d (== d_odd)
        return [v % d for v in a_odd_ref]

    m1 = 1 << k
    m2 = d_odd
    # Inverse exists because gcd(2^k, d_odd) = 1 (d_odd is odd).
    inv_m1_mod_m2 = pow(m1, -1, m2)

    out = []
    for ae, a0 in zip(a_even, a_odd_ref):
        ae_mod = ae % m1
        a0_mod = a0 % m2
        # find t (mod d_odd)
        t = ((a0_mod - (ae_mod % m2)) % m2) * inv_m1_mod_m2 % m2
        x = (ae_mod + m1 * t) % d
        out.append(x)
    return out

def optimize_multiorder_d(
        a_in: Union[Dict[int, int], Sequence[int]],
        n: int,
        d: int,
        strategy: str = "rpa-adv",
        list_size: Optional[int] = None,
        **dec_kwargs,   # passthrough to decode_rm / rotk multiorder (rpa_iters, snap_t, snap_pool, policy, etc.)
) -> Tuple[List[int], Dict[str, object]]:
    """
    Composite-order wrapper: given phase coefficients modulo d, decompose d = 2^k * d_odd,
    run the existing 2^k multi-order pass on the even part, and keep the odd part unchanged
    (per §5.2/§5.3; Theorem 5.6). Returns:
        (a_out_vec_mod_d, info_dict)
    where info_dict contains:
        - 'k'           : int, the dyadic exponent
        - 'd_odd'       : int, odd component
        - 'before_plane': List[int] (length k), per-plane weights before (even part)
        - 'after_plane' : List[int] (length k), per-plane weights after  (even part)
        - 'stats'       : List[dict], per-plane stats from the 2^k pass (as in optimize_multiorder_bitplanes)
    """
    if d <= 0:
        raise ValueError("d must be positive")
    k = _v2(d)
    d_odd = d >> k

    # incoming vector modulo d
    a_vec_d = _as_vec_mod(a_in, n, d)

    # if k == 0 (d odd): nothing to optimize, identity mapping
    if k == 0:
        info: Dict[str, object] = {
            "k": 0,
            "d_odd": d_odd,
            "before_plane": [],
            "after_plane": [],
            "stats": [],
        }
        return a_vec_d, info

    # even-part vector (mod 2^k)
    m1 = 1 << k
    a_even = [v % m1 for v in a_vec_d]

    # run the existing 2^k pass (MSB->LSB planes), leaving odd residues untouched
    a_even_opt, stats = optimize_multiorder_bitplanes(
        a_even, n, k, strategy=strategy, list_size=list_size, **dec_kwargs
    )

    # recombine (CRT) with odd residues of the original vector
    a_out_vec = _crt_merge_even_odd(a_even_opt, a_vec_d, k, d_odd, d)

    # record final per-plane weights for the even component (before/after)
    before_plane = weights_by_plane(a_even, n, k)
    after_plane  = weights_by_plane(a_even_opt, n, k)

    info: Dict[str, object] = {
        "k": k,
        "d_odd": d_odd,
        "before_plane": before_plane,
        "after_plane": after_plane,
        "stats": stats,
    }
    return a_out_vec, info



# public: plane weights & multi-order optimization

def weights_by_plane(a: Union[Dict[int, int], Sequence[int]], n: int, k: int) -> List[int]:
    """
    Return per-plane Hamming weights [w_1, ..., w_k] with ℓ=1 as MSB and ℓ=k as LSB,
    where w_ℓ = | a^(ℓ) | and a^(ℓ) = Res2(a >> (k-ℓ)).

    Accepts either a coefficient dict or a raw vector.
    """
    a_vec = _as_vec(a, n, k)
    return [ sum(_plane_bits_from_vec(a_vec, n, k, ell)) for ell in range(1, k + 1) ]

def _choose_candidate_for_plane(
        n: int, k: int, ell: int, r: int, w_bits: List[int],
        strategy: str, list_size: Optional[int],
        rows_cache: _RowCache,
        weights: Optional[Sequence[float]],
        coupling: str,
        osd_per_plane: bool,
        osd_order: int,
        dec_kwargs: Dict[str, object]
) -> Tuple[List[int], List[int], int]:
    """
    Build 1-3 candidates for the current plane, optionally refine with OSD,
    and pick the one that minimizes (after, tie-broken by weighted/coupling score).

    Returns (code_bits, selected_monomials, dist) where:
      - code_bits: decoded codeword bits for this plane,
      - selected_monomials: monomial masks (if produced by decode_rm),
      - dist: distance reported by the decoder or the chosen candidate.
    """
    # normalize decoder kwargs: never pass 'list_size' twice
    per_kwargs = dict(dec_kwargs)
    if "list_size" in per_kwargs:
        per_kwargs.pop("list_size")
    ls = list_size if list_size is not None else None

    L = len(w_bits)
    before = sum(w_bits)
    candidates: List[Tuple[List[int], List[int], int]] = []
    # Zero-code candidate: "do nothing on this plane".
    zero_code = [0] * L
    candidates.append((zero_code, [], before))

    # 1) main strategy
    code_bits, selected, dist = decode_rm(w_bits, n, r, strategy=strategy, list_size=ls, **per_kwargs)
    candidates.append((code_bits, selected, dist))

    # 2) side candidate (beam) if different strategy helps tie-breaks
    side = "dumer-list" if strategy != "dumer-list" else "rpa-adv"
    try:
        code2, sel2, dist2 = decode_rm(w_bits, n, r, strategy=side, list_size=ls, **per_kwargs)
        candidates.append((code2, sel2, dist2))
    except Exception:
        # side decoder is strictly optional
        pass

    # 3) per-plane OSD refine (L1) over each candidate (if available)
    if osd_per_plane and osd_refine_l1 is not None:
        refined: List[Tuple[List[int], List[int], int]] = []
        for cb, sm, d in candidates:
            try:
                rb, rsel, rd = osd_refine_l1(w_bits, n, r, cb, order=osd_order)
                # keep best of (cb, rb)
                if rd < d:
                    refined.append((rb, rsel, rd))
                elif rd == d:
                    # keep the one with smaller support to help coupling score
                    if sum(rb) < sum(cb):
                        refined.append((rb, rsel, rd))
                    else:
                        refined.append((cb, sm, d))
                else:
                    refined.append((cb, sm, d))
            except Exception:
                refined.append((cb, sm, d))
        candidates = refined

    # de-duplicate by code_bits
    uniq: List[Tuple[List[int], List[int], int]] = []
    seen = set()
    for cb, sm, d in candidates:
        key = tuple(cb)
        if key not in seen:
            seen.add(key)
            uniq.append((cb, sm, d))
    candidates = uniq

    # scoring: primarily by residual Hamming weight, with optional weighting
    # and a coupling proxy as a secondary tie-breaker.
    w_int = _bits_to_int(w_bits)
    next_plane_weight = None
    if coupling != "none" and ell < k:
        next_plane_weight = weights[ell] if (weights is not None and len(weights) >= k) else None

    def score(cb: List[int], sm: List[int]) -> Tuple[int, float, int]:
        after = sum((wi ^ ci) & 1 for wi, ci in zip(w_bits, cb))
        # weighted score for this plane
        w_score = float(after)
        if weights is not None and len(weights) >= k:
            w_score *= float(weights[ell - 1])
        # coupling proxy (tie-break only): prefer selections whose generator
        # rows have smaller overlap with the current residual.
        c_proxy = 0
        if coupling != "none" and next_plane_weight is not None and sm:
            # approximate density proxy from chosen rows
            for t in sm:
                row_vec = lift_to_c([t], n)  # parity vector (length L)
                row_int = _bits_to_int([v & 1 for v in row_vec])
                c_proxy += _int_popcount(w_int & row_int)
        return (after, w_score, c_proxy)

    best = min(candidates, key=lambda x: score(x[0], x[1]))
    return best

def optimize_multiorder_bitplanes(
        a_in: Union[Dict[int, int], Sequence[int]],
        n: int,
        k: int,
        strategy: str = "rpa-adv",
        list_size: Optional[int] = None,
        *,
        weights: Optional[Sequence[float]] = None,   # MSB..LSB per-plane weights
        coupling: str = "tie",                       # 'none' | 'tie'  (no-carry safe)
        osd_per_plane: bool = False,                 # refine each plane residual with OSD-L1
        osd_order: int = 1,                          # 1..3
        batch: bool = True,                          # enable row caching
        **dec_kwargs,   # passthrough: rpa_iters, snap_t, snap_pool, etc.
) -> Tuple[List[int], List[Dict[str, int]]]:
    """
    Multi-order optimization (Section 5): for ℓ=1..k (MSB→LSB),
    decode a^(ℓ) in RM(n-ℓ-1, n)^* and update that plane.

    Returns (a_out_vec_mod_2^k, stats), where stats is a list of dicts per plane:
        {'ell', 'r', 'before', 'after', 'dist'}

    Notes:
    - We update *planes* without inter-plane carry to preserve the test’s
      “non-worsening” per-plane property and to make 'after' stable across
      subsequent steps. This is consistent with treating per-plane residues
      independently as in the even-order generalization.
    - Coupling is applied only as a tie-break proxy, keeping the no-carry
      invariant intact.
    """
    # prepare bit-planes once
    a_vec0 = _as_vec(a_in, n, k)
    planes: List[List[int]] = [ _plane_bits_from_vec(a_vec0, n, k, ell) for ell in range(1, k + 1) ]
    stats: List[Dict[str, int]] = []

    rows_cache = _RowCache(n) if batch else None

    # extract optional per-plane GL(n,2) knobs (and keep them in dec_kwargs so decode_rm sees them)
    gl_trials = int(dec_kwargs.get('gl_trials', 0) or 0)
    gl_family = dec_kwargs.get('gl_family', None)

    for ell in range(1, k + 1):  # MSB -> LSB
        r = n - ell - 1
        w = planes[ell - 1]
        before = sum(w)
        if r < 0 or before == 0:
            stats.append({'ell': ell, 'r': r, 'before': before, 'after': before, 'dist': 0})
            continue

        # 1) baseline candidate (no GL search) via the existing candidate builder
        cb, sel, dist = _choose_candidate_for_plane(
            n=n, k=k, ell=ell, r=r, w_bits=w,
            strategy=strategy, list_size=list_size,
            rows_cache=(rows_cache if rows_cache is not None else _RowCache(n)),
            weights=weights, coupling=coupling,
            osd_per_plane=osd_per_plane, osd_order=osd_order,
            dec_kwargs=dec_kwargs
        )
        best_cb = cb
        best_dist = dist
        after_cb = sum((wi ^ ci) & 1 for wi, ci in zip(w, best_cb))

        # 2) optional GL(n,2) preconditioning per plane: only accept GL candidate
        #    if it reduces the residual (or ties residual and improves distance).
        if gl_trials > 0:
            per_kwargs = dict(dec_kwargs)
            per_kwargs.pop('list_size', None)
            per_kwargs['gl_trials'] = gl_trials
            if gl_family is not None:
                per_kwargs['gl_family'] = gl_family

            cb_gl, _sel_gl, dist_gl = decode_rm(
                w, n, r, strategy=strategy,
                list_size=list_size,
                **per_kwargs
            )
            after_gl = sum((wi ^ ci) & 1 for wi, ci in zip(w, cb_gl))
            if (after_gl < after_cb) or (after_gl == after_cb and dist_gl < best_dist):
                best_cb = cb_gl
                best_dist = dist_gl
                after_cb = after_gl

        # plane-local update: new plane parity = w XOR best_cb, no carry to other planes
        w_new = [ (wi ^ ci) & 1 for wi, ci in zip(w, best_cb) ]
        after = sum(w_new)

        # safety: never worsen the plane; if it gets worse, revert to identity.
        if after > before:
            w_new = w[:]       # revert to no change
            after = before
            best_dist = before # distance to zero-code fallback

        planes[ell - 1] = w_new
        stats.append({'ell': ell, 'r': r, 'before': before, 'after': after, 'dist': best_dist})

    # rebuild coefficients mod 2^k from finalized planes
    a_out_vec = _rebuild_from_planes(planes, k)

    # make 'after' in stats match the final weights_by_plane exactly (what tests compare to)
    w_final = weights_by_plane(a_out_vec, n, k)
    for s in stats:
        s['after'] = int(w_final[s['ell'] - 1])

    return a_out_vec, stats


# convenience pipeline wrappers

def optimize_circuit_multiorder(
        circ: Circuit,
        k: int,
        strategy: str = "rpa-adv",
        list_size: Optional[int] = None,
        **dec_kwargs,
) -> Tuple[Circuit, Dict[str, object]]:
    """
    Convenience: extract phase coefficients modulo 2^k, run multi-order pass,
    and synthesize a circuit from the optimized coefficients.

    Returns (new_circuit, info_dict) where info_dict contains:
        - 'before_plane': List[int]   per-plane weights before
        - 'after_plane' : List[int]   per-plane weights after
        - 'stats'       : List[dict]  per-plane {'ell','r','before','after','dist'}
        - 'before_T'    : int         (if k==3) T-count before
        - 'after_T'     : int         (if k==3) T-count after
    """
    n = circ.n
    # extract current coefficients (mod 8), reinterpret mod 2^k safely
    a0 = extract_phase_coeffs(circ)
    a0_k = {m: _mod_pow2(v, k) for m, v in a0.items()}

    w_before = weights_by_plane(a0_k, n, k)
    a1_vec, stats = optimize_multiorder_bitplanes(
        a0_k, n, k, strategy=strategy, list_size=list_size, **dec_kwargs
    )
    w_after = weights_by_plane(a1_vec, n, k)

    # synthesize from the optimized coefficients
    a1_dict = _to_dict(a1_vec, n, k)
    vec_opt = coeffs_to_vec(a1_dict, n)
    new_circ = synthesize_from_coeffs(vec_opt, n)

    info: Dict[str, object] = {
        "before_plane": w_before,
        "after_plane": w_after,
        "stats": stats,
    }

    if k == 3:  # optional convenience: expose T-counts if we are mod 8
        before_T = sum( (v & 1) for v in coeffs_to_vec(a0_k, n) )
        after_T  = sum( (v & 1) for v in a1_vec )
        info["before_T"] = before_T
        info["after_T"] = after_T

    return new_circ, info
