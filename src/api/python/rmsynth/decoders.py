from __future__ import annotations
from typing import List, Tuple, Optional
import os, time
import random

from .osd import osd_decode, osd_refine_l1  # OSD (already implemented)
from .rm_cache import get_rows, get_monom_list


# Decoder front-end
#
# This module exposes a single high-level entry point:
#
#     decode_rm(w_bits, n, r, strategy, **kwargs)
#
# where `w_bits` is a *punctured* word of length 2^n - 1 (LSB-first bit list),  and `RM(r,n)^*` is the underlying punctured Reed–Muller code.
#
# Internally it:
#   - wraps the C++ rmcore extension (Dumer / Dumer-list / RPA decoders),
#   - optionally falls back to a Python ML decoder (ml-exact),
#   - adds policy-based tie-breaking (distance vs T-depth),
#   - provides SNAP local refinement, OSD refinements, RPA-2 permutations,
#   - and an optional GL(n,2) search wrapper.
#
# All decoders return:
#   (code_bits, selected_monomials, distance)
# where:
#   - code_bits is the punctured RM codeword (length 2^n-1),
#   - selected_monomials is a list of monomial masks (degree ≤ r),
#   - distance is the Hamming distance to w_bits.


# bits helpers (little-endian)

def pack_bits_le(bit_list: List[int]) -> bytes:
    """Pack a list of bits (LSB-first) into a little-endian byte string."""
    L = len(bit_list)
    out = bytearray((L + 7) // 8)
    for i, b in enumerate(bit_list):
        if b & 1:
            out[i >> 3] |= (1 << (i & 7))
    return bytes(out)


def unpack_bits_le(b: bytes, L: int) -> List[int]:
    """Unpack a little-endian byte string into a list of L bits (LSB-first)."""
    out = [0] * L
    for i in range(L):
        if b[i >> 3] >> (i & 7) & 1:
            out[i] = 1
    return out


# rmcore loader (robust)

def _load_rmcore():
    """
    Best-effort loader for the native rmcore extension.

    Search order:
      1) in-package:   rmsynth.rmcore
      2) top-level:    rmcore
      3) filesystem scan of sys.path for rmcore*.so / .pyd under
         either 'rmsynth/' or directly in a path entry.
    """
    # in-package (use relative import to avoid cycle through __init__)
    try:
        from . import rmcore as _rmcore
        return _rmcore
    except Exception:
        pass
    # top-level
    try:
        import rmcore as _rmcore
        return _rmcore
    except Exception:
        pass
    # scan sys.path for a binary named rmcore*
    try:
        import sys, glob, importlib.util, importlib.machinery, os
        suffixes = importlib.machinery.EXTENSION_SUFFIXES
        for base in list(sys.path):
            # rmsynth/rmcore*.so
            p = os.path.join(base, "rmsynth")
            if os.path.isdir(p):
                for suf in suffixes:
                    import glob as _glob
                    for path in _glob.glob(os.path.join(p, "rmcore" + suf)):
                        spec = importlib.util.spec_from_file_location("rmcore", path)
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            return mod
            # flat rmcore*.so
            for suf in suffixes:
                import glob as _glob
                for path in _glob.glob(os.path.join(base, "rmcore" + suf)):
                    spec = importlib.util.spec_from_file_location("rmcore", path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        return mod
    except Exception:
        pass
    return None


def best_available_decoder() -> str:
    """
    Return the default decoder strategy to use in this environment.

    If the rmcore extension is available, return "dumer" (fast native code),
    otherwise fall back to the Python ML decoder "ml-exact".
    """
    return "dumer" if _load_rmcore() is not None else "ml-exact"


# small helpers

def _bits_to_int(bits: List[int]) -> int:
    """Pack a list of bits (LSB-first) into an integer."""
    acc = 0
    for i, b in enumerate(bits):
        if b & 1:
            acc |= (1 << i)
    return acc


def _int_to_bits(x: int, L: int) -> List[int]:
    """Unpack an integer into L LSB-first bits."""
    return [(x >> i) & 1 for i in range(L)]


# depth helpers (prefer scheduler if available)

def _depth_from_solution(
        w_bits: List[int],
        n: int,
        code_bits: List[int],
        monoms: Optional[List[int]],
) -> int:
    """
    Estimate T-depth for tie-breaking inside decoding.

    We always start from the *punctured residual* (w XOR code_bits), so all
    methods are upper bounds on the true T-depth after synthesis.

    Modes (controlled by environment):

      RM_DEPTH_MODE = "est"   -> estimator from punctured odd set
                      "sched" -> attempt exact scheduler (bounded)

      RM_SCHED_BUDGET = integer >= 0 (node/time budget interpreted by rmcore)

    Notes:
      * This is invoked frequently inside the decoder; keep it fast.
      * For final circuit scheduling, call the scheduler *once* outside.
    """
    rmcore = _load_rmcore()

    # residual odd set (post-correction parity)
    odd_bits = [(wb ^ cb) & 1 for wb, cb in zip(w_bits, code_bits)]

    if rmcore is None:
        # Fallback: T-depth upper bound = Hamming weight of residual
        return sum(odd_bits)

    mode = os.environ.get("RM_DEPTH_MODE", "est").strip().lower()

    # fast estimator path (default)
    if mode != "sched":
        return rmcore.tdepth_from_punctured(pack_bits_le(odd_bits), n)

    # optional exact scheduler (bounded), used only if explicitly requested
    try:
        if monoms is not None and hasattr(rmcore, "tdepth_schedule_from_monoms"):
            budget_env = os.environ.get("RM_SCHED_BUDGET", "")
            budget = int(budget_env) if budget_env.strip() else 20000
            depth, _layers = rmcore.tdepth_schedule_from_monoms(monoms, n, budget)
            return int(depth)
    except Exception:
        pass

    # if scheduling failed, fall back to estimator
    return rmcore.tdepth_from_punctured(pack_bits_le(odd_bits), n)


# GF(2) invert (K x K) for monomial recovery

def _gf2_inv(rows_kxk: List[int], K: int) -> List[int]:
    """
    Gauss-Jordan over GF(2) on KxK matrix given as list of K row-int
    bitmasks. Returns inverse as list of K row-int bitmasks (rows of A^{-1}).

    Here each row-int encodes a row of length K, with LSB = column 0.
    """
    AugL = rows_kxk[:]
    AugR = [1 << i for i in range(K)]  # identity as row-int
    for col in range(K):
        pivot = -1
        for row in range(col, K):
            if (AugL[row] >> col) & 1:
                pivot = row
                break
        if pivot == -1:
            raise RuntimeError("Singular matrix in inversion.")
        if pivot != col:
            AugL[col], AugL[pivot] = AugL[pivot], AugL[col]
            AugR[col], AugR[pivot] = AugR[pivot], AugR[col]
        # eliminate all other rows on this column
        for row in range(K):
            if row == col:
                continue
            if (AugL[row] >> col) & 1:
                AugL[row] ^= AugL[col]
                AugR[row] ^= AugR[col]
    return AugR  # rows of A^{-1}


def _choose_info_columns(rows: List[int], K: int, L: int) -> Optional[List[int]]:
    """
    Choose K pivot columns (an information set) from a K×L generator matrix.

    The generator is given as list of K row-bitmasks `rows` of length L.
    We perform Gauss–Jordan elimination over the columns and return a list
    of K column indices whose K×K submatrix is invertible.

    This is used to reconstruct monomial coefficients from a punctured
    RM codeword; for RM codes this should almost always succeed.
    """
    R = rows[:]  # work on a copy
    chosen: List[int] = []
    r = 0  # current pivot row
    for col in range(L):
        # find a pivot row at/below r with a 1 in this column
        pivot = -1
        for i in range(r, K):
            if (R[i] >> col) & 1:
                pivot = i
                break
        if pivot == -1:
            continue
        # swap pivot row into position r
        if pivot != r:
            R[r], R[pivot] = R[pivot], R[r]
        # eliminate this column from all other rows
        for i in range(K):
            if i != r and ((R[i] >> col) & 1):
                R[i] ^= R[r]
        chosen.append(col)
        r += 1
        if r == K:
            break
    return chosen if len(chosen) == K else None


def _recover_monoms_from_code_bits(code_bits: List[int], n: int, r: int) -> List[int]:
    """
    Given a punctured RM(r,n) codeword (as bits), recover one coefficient
    vector 'a' (degree≤r monomials).

    Steps:
      1. Build the punctured generator matrix rows K×L for RM(r,n).
      2. Pick an information set of K columns (invertible submatrix).
      3. Solve A a = s where s is the restriction of the codeword to
         the chosen columns.

    Returns a list of monomial masks corresponding to 1 entries in a.
    """
    if r < 0:
        return []

    rows = get_rows(n, r)                 # K row-int masks (punctured length L)
    mons = get_monom_list(n, r)           # monomial masks aligned with rows
    K = len(rows)
    L = len(code_bits)
    if K == 0:
        return []

    # choose a robust information set (K pivot columns)
    chosen = _choose_info_columns(rows, K, L)
    if not chosen:
        # extremely unlikely for RM, fall back (empty selection) rather than crash
        return []

    # build the K×K submatrix A from chosen columns (rows[i], chosen[j])
    A_rows: List[int] = []
    for i in range(K):
        row_bits = 0
        for k, j in enumerate(chosen):
            if (rows[i] >> j) & 1:
                row_bits |= (1 << k)
        A_rows.append(row_bits)

    # invert A (guaranteed invertible because of pivot selection)
    Ainv_rows = _gf2_inv(A_rows, K)

    # s = code bits at the chosen information columns
    s_bits = 0
    for k, j in enumerate(chosen):
        if code_bits[j] & 1:
            s_bits |= (1 << k)

    # a = A^{-1} s  (in GF(2))
    a_bits = 0
    for j in range(K):
        if (Ainv_rows[j] & s_bits).bit_count() & 1:
            a_bits |= (1 << j)

    # materialize selected monomials (degree ≤ r)
    selected = [mons[i] for i in range(K) if ((a_bits >> i) & 1)]
    return selected


# GF(2) matrix & GL(n,2) helpers

def _gf2_mat_inv_rows(rows: List[int], n: int) -> Optional[List[int]]:
    """
    Invert an n×n GF(2) matrix given as list of n row-bitmasks (LSB=col0).

    Returns rows of A^{-1} (same format) or None if singular.
    """
    L = rows[:]                         # left: A
    R = [1 << i for i in range(n)]      # right: I
    for col in range(n):
        pivot = -1
        for r in range(col, n):
            if (L[r] >> col) & 1:
                pivot = r
                break
        if pivot == -1:
            return None
        if pivot != col:
            L[col], L[pivot] = L[pivot], L[col]
            R[col], R[pivot] = R[pivot], R[col]
        # eliminate column in all other rows
        for r in range(n):
            if r == col:
                continue
            if (L[r] >> col) & 1:
                L[r] ^= L[col]
                R[r] ^= R[col]
    return R


def _gf2_matmul_rows(A_rows: List[int], B_rows: List[int], n: int) -> List[int]:
    """
    C = A*B over GF(2), all in row-bitmask form.

    Row i of C is the XOR of rows of B selected by 1-bits in row i of A.
    """
    C = [0] * n
    for i in range(n):
        acc = 0
        x = A_rows[i]
        while x:
            lsb = x & -x
            k = (lsb.bit_length() - 1)
            acc ^= B_rows[k]
            x ^= lsb
        C[i] = acc
    return C


def _apply_linear_rows_to_mask(mask: int, rows: List[int], n: int) -> int:
    """
    Apply v' = A * v to an n-bit column vector v (encoded as bitmask 'mask'),
    where A is given as row-bitmasks 'rows'.
    """
    out = 0
    for i in range(n):
        if (rows[i] & mask).bit_count() & 1:
            out |= (1 << i)
    return out


def _permute_punctured_bits_gl(bits: List[int], n: int, A_rows: List[int]) -> List[int]:
    """
    Column-permute a punctured length-(2^n-1) word by the GL(n,2) map j ↦ A*(j+1).

    Each coordinate j corresponds to a non-zero mask m=j+1. We map m to
    m' = A*m and move bit j to position m'-1.
    """
    L = (1 << n) - 1
    out = [0] * L
    for j in range(L):
        m = j + 1
        mp = _apply_linear_rows_to_mask(m, A_rows, n)
        # mp is nonzero (A invertible), so mp-1 ∈ [0..L-1]
        out[mp - 1] = bits[j]
    return out


def _rand_gl_rows(n: int, family: str = "random") -> Tuple[List[int], List[int]]:
    """
    Sample an invertible A ∈ GL(n,2) and return (A_rows, Ainv_rows).

    Families:
      - "random"     : random dense matrices (reject if singular).
      - "triangular" : L (lower, diag=1) * U (upper, diag=1) random — always invertible.
      - "perm"       : permutation matrices (subset of GL).
    """
    fam = (family or "random").lower()
    if fam == "perm":
        p = list(range(n))
        random.shuffle(p)
        # rows of P: row i has 1 at column p_inv[i]
        p_inv = [0] * n
        for j, pj in enumerate(p):
            p_inv[pj] = j
        A_rows = [1 << p_inv[i] for i in range(n)]
        # inverse rows: row i has 1 at column p[i]
        Ainv_rows = [1 << p[i] for i in range(n)]
        return A_rows, Ainv_rows

    if fam == "triangular":
        # lower with diag=1
        L_rows = []
        for i in range(n):
            row = (1 << i)
            for j in range(i):
                if random.getrandbits(1):
                    row |= (1 << j)
            L_rows.append(row)
        # upper with diag=1
        U_rows = []
        for i in range(n):
            row = (1 << i)
            for j in range(i + 1, n):
                if random.getrandbits(1):
                    row |= (1 << j)
            U_rows.append(row)
        A_rows = _gf2_matmul_rows(L_rows, U_rows, n)
        Ainv_rows = _gf2_mat_inv_rows(A_rows, n)
        assert Ainv_rows is not None  # constructed invertible
        return A_rows, Ainv_rows

    # default: random dense with rejection on singular
    while True:
        A_rows = [random.getrandbits(n) for _ in range(n)]
        inv = _gf2_mat_inv_rows(A_rows, n)
        if inv is not None:
            return A_rows, inv


# GL(n,2) search wrapper

def _decode_rm_with_gl_search(
        w_bits_list: List[int],
        n: int,
        r: int,
        strategy: str,
        # pass-through knobs:
        list_size: Optional[int], chase_t: Optional[int], chase_limit: Optional[int],
        snap: bool, snap_t: Optional[int], snap_pool: Optional[int], rpa_iters: Optional[int],
        osd_top: Optional[int],
        snap_strong: Optional[bool], snap_time_ms: Optional[int], snap_node_limit: Optional[int],
        # policy/λ:
        depth_tradeoff: Optional[int], policy: Optional[str], policy_lambda: Optional[int],
        # GL controls:
        gl_trials: int, gl_family: Optional[str],
) -> Tuple[List[int], List[int], int]:
    """
    Try multiple A ∈ GL(n,2) as preconditioners:

      - permute columns of w by A,
      - run the requested strategy once,
      - unpermute the candidate by A^{-1},
      - recover monomials and score via policy_decide.

    We always include the identity baseline; gl_trials controls the total
    number of A's (including identity).
    """
    # identity baseline
    best_bits, best_sel, best_dist = decode_rm(
        w_bits_list, n, r, strategy,
        list_size=list_size, chase_t=chase_t, chase_limit=chase_limit,
        snap=snap, snap_t=snap_t, snap_pool=snap_pool, rpa_iters=rpa_iters, osd_top=osd_top,
        snap_strong=snap_strong, snap_time_ms=snap_time_ms, snap_node_limit=snap_node_limit,
        depth_tradeoff=depth_tradeoff, policy=policy, policy_lambda=policy_lambda,
        gl_trials=1, gl_family=None,   # IMPORTANT: disable recursion
    )
    best_depth = _depth_from_solution(w_bits_list, n, best_bits, best_sel)

    trials = max(0, int(gl_trials) - 1)
    for _ in range(trials):
        # sample an invertible GL(n,2) element and decode once
        A_rows, Ainv_rows = _rand_gl_rows(n, gl_family or "random")
        w_perm = _permute_punctured_bits_gl(w_bits_list, n, A_rows)

        cand_bits_perm, _sel_unused, _dist_perm = decode_rm(
            w_perm, n, r, strategy,
            list_size=list_size, chase_t=chase_t, chase_limit=chase_limit,
            snap=snap, snap_t=snap_t, snap_pool=snap_pool, rpa_iters=rpa_iters, osd_top=osd_top,
            snap_strong=snap_strong, snap_time_ms=snap_time_ms, snap_node_limit=snap_node_limit,
            depth_tradeoff=depth_tradeoff, policy=policy, policy_lambda=policy_lambda,
            gl_trials=1, gl_family=None,  # IMPORTANT: disable recursion
        )
        # unpermute candidate back by A^{-1}
        cand_bits = _permute_punctured_bits_gl(cand_bits_perm, n, Ainv_rows)
        # recover a valid monomial set for scheduling/tie-breaks
        cand_sel = _recover_monoms_from_code_bits(cand_bits, n, r)
        cand_dist = _hamming(w_bits_list, cand_bits)
        cand_depth = _depth_from_solution(w_bits_list, n, cand_bits, cand_sel)

        if policy_decide(
                best_dist, best_depth,
                cand_dist, cand_depth,
                policy=policy, policy_lambda=policy_lambda, depth_tradeoff=depth_tradeoff
        ):
            best_bits, best_sel, best_dist = cand_bits, cand_sel, cand_dist
            best_depth = cand_depth

    return best_bits, best_sel, best_dist


# trade-off / policy helpers

def _lambda_from_policy(
        policy: Optional[str],
        policy_lambda: Optional[int],
        fallback: Optional[int],
) -> Optional[int]:
    """
    Map textual policy to lambda (depth_tradeoff).

      - None -> fallback (usually None)
      - "distance+depth" (aliases: "dist+depth","linear","lin")
           -> lambda = policy_lambda (must be int)
      - "depth-only" or "min-depth"
           -> lambda = 0
      - "min-distance" or "distance"
           -> None (pure distance-first, legacy behaviour)
    """
    if policy is None:
        return fallback
    key = policy.strip().lower()
    if key in ("distance+depth", "dist+depth", "linear", "lin"):
        return int(policy_lambda) if policy_lambda is not None else fallback
    if key in ("depth-only", "min-depth"):
        return 0
    if key in ("min-distance", "distance"):
        return None
    # unknown -> fallback
    return fallback


def _distance_depth_cost(
        w_bits: List[int],
        n: int,
        code_bits: List[int],
        monoms: Optional[List[int]],
        k: Optional[int],
) -> Tuple[int, int, int]:
    """
    Returns (dist, depth, cost) with:

        dist  = Hamming distance to w_bits
        depth = T-depth estimate (or exact if scheduler used)
        cost  = dist if k is None, else k*dist + depth

    If `monoms` is None we compute depth via the residual estimator only.
    """
    dist = sum((wb ^ cb) & 1 for wb, cb in zip(w_bits, code_bits))
    if k is None:
        depth = _depth_from_solution(w_bits, n, code_bits, monoms=None)  # only used for tie-break elsewhere
        return dist, depth, dist
    depth = _depth_from_solution(w_bits, n, code_bits, monoms)
    return dist, depth, (k * dist + depth)


def _choose_by_tradeoff(
        cands: List[Tuple[List[int], Optional[List[int]], int]],
        w_bits: List[int],
        n: int,
        k: Optional[int],
):
    """
    Choose the best candidate from:

        cands: list of (code_bits, monoms_or_None, dist) tuples.

    Selection rule:
      - if k is None: minimise (dist, depth)
      - else: minimise (k*dist + depth, dist, depth)
    """
    best = None
    best_payload = None
    for code_bits, monoms, dist in cands:
        d, dep, cost = _distance_depth_cost(w_bits, n, code_bits, monoms, k)
        score = (cost, d, dep) if k is not None else (d, dep)
        if best is None or score < best:
            best = score
            best_payload = (code_bits, monoms, dist if dist is not None else d)
    return best_payload  # type: ignore[return-value]


# policy helper (exported for testing)

def policy_decide(
        best_dist: int,
        best_depth: int,
        cand_dist: int,
        cand_depth: int,
        policy: str | None = None,
        policy_lambda: float | None = None,
        depth_tradeoff: int | None = None,
) -> bool:
    """
    Decide whether the candidate (cand_dist, cand_depth) should be accepted
    over (best_dist, best_depth) under the configured policy.

    When 'depth_tradeoff' (k) is set (guard semantics):

      - If distance gets worse by Δd>0, accept only if depth improves by
        STRICTLY more than k*Δd.
      - If depth gets worse by Δh>0, accept only if distance improves by
        STRICTLY more than k*Δh.
      - If Δd==0, accept only on strict depth improvement.

    Otherwise:

      - If policy == "distance+depth": use linear cost
            dist + λ * depth
      - Else (default): lexicographic (dist, depth).
    """
    dd = cand_dist - best_dist
    dh = cand_depth - best_depth

    if depth_tradeoff is not None:
        k = int(depth_tradeoff)

        if dd > 0:
            # distance worse: require strictly more than k*dd depth gain.
            return (-dh) > k * dd

        elif dd < 0:
            # distance better:
            if dh <= 0:
                return True  # better-or-equal depth with better distance
            # depth worse: require strictly more than k*dh distance gain.
            return (-dd) > k * dh

        else:
            # same distance: only accept on strict depth improvement.
            return dh < 0

    # no guard -> linear or lexicographic
    if (policy or "").lower() == "distance+depth":
        lam = float(policy_lambda or 0.0)
        return (cand_dist + lam * cand_depth) < (best_dist + lam * best_depth)

    # default: lexicographic
    if cand_dist < best_dist:
        return True
    if cand_dist > best_dist:
        return False
    return cand_depth < best_depth


# SNAP (local exact) - current light version (size-1 / size-2 subsets)

def snap_local_exact_bits(
        w_bits: List[int],
        n: int,
        r: int,
        code_bits: List[int],
        monoms: List[int],
        snap_t: int = 2,
        pool_max: int = 16,
        comb_limit: int = 200,
) -> Tuple[List[int], List[int], int]:
    """
    Lightweight local exact search around a given candidate codeword.

    Procedure:
      - Build a pool of up to 'pool_max' generator rows with the largest
        overlap with the residual (w XOR code).
      - Try all size-1 and (optionally) size-2 subsets of the pool
        (bounded by 'comb_limit') as toggles.
      - Return the best candidate found, or the input candidate if no
        improvement is found.

    Returns (code_bits, monoms, dist).
    """
    L = (1 << n) - 1
    w = _bits_to_int(w_bits)
    cw = _bits_to_int(code_bits)
    res = w ^ cw
    if res == 0:
        return code_bits, monoms, 0

    rows = get_rows(n, r) if r >= 0 else []
    monom_list = get_monom_list(n, r)

    # score rows by overlap with residual
    scored = []
    for i, row in enumerate(rows):
        score = (res & row).bit_count()
        if score > 0:
            scored.append((score, i))
    scored.sort(reverse=True)
    pool_idx = [i for _, i in scored[:pool_max]]
    pool_rows = [rows[i] for i in pool_idx]
    pool_monoms = [monom_list[i] for i in pool_idx]

    best_cw = cw
    best_dist = (w ^ cw).bit_count()
    best_subset: List[int] = []

    # size-1 toggles
    tried = 0
    for i in range(len(pool_idx)):
        cand = cw ^ pool_rows[i]
        d = (w ^ cand).bit_count()
        if d < best_dist:
            best_dist = d
            best_cw = cand
            best_subset = [i]
        tried += 1
        if tried >= comb_limit:
            break

    # size-2 toggles
    if snap_t >= 2 and tried < comb_limit:
        for i in range(len(pool_idx)):
            for j in range(i + 1, len(pool_idx)):
                cand = cw ^ pool_rows[i] ^ pool_rows[j]
                d = (w ^ cand).bit_count()
                if d < best_dist:
                    best_dist = d
                    best_cw = cand
                    best_subset = [i, j]
                tried += 1
                if tried >= comb_limit:
                    break
            if tried >= comb_limit:
                break

    if best_dist < (w ^ cw).bit_count():
        # apply best subset to monomial set
        toggles = [pool_monoms[k] for k in best_subset]
        mono_set = set(monoms)
        for t in toggles:
            if t in mono_set:
                mono_set.remove(t)
            else:
                mono_set.add(t)
        monoms2 = sorted(mono_set)
        code_bits2 = _int_to_bits(best_cw, L)
        return code_bits2, monoms2, best_dist
    else:
        return code_bits, monoms, best_dist


# STRONG SNAP (branch-and-bound over a larger pool, time-capped)

def snap_branch_and_bound_bits(
        w_bits: List[int],
        n: int,
        r: int,
        code_bits: List[int],
        monoms: List[int],
        pool_max: int = 24,
        time_ms: int = 15,
        node_limit: int = 100_000,
) -> Tuple[List[int], List[int], int]:
    """
    Stronger neighbourhood search using branch-and-bound over a pool
    of generator rows.

    Objective: minimise Hamming(w XOR (code XOR sum_rows S)), where S is
    a subset of the pool.

    The search is time-capped (time_ms) and node-capped (node_limit).
    We seed the incumbent using exact size-1/2 search over the same pool,
    so the result is guaranteed ≤ the simple SNAP result even on timeout.
    """
    L = (1 << n) - 1
    w = _bits_to_int(w_bits)
    cw = _bits_to_int(code_bits)
    res0 = w ^ cw
    base_dist = res0.bit_count()
    if base_dist == 0:
        return code_bits, monoms, 0
    if r < 0:
        return code_bits, monoms, base_dist

    rows_all = get_rows(n, r)
    monoms_all = get_monom_list(n, r)

    # build pool by overlap with residual
    scored = []
    for i, row in enumerate(rows_all):
        ov = (res0 & row).bit_count()
        if ov > 0:
            scored.append((ov, i))
    scored.sort(reverse=True)
    pool_idx = [i for _, i in scored[:pool_max]]
    if not pool_idx:
        return code_bits, monoms, base_dist

    pool_rows = [rows_all[i] for i in pool_idx]
    pool_mons = [monoms_all[i] for i in pool_idx]
    pool_wts = [row.bit_count() for row in pool_rows]
    m = len(pool_rows)

    # order pool by "gain" on res0: 2*overlap - wt(row)
    init_gains = []
    for k in range(m):
        ov = (res0 & pool_rows[k]).bit_count()
        init_gains.append((2 * ov - pool_wts[k], k))
    init_gains.sort(reverse=True)
    order = [k for _, k in init_gains]
    pool_rows = [pool_rows[k] for k in order]
    pool_mons = [pool_mons[k] for k in order]
    pool_wts = [pool_wts[k] for k in order]

    # seed incumbent (size-1/2)
    best_cw_int = cw
    best_res = res0
    best_dist = base_dist
    best_mask = 0

    # size-1
    for i in range(m):
        cand_cw = cw ^ pool_rows[i]
        d = (w ^ cand_cw).bit_count()
        if d < best_dist:
            best_dist = d
            best_cw_int = cand_cw
            best_res = w ^ cand_cw
            best_mask = (1 << i)

    # size-2
    for i in range(m):
        for j in range(i + 1, m):
            cand_cw = cw ^ pool_rows[i] ^ pool_rows[j]
            d = (w ^ cand_cw).bit_count()
            if d < best_dist:
                best_dist = d
                best_cw_int = cand_cw
                best_res = w ^ cand_cw
                best_mask = (1 << i) | (1 << j)

    if best_dist == 0:
        # early exit: we already found a perfect codeword in the pool
        toggles = []
        x = best_mask
        t_rows_xor = 0
        while x:
            lsb = x & -x
            j = (lsb.bit_length() - 1)
            toggles.append(pool_mons[j])
            t_rows_xor ^= pool_rows[j]
            x ^= lsb
        mono_set = set(monoms)
        for t in toggles:
            if t in mono_set:
                mono_set.remove(t)
            else:
                mono_set.add(t)
        monoms2 = sorted(mono_set)
        cw2 = cw ^ t_rows_xor
        code_bits2 = _int_to_bits(cw2, L)
        return code_bits2, monoms2, best_dist

    # branch & bound on the pool (depth-first)
    t_deadline = time.perf_counter() + max(0.0, time_ms) / 1000.0
    nodes = 0
    stop = False

    def lower_bound(curRes: int, idx: int) -> int:
        """
        Simple lower bound on achievable residual weight from curRes
        if we only consider rows[idx:].
        """
        cur = curRes.bit_count()
        sum_pos = 0
        for j in range(idx, m):
            ov = (curRes & pool_rows[j]).bit_count()
            g = 2 * ov - pool_wts[j]
            if g > 0:
                sum_pos += g
        lb = cur - sum_pos
        return 0 if lb < 0 else lb

    def dfs(idx: int, curRes: int, chosen_mask: int):
        nonlocal best_res, best_dist, best_mask, best_cw_int, nodes, stop
        if stop:
            return
        if time.perf_counter() > t_deadline:
            stop = True
            return
        nodes += 1
        if nodes >= node_limit:
            stop = True
            return

        cur_wt = curRes.bit_count()
        if cur_wt < best_dist:
            best_dist = cur_wt
            best_res = curRes
            best_mask = chosen_mask
            # materialize best cw
            cw_from_mask = cw
            x = chosen_mask
            while x:
                lsb = x & -x
                j = (lsb.bit_length() - 1)
                cw_from_mask ^= pool_rows[j]
                x ^= lsb
            best_cw_int = cw_from_mask
            if best_dist == 0:
                stop = True
                return

        if idx >= m:
            return

        if lower_bound(curRes, idx) >= best_dist:
            return

        row = pool_rows[idx]
        g = 2 * (curRes & row).bit_count() - pool_wts[idx]

        # explore promising branch first
        if g > 0:
            dfs(idx + 1, curRes ^ row, chosen_mask | (1 << idx))
            if stop:
                return
            dfs(idx + 1, curRes, chosen_mask)
        else:
            dfs(idx + 1, curRes, chosen_mask)
            if stop:
                return
            dfs(idx + 1, curRes ^ row, chosen_mask | (1 << idx))

    dfs(0, res0, 0)

    # build output from best_mask
    toggles = []
    x = best_mask
    t_rows_xor = 0
    while x:
        lsb = x & -x
        j = (lsb.bit_length() - 1)
        toggles.append(pool_mons[j])
        t_rows_xor ^= pool_rows[j]
        x ^= lsb

    if best_dist < base_dist:
        mono_set = set(monoms)
        for t in toggles:
            if t in mono_set:
                mono_set.remove(t)
            else:
                mono_set.add(t)
        monoms2 = sorted(mono_set)
        cw2 = cw ^ t_rows_xor
        code_bits2 = _int_to_bits(cw2, L)
        return code_bits2, monoms2, best_dist
    else:
        return code_bits, monoms, base_dist


# RPA-2 helpers (permutations)

def _perm_inv(perm: List[int]) -> List[int]:
    """Return inverse permutation of perm (list of indices)."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _apply_perm_to_mask(mask: int, n: int, perm: List[int]) -> int:
    """
    Apply a qubit permutation to a monomial mask.

    If bit b is set in `mask`, after permutation it moves to position perm[b].
    """
    out = 0
    for b in range(n):
        if (mask >> b) & 1:
            out |= (1 << perm[b])
    return out


def _permute_punctured_bits(bits: List[int], n: int, perm: List[int]) -> List[int]:
    """
    Column-permute a punctured word using a qubit permutation `perm`.
    """
    L = (1 << n) - 1
    out = [0] * L
    for j in range(L):
        m = j + 1
        mp = _apply_perm_to_mask(m, n, perm)
        out[mp - 1] = bits[j]
    return out


def _unpermute_punctured_bits(bits: List[int], n: int, perm: List[int]) -> List[int]:
    """Inverse of _permute_punctured_bits."""
    inv = _perm_inv(perm)
    return _permute_punctured_bits(bits, n, inv)


def _hamming(a: List[int], b: List[int]) -> int:
    """Hamming distance between two bit lists."""
    return sum((x ^ y) & 1 for x, y in zip(a, b))


def _gen_rpa2_perms(n: int, max_perms: Optional[int]) -> List[List[int]]:
    """
    Generate a small, structured set of qubit permutations for RPA-2.

    Includes identity, single swaps with qubit 0, then double swaps
    (0 with i, 1 with j). This covers a useful subset of the symmetry
    group without blowing up.
    """
    seen = set()
    out: List[List[int]] = []

    def add(p):
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            out.append(p[:])

    add(list(range(n)))  # identity
    if max_perms is not None and len(out) >= max_perms:
        return out
    for i in range(1, n):
        p = list(range(n))
        p[0], p[i] = p[i], p[0]
        add(p)
        if max_perms is not None and len(out) >= max_perms:
            return out
    for i in range(1, n):
        for j in range(i + 1, n):
            p = list(range(n))
            p[0], p[i] = p[i], p[0]
            p[1], p[j] = p[j], p[1]
            add(p)
            if max_perms is not None and len(out) >= max_perms:
                return out
    return out


# main dispatcher

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
        osd_top: int | None = None,            # beam+OSD
        # strong snap knobs:
        snap_strong: bool | None = None,
        snap_time_ms: int | None = None,
        snap_node_limit: int | None = None,
        # linear trade-off:
        depth_tradeoff: int | None = None,
        # ergonomic alias:
        policy: Optional[str] = None,
        policy_lambda: Optional[int] = None,
        # GL(n,2) search controls (optional)
        gl_trials: Optional[int] = None,
        gl_family: Optional[str] = None,
):
    """
    High-level Reed–Muller decoder dispatcher.

    Parameters

    w_bits_list : List[int]
        Punctured observation word of length 2^n - 1 (LSB-first).
    n, r : int
        RM(r,n) code parameters (punctured).
    strategy : str
        Decoder flavour. Supported values include:

          - "ml-exact"           : brute-force ML (Python, slow).
          - "dumer"              : Dumer hard-decision decoder (rmcore).
          - "dumer-list"         : list/beam version.
          - "dumer-list-chase"   : beam + chase reliability flips.
          - "rpa-seed-beam"      : RPA-1 seeding + beam (guarded).
          - "rpa-adv"            : RPA-1 + SNAP + OSD (advanced).
          - "rpa2-seed-beam"     : multi-axis RPA-1.
          - "rpa2"               : RPA-2 + SNAP + OSD.
          - "osd1/2/3"           : OSD of order 1/2/3 around a baseline.
          - "beam-osd1/2/3"      : Top-K beam seeds + OSD.
          - "rpa"                : convenience alias for "rpa-adv".

    Returns

    code_bits : List[int]
        Punctured RM codeword.
    selected_monomials : List[int]
        Monomials (masks) describing the codeword.
    dist : int
        Hamming distance between w_bits_list and code_bits.
    """
    L = len(w_bits_list)
    # map policy -> lambda if provided
    depth_tradeoff = _lambda_from_policy(policy, policy_lambda, depth_tradeoff)

    # full GL(n,2) preconditioning search wrapper (opt-in)
    if gl_trials is not None and int(gl_trials) > 1:
        return _decode_rm_with_gl_search(
            w_bits_list, n, r, strategy,
            list_size, chase_t, chase_limit,
            snap, snap_t, snap_pool, rpa_iters, osd_top,
            snap_strong, snap_time_ms, snap_node_limit,
            depth_tradeoff, policy, policy_lambda,
            int(gl_trials), gl_family,
        )

    # central decision helper for all comparisons (uses exported policy_decide)
    def _decide(
            cur_bits: List[int],
            cur_sel: List[int],
            cur_dist: int,
            cand_bits: List[int],
            cand_sel: List[int],
            cand_dist: int,
    ) -> bool:
        dep_cur = _depth_from_solution(w_bits_list, n, cur_bits, cur_sel)
        dep_cand = _depth_from_solution(w_bits_list, n, cand_bits, cand_sel)
        return policy_decide(
            cur_dist,
            dep_cur,
            cand_dist,
            dep_cand,
            policy=policy,
            policy_lambda=policy_lambda,
            depth_tradeoff=depth_tradeoff,
        )

    if strategy == "ml-exact":
        # pure Python maximum-likelihood decoder (slow, but extension-free)
        from .core import decode_rm_bruteforce
        w_int = _bits_to_int(w_bits_list)
        cw, selected, _ = decode_rm_bruteforce(w_int, n, r)
        code_bits = _int_to_bits(cw, L)
        dist = (cw ^ w_int).bit_count()
        return code_bits, selected, dist

    rmcore = _load_rmcore()
    if rmcore is None:
        # extension missing -> fall back to ML decoder
        return decode_rm(w_bits_list, n, r, "ml-exact")

    w_bytes = pack_bits_le(w_bits_list)

    # simple strategies
    if strategy == "dumer":
        cw_bytes, selected, dist = rmcore.decode_rm_dumer_punctured(w_bytes, n, r)
        return unpack_bits_le(cw_bytes, L), selected, dist

    if strategy == "dumer-list":
        ls = int(list_size or 4)
        if depth_tradeoff is not None and hasattr(rmcore, "decode_rm_dumer_list_topk_punctured"):
            # pull Top-K candidates and choose by trade-off; recover monomials after choosing.
            cw_list = rmcore.decode_rm_dumer_list_topk_punctured(w_bytes, n, r, ls, ls)
            cands: List[Tuple[List[int], Optional[List[int]], int]] = []
            for cwb in cw_list:
                code_bits = unpack_bits_le(cwb, L)
                dist = sum((wb ^ cb) & 1 for wb, cb in zip(w_bits_list, code_bits))
                cands.append((code_bits, None, dist))
            code_bits, _sel_unused, _dist = _choose_by_tradeoff(cands, w_bits_list, n, depth_tradeoff)
            selected = _recover_monoms_from_code_bits(code_bits, n, r)
            dist = sum((wb ^ cb) & 1 for wb, cb in zip(w_bits_list, code_bits))
            return code_bits, selected, dist
        else:
            cw_bytes, selected, dist = rmcore.decode_rm_dumer_list_punctured(w_bytes, n, r, ls)
            return unpack_bits_le(cw_bytes, L), selected, dist

    if strategy == "dumer-list-chase":
        ls = int(list_size or 4)
        ct = int(chase_t or 1)
        cl = int(chase_limit or 16)
        cw_bytes, selected, dist = rmcore.decode_rm_dumer_list_chase_punctured(
            w_bytes, n, r, ls, ct, cl
        )
        return unpack_bits_le(cw_bytes, L), selected, dist

    # guarded RPA-1 (single-axis RPA + plain beam)
    if strategy in ("rpa-seed-beam", "rpa-adv"):
        iters = int(rpa_iters or 2)
        ls = int(list_size or 4)

        # Baseline beam and RPA-1 candidate
        cw_b, sel_b, dist_b = rmcore.decode_rm_dumer_list_punctured(w_bytes, n, r, ls)
        cw_r, sel_r, dist_r = rmcore.decode_rm_rpa1_punctured(w_bytes, n, r, iters, ls)

        code_b = unpack_bits_le(cw_b, L)
        code_r = unpack_bits_le(cw_r, L)

        # choose via unified policy
        if _decide(code_b, sel_b, dist_b, code_r, sel_r, dist_r):
            cw_best, sel_best, dist_best = cw_r, sel_r, dist_r
        else:
            cw_best, sel_best, dist_best = cw_b, sel_b, dist_b

        code_bits = unpack_bits_le(cw_best, L)

        if strategy == "rpa-adv":
            # light SNAP refinement over the selected candidate
            st = int(snap_t or 2)
            sp = int(snap_pool or 16)
            code_bits, sel_best, dist_best = snap_local_exact_bits(
                w_bits_list, n, r, code_bits, sel_best, st, sp
            )
            # strong snap (branch-and-bound) — opt-in via snap_strong / env
            use_strong = bool(snap_strong) if snap_strong is not None else False
            if use_strong and r >= 0:
                tcap = int(os.environ.get("RM_SNAP_BB_MS", str(snap_time_ms if snap_time_ms is not None else 15)))
                ncap = int(os.environ.get("RM_SNAP_BB_NODES", str(snap_node_limit if snap_node_limit is not None else 100_000)))
                sp2 = int(os.environ.get("RM_SNAP_BB_POOL", str(sp)))
                cand_bits, cand_sel, cand_dist = snap_branch_and_bound_bits(
                    w_bits_list, n, r, code_bits, sel_best, pool_max=sp2, time_ms=tcap, node_limit=ncap
                )
                if _decide(code_bits, sel_best, dist_best, cand_bits, cand_sel, cand_dist):
                    code_bits, sel_best, dist_best = cand_bits, cand_sel, cand_dist

            # OSD-L1 refine (policy-guarded)
            cand_bits, cand_sel, cand_dist = osd_refine_l1(w_bits_list, n, r, code_bits)
            if _decide(code_bits, sel_best, dist_best, cand_bits, cand_sel, cand_dist):
                code_bits, sel_best, dist_best = cand_bits, cand_sel, cand_dist

        return code_bits, sel_best, dist_best

    # RPA-2 (permutation-averaged RPA-1)
    if strategy in ("rpa2-seed-beam", "rpa2"):
        iters = int(rpa_iters or 2)
        ls = int(list_size or 4)

        # beam baseline
        cw_b, sel_b, dist_b = rmcore.decode_rm_dumer_list_punctured(w_bytes, n, r, ls)
        code_b = unpack_bits_le(cw_b, L)

        # generate qubit permutations
        max_perms_env = os.environ.get("RM_RPA2_MAX_PERMS", "")
        max_perms = int(max_perms_env) if max_perms_env.strip() else None
        perms = _gen_rpa2_perms(n, max_perms)

        best_bits = None
        best_sel = None
        best_dist = 1 << 30
        best_depth = 1 << 30

        for p in perms:
            # permute the punctured word, run RPA-1, unpermute solution
            w_perm = _permute_punctured_bits(w_bits_list, n, p)
            cw_p, sel_p, _ = rmcore.decode_rm_rpa1_punctured(pack_bits_le(w_perm), n, r, iters, ls)
            code_p = unpack_bits_le(cw_p, L)
            code_un = _unpermute_punctured_bits(code_p, n, p)
            inv = _perm_inv(p)
            sel_un = [_apply_perm_to_mask(m, n, inv) for m in sel_p]
            dist = sum((a ^ b) & 1 for a, b in zip(w_bits_list, code_un))
            depth = _depth_from_solution(w_bits_list, n, code_un, sel_un)
            if (dist < best_dist) or (dist == best_dist and depth < best_depth):
                best_bits, best_sel, best_dist, best_depth = code_un, sel_un, dist, depth

        # guard vs plain beam using policy
        if _decide(code_b, sel_b, dist_b, best_bits, best_sel, best_dist):  # type: ignore[arg-type]
            code_bits, selected, dist = best_bits, best_sel, best_dist  # type: ignore[assignment]
        else:
            code_bits, selected, dist = code_b, sel_b, dist_b

        if strategy == "rpa2":
            # add SNAP + strong SNAP + OSD on top of RPA-2
            st = int(snap_t or 2)
            sp = int(snap_pool or 16)
            code_bits, selected, dist = snap_local_exact_bits(
                w_bits_list, n, r, code_bits, selected, st, sp
            )
            # strong snap
            use_strong = bool(snap_strong) if snap_strong is not None else False
            if use_strong and r >= 0:
                tcap = int(os.environ.get("RM_SNAP_BB_MS", str(snap_time_ms if snap_time_ms is not None else 15)))
                ncap = int(os.environ.get("RM_SNAP_BB_NODES", str(snap_node_limit if snap_node_limit is not None else 100_000)))
                sp2 = int(os.environ.get("RM_SNAP_BB_POOL", str(sp)))
                cand_bits, cand_sel, cand_dist = snap_branch_and_bound_bits(
                    w_bits_list, n, r, code_bits, selected, pool_max=sp2, time_ms=tcap, node_limit=ncap
                )
                if _decide(code_bits, selected, dist, cand_bits, cand_sel, cand_dist):
                    code_bits, selected, dist = cand_bits, cand_sel, cand_dist

            # OSD-L1
            cand_bits, cand_sel, cand_dist = osd_refine_l1(w_bits_list, n, r, code_bits)
            if _decide(code_bits, selected, dist, cand_bits, cand_sel, cand_dist):
                code_bits, selected, dist = cand_bits, cand_sel, cand_dist

        return code_bits, selected, dist

    # OSD family (policy-guarded, around a beam baseline)
    if strategy in ("osd1", "osd2", "osd3"):
        ls = int(list_size or 8)
        cw_b, sel_b, dist_b = rmcore.decode_rm_dumer_list_punctured(w_bytes, n, r, ls)
        code_b = unpack_bits_le(cw_b, L)
        order = 1 if strategy == "osd1" else (2 if strategy == "osd2" else 3)
        cand_bits, cand_sel, cand_dist = osd_decode(w_bits_list, n, r, code_b, order=order)
        return (
            (cand_bits, cand_sel, cand_dist)
            if _decide(code_b, sel_b, dist_b, cand_bits, cand_sel, cand_dist)
            else (code_b, sel_b, dist_b)
        )

    # beam + OSD family (policy-guarded)
    if strategy in ("beam-osd1", "beam-osd2", "beam-osd3"):
        ls = int(list_size or 8)
        top = int(osd_top or min(ls, 8))
        order = 1 if strategy == "beam-osd1" else (2 if strategy == "beam-osd2" else 3)

        # Beam baseline (for guarding; we might end up choosing this)
        cw_b, sel_b, dist_b = rmcore.decode_rm_dumer_list_punctured(w_bytes, n, r, ls)
        code_b = unpack_bits_le(cw_b, L)

        # top-K beam seeds (bytes only)
        cw_list = [cw_b]
        if hasattr(rmcore, "decode_rm_dumer_list_topk_punctured"):
            cw_list = rmcore.decode_rm_dumer_list_topk_punctured(w_bytes, n, r, ls, top)

        # refine each seed by OSD-order and pick best candidate by (dist, depth)
        best_bits = None
        best_sel = None
        best_dist = 1 << 30
        best_depth = 1 << 30
        for cwb in cw_list:
            base_bits = unpack_bits_le(cwb, L)
            cand_bits, cand_sel, cand_dist = osd_decode(w_bits_list, n, r, base_bits, order=order)
            # compute depth lazily only when needed for comparison
            cand_depth = _depth_from_solution(w_bits_list, n, cand_bits, cand_sel)
            if (cand_dist < best_dist) or (
                    cand_dist == best_dist and cand_depth < best_depth
            ):
                best_bits, best_sel, best_dist, best_depth = cand_bits, cand_sel, cand_dist, cand_depth

        return (
            (best_bits, best_sel, best_dist)
            if _decide(code_b, sel_b, dist_b, best_bits, best_sel, best_dist)
            else (code_b, sel_b, dist_b)
        )

    if strategy == "rpa":
        # ergonomic alias: RPA with SNAP
        return decode_rm(
            w_bits_list,
            n,
            r,
            "rpa-adv",
            list_size=list_size,
            rpa_iters=rpa_iters,
            snap=True,
            snap_t=snap_t,
            snap_pool=snap_pool,
            snap_strong=snap_strong,
            snap_time_ms=snap_time_ms,
            snap_node_limit=snap_node_limit,
            depth_tradeoff=depth_tradeoff,
        )

    raise ValueError(f"Unknown decoder strategy: {strategy}")
