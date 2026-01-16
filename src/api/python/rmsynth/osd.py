from __future__ import annotations
from typing import List, Tuple
import os

from .rm_cache import get_rows, get_monom_list
from .core import popcount
from . import rmcore  # for depth-aware tie-break

# Ordered Statistics Decoding (OSD) for RM(r,n)
#
# This module implements OSD of order 1/2/3 around a baseline codeword.
# It is used as a refinement step on top of Dumer / RPA decoders.
#
# Given:
#   - w_bits      : punctured observation word (length 2^n-1),
#   - base_code   : baseline RM codeword near w_bits,
#
# osd_decode(...) returns a better (or equal) codeword by searching a small neighbourhood around the information set using OSD. Ties in distance are broken using T-depth obtained from rmcore.


# bit helpers (little-endian)

def _bits_to_int_le(bits: List[int]) -> int:
    """Pack bits (LSB-first) into an integer."""
    x = 0
    for i, b in enumerate(bits):
        if b & 1:
            x |= (1 << i)
    return x


def _int_to_bits_le(x: int, L: int) -> List[int]:
    """Unpack an integer into L bits (LSB-first)."""
    return [(x >> i) & 1 for i in range(L)]


def _pack_bits_le(bits: List[int]) -> bytes:
    """Pack bits into a little-endian byte string."""
    L = len(bits)
    nbytes = (L + 7) // 8
    buf = bytearray(nbytes)
    for i, b in enumerate(bits):
        if b & 1:
            buf[i >> 3] |= (1 << (i & 7))
    return bytes(buf)


# build punctured G (generator) and columns

def _rows_int(n: int, r: int) -> List[int]:
    """Generator rows (punctured) as Python ints with L bits (bit i is column i)."""
    return get_rows(n, r)


def _monoms(n: int, r: int) -> List[int]:
    """Monomial masks aligned with _rows_int order."""
    return get_monom_list(n, r)


def _columns_from_rows(rows: List[int], L: int, K: int) -> List[int]:
    """
    Convert row-oriented G (K rows of L bits) to column vectors (L ints of K bits).

    col[j] bit i = 1 iff row i has bit j set.
    """
    cols = [0] * L
    for i in range(K):
        ri = rows[i]
        x = ri
        while x:
            lsb = x & -x
            j = lsb.bit_length() - 1
            cols[j] |= (1 << i)   # set row i in column j
            x ^= lsb
    return cols


# select information set of size K by reliability rank

def _column_weight_by_mask_weight(n: int, r: int, col_index: int) -> int:
    """
    Heuristic weight used in reliability ranking for columns.

    It approximates the number of RM(r,n) monomials that involve this
    coordinate, based on the Hamming weight of the mask j+1.
    """
    m = col_index + 1
    wt = popcount(m)
    s = 0
    from math import comb
    for d in range(0, min(r, wt) + 1):
        s += comb(wt, d)
    return s


def _select_info_set(
        w_bits: List[int],
        c_bits: List[int],
        n: int,
        r: int,
        rows: List[int],
        cols: List[int],
) -> List[int]:
    """
    Greedy selection of K columns with highest reliability that remain
    linearly independent.

    Reliability key: (agree_with_baseline, column_weight_by_mask, -index)
    in descending order, where 'agree_with_baseline' indicates whether
    w_bits[j] == c_bits[j].

    Independence is enforced via a proper GF(2) linear basis using MSB
    pivots, guaranteeing we obtain a full-rank information set.
    """
    L = len(w_bits)
    K = len(rows)
    rel = []
    for j in range(L):
        agree = 1 if w_bits[j] == c_bits[j] else 0
        tiew = _column_weight_by_mask_weight(n, r, j)
        rel.append((agree, tiew, -j, j))
    rel.sort(reverse=True)

    # MSB-pivot linear basis (for K-bit column vectors)
    basis = [0] * K

    def _insert_vec_msb(v: int) -> bool:
        vv = v
        while vv:
            p = vv.bit_length() - 1  # MSB position
            if basis[p]:
                vv ^= basis[p]
            else:
                basis[p] = vv
                return True
        return False

    chosen: List[int] = []
    for *_, j in rel:
        if _insert_vec_msb(cols[j]):
            chosen.append(j)
            if len(chosen) == K:
                return chosen

    # fallback: try remaining columns in order, still enforcing independence
    for j in range(L):
        if j in chosen:
            continue
        if _insert_vec_msb(cols[j]):
            chosen.append(j)
            if len(chosen) == K:
                break

    if len(chosen) < K:
        raise RuntimeError("OSD: could not find full-rank information set for RM(r,n).")
    return chosen


# GF(2) inverse (K x K) mat

def _gf2_inv(rows_kxk: List[int], K: int) -> List[int]:
    """
    Gauss-Jordan over GF(2) on KxK matrix given as list of K row-int
    bitmasks. Returns inverse as list of K row-int bitmasks (rows of A^{-1}).
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
            raise RuntimeError("OSD: Singular matrix in inversion.")
        if pivot != col:
            AugL[col], AugL[pivot] = AugL[pivot], AugL[col]
            AugR[col], AugR[pivot] = AugR[pivot], AugR[col]
        for row in range(K):
            if row == col:
                continue
            if (AugL[row] >> col) & 1:
                AugL[row] ^= AugL[col]
                AugR[row] ^= AugR[col]
    return AugR


# build base codeword from coefficients int

def _encode_from_coeffs(rows: List[int], a_bits: int, K: int) -> int:
    """
    Return punctured codeword int: XOR rows[i] for all i where bit i in
    a_bits is 1.
    """
    cw = 0
    x = a_bits
    while x:
        lsb = x & -x
        i = lsb.bit_length() - 1
        cw ^= rows[i]
        x ^= lsb
    return cw


# OSD L1 / L2 / L3 main entry

def osd_decode(
        w_bits: List[int],
        n: int,
        r: int,
        base_code_bits: List[int],
        order: int = 1,
) -> Tuple[List[int], List[int], int]:
    """
    Perform OSD of order 1, 2 or 3 around a baseline codeword.

    Parameters
    ----------
    w_bits : List[int]
        Punctured received word (length 2^n-1).
    n, r : int
        RM(r,n) parameters.
    base_code_bits : List[int]
        Baseline codeword used to rank reliabilities.
    order : {1,2,3}
        OSD order (maximum size of information-set flip patterns).

    Returns
    -------
    code_bits : List[int]
        Improved (or equal) codeword.
    selected_monomials : List[int]
        Monomials corresponding to the final coefficient vector.
    distance : int
        Hamming distance to w_bits.
    """
    assert order in (1, 2, 3)
    L = len(w_bits)
    rows = _rows_int(n, r)
    K = len(rows)
    cols = _columns_from_rows(rows, L, K)

    # 1) info set (size K) ranked by reliability, guaranteed independent
    info = _select_info_set(w_bits, base_code_bits, n, r, rows, cols)  # len K

    # 2) build KxK submatrix A and invert
    A_rows = []
    for i in range(K):
        row_bits = 0
        for j, col_idx in enumerate(info):
            if (rows[i] >> col_idx) & 1:
                row_bits |= (1 << j)
        A_rows.append(row_bits)
    Ainv_rows = _gf2_inv(A_rows, K)

    # 3) s = received bits at info positions, as K-bit int (column vector)
    s_bits = 0
    for j, col_idx in enumerate(info):
        if w_bits[col_idx] & 1:
            s_bits |= (1 << j)

    # 4) a = A^{-1} s  (K-bit coefficients). compute row-wise: a_j = parity(row_j & s)
    a_bits = 0
    for j in range(K):
        if (Ainv_rows[j] & s_bits).bit_count() & 1:
            a_bits |= (1 << j)

    # 5) base codeword (as int)
    cw0 = _encode_from_coeffs(rows, a_bits, K)

    # 6) precompute delta codewords for unit flips in info set
    Ainv_cols: List[int] = []
    delta_cw: List[int] = []
    for j in range(K):
        col_bits = 0
        for i in range(K):
            if (Ainv_rows[i] >> j) & 1:
                col_bits |= (1 << i)
        Ainv_cols.append(col_bits)
        delta_cw.append(_encode_from_coeffs(rows, col_bits, K))

    w_int = _bits_to_int_le(w_bits)

    def _depth_of(cw_int: int) -> int:
        """
        Compute T-depth upper bound from residual odd set using rmcore's
        tdepth_from_punctured. Used as a tie-breaker between codewords
        with equal Hamming distance.
        """
        odd_bits = _int_to_bits_le((w_int ^ cw_int), L)
        return rmcore.tdepth_from_punctured(_pack_bits_le(odd_bits), n)

    best_cw = cw0
    best_a = a_bits
    best_dist = (w_int ^ cw0).bit_count()
    best_depth = _depth_of(best_cw)

    # 7) L1 neighborhood (flip each info bit once)
    for j in range(K):
        cw = cw0 ^ delta_cw[j]
        dist = (w_int ^ cw).bit_count()
        if dist < best_dist:
            best_dist, best_cw, best_a, best_depth = dist, cw, (a_bits ^ Ainv_cols[j]), _depth_of(cw)
        elif dist == best_dist:
            d = _depth_of(cw)
            if d < best_depth:
                best_cw, best_a, best_depth = cw, (a_bits ^ Ainv_cols[j]), d

    # 8) L2 neighborhood (optional)
    if order >= 2:
        limit_pairs = int(os.environ.get("RM_OSD2_MAX_PAIRS", "0"))
        checked = 0
        for i in range(K):
            for j in range(i + 1, K):
                if limit_pairs and checked >= limit_pairs:
                    break
                cw = cw0 ^ delta_cw[i] ^ delta_cw[j]
                dist = (w_int ^ cw).bit_count()
                if dist < best_dist:
                    best_dist, best_cw, best_a, best_depth = (
                        dist,
                        cw,
                        (a_bits ^ Ainv_cols[i] ^ Ainv_cols[j]),
                        _depth_of(cw),
                    )
                elif dist == best_dist:
                    d = _depth_of(cw)
                    if d < best_depth:
                        best_cw, best_a, best_depth = cw, (a_bits ^ Ainv_cols[i] ^ Ainv_cols[j]), d
                checked += 1

    # 9) L3 neighborhood (optional)
    if order >= 3:
        limit_triples = int(os.environ.get("RM_OSD3_MAX_TRIPLES", "0"))
        checked = 0
        for i in range(K):
            for j in range(i + 1, K):
                for k in range(j + 1, K):
                    if limit_triples and checked >= limit_triples:
                        break
                    cw = cw0 ^ delta_cw[i] ^ delta_cw[j] ^ delta_cw[k]
                    dist = (w_int ^ cw).bit_count()
                    if dist < best_dist:
                        best_dist, best_cw, best_a, best_depth = (
                            dist,
                            cw,
                            (a_bits ^ Ainv_cols[i] ^ Ainv_cols[j] ^ Ainv_cols[k]),
                            _depth_of(cw),
                        )
                    elif dist == best_dist:
                        d = _depth_of(cw)
                        if d < best_depth:
                            best_cw, best_a, best_depth = (
                                cw,
                                (a_bits ^ Ainv_cols[i] ^ Ainv_cols[j] ^ Ainv_cols[k]),
                                d,
                            )
                    checked += 1
                if limit_triples and checked >= limit_triples:
                    break
            if limit_triples and checked >= limit_triples:
                break

    code_bits = _int_to_bits_le(best_cw, L)
    mons_all = _monoms(n, r)
    selected = [mons_all[i] for i in range(K) if ((best_a >> i) & 1)]
    return code_bits, selected, best_dist


def osd_refine_l1(
        w_bits: List[int],
        n: int,
        r: int,
        base_code_bits: List[int],
) -> Tuple[List[int], List[int], int]:
    """
    Convenience wrapper: perform OSD of order 1 around base_code_bits.

    This is used as a cheap refinement step in several decoders.
    """
    return osd_decode(w_bits, n, r, base_code_bits, order=1)
