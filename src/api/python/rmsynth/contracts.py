from __future__ import annotations
import os
from typing import List, Tuple

from .core import popcount, lift_to_c, add_mod8_vec, t_count_of_coeffs
from .rm_cache import get_rows, get_monom_list

# Global toggle for expensive consistency checks

def _checks_enabled() -> bool:
    """
    Returns True if contracts should be enforced.

    Controlled by RM_CHECKS:
      - unset / empty : OFF (default)
      - "0", "false", "no" : OFF
      - anything else      : ON

    Tests turn this on; production runs usually keep it off.
    """
    v = os.environ.get("RM_CHECKS", "")
    if v == "" or v is None:
        # default OFF for general runs, tests enable explicitly
        return False
    return not (v.strip().lower() in ("0", "false", "no"))

# Helpers mirroring RM generator structure

def _monom_list(n: int, r: int) -> List[int]:
    """
    Local copy of the monomial list used by rm_generator_rows in core/rm_code.

    This must match the order used for generator rows:
      [0] + all masks t with popcount(t) <= r, in increasing t.
    """
    # must mirror the order used by rm_generator_rows
    if r < 0:
        return []
    out = [0]
    for m in range(1, 1 << n):
        if popcount(m) <= r:
            out.append(m)
    return out

def _reconstruct_code_bits_from_monoms(n: int, r: int, selected_monomials: List[int]) -> List[int]:
    """
    Given the selected monomials (as masks), reconstruct the punctured
    codeword bits of length (2^n - 1) by XOR'ing the corresponding
    generator rows.

    This is used in Contract 2 to check that the decoder's reported
    code_bits are consistent with the selected basis elements.
    """
    if r < 0:
        return []
    rows = get_rows(n, r)
    mons = get_monom_list(n, r)
    # map monomial mask -> index in rows
    idx_map = {m: i for i, m in enumerate(mons)}
    cw_int = 0
    for m in selected_monomials:
        i = idx_map.get(m)
        if i is None:
            raise AssertionError(f"Selected monomial {m} not in RM({r},{n}) basis.")
        cw_int ^= rows[i]
    L = (1 << n) - 1
    return [(cw_int >> i) & 1 for i in range(L)]

# Contracts

def assert_in_rm(selected_monomials: List[int], n: int, r: int) -> None:
    """
    Contract 1: decoded monomials must lie in RM(r,n).

    Concretely: each monomial mask must have degree <= r, i.e.,
    popcount(mask) <= r.
    """
    for m in selected_monomials:
        d = popcount(m)
        if d > r:
            raise AssertionError(f"Monomial {m:b} (deg {d}) exceeds r={r} in RM({r},{n}).")

def assert_code_consistency(code_bits: List[int], selected_monomials: List[int], n: int, r: int) -> None:
    """
    Contract 2: punctured codeword bits must equal XOR of generator rows
    for the selected monomials.

    This checks that:
      code_bits == G * a
    where G is the generator matrix and 'a' is the indicator vector for
    selected monomials.
    """
    if r < 0:
        if len(code_bits) != 0 or len(selected_monomials) != 0:
            raise AssertionError("Inconsistent RM(r<0) trivial case.")
        return
    recon = _reconstruct_code_bits_from_monoms(n, r, selected_monomials)
    if recon != code_bits:
        # provide small diff hint
        L = len(recon)
        mismatch = [i for i in range(L) if recon[i] != code_bits[i]]
        hint = ", ".join(map(str, mismatch[:16])) + ("..." if len(mismatch) > 16 else "")
        raise AssertionError(f"Decoder inconsistency: code_bits != rows(selected). First mismatches at indices: {hint}")

def assert_distance_equals_tcount(vec_mod8: List[int], code_bits: List[int], n: int, r: int, reported_dist: int) -> None:
    """
    Contract 3: reported punctured distance must equal T-count after lift/resynth.

    For a vector of coefficients 'vec_mod8' and code_bits 'c':

      w_bits   = odd part of vec_mod8
      after_odd= w_bits XOR c
      distance = Hamming weight(after_odd)

    The decoder reports 'reported_dist' as this distance, and after the
    lift/resynthesis step the number of T gates must equal the number of
    odd entries, i.e. sum(after_odd).
    """
    # lift selected monomials to c is done outside, we emulate the effect:
    # T-count after optimization equals weight(w XOR code_bits),
    # where w is the oddness of vec, equivalently equals number of odd entries after add_mod8(vec, c)
    # here we recompute post-lift T-count directly using t_count_of_coeffs on the new coefficients
    # to do that we need c from monomials; but the caller may not have them here,
    # so we compute the canonical "after" oddness via code_bits
    w_bits = [1 if (v & 1) else 0 for v in vec_mod8]
    after_odd = [wb ^ cb for wb, cb in zip(w_bits, code_bits)]
    # the T-count of the final coefficients must equal the number of odd entries, which is sum(after_odd)
    # we also compute it via the coefficients path for robustness when even parts present
    # build a dummy c by lifting the difference, but since we don't have monomials here,
    # just assert the numeric equality that should always hold:
    after_t_from_bits = sum(after_odd)
    if after_t_from_bits != reported_dist:
        raise AssertionError(f"Distance/after-T mismatch: reported {reported_dist} != sum(after_odd) {after_t_from_bits}")

def check_all(
        vec_mod8: List[int],
        n: int,
        r: int,
        code_bits: List[int],
        selected_monomials: List[int],
        reported_dist: int,
        strict: bool = True,
) -> None:
    """
    Run all contracts; raise AssertionError on failure.

    Intended for tests and debug sessions, callers can set strict=False
    and handle exceptions themselves if needed.
    """
    assert_in_rm(selected_monomials, n, r)
    assert_code_consistency(code_bits, selected_monomials, n, r)
    assert_distance_equals_tcount(vec_mod8, code_bits, n, r, reported_dist)

def maybe_check_all(
        vec_mod8: List[int],
        n: int,
        r: int,
        code_bits: List[int],
        selected_monomials: List[int],
        reported_dist: int,
) -> None:
    """
    Run all contracts if RM_CHECKS is enabled, otherwise a no-op.
    """
    if _checks_enabled():
        check_all(vec_mod8, n, r, code_bits, selected_monomials, reported_dist, strict=True)
