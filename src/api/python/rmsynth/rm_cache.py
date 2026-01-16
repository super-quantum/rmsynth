from __future__ import annotations
from functools import lru_cache
from typing import List
from .core import rm_generator_rows, popcount

# Small LRU cache for RM generator matrices and monomial lists.
#
# Many parts of the Python layer (decoders, contracts, OSD, etc.) need
# the generator rows and the list of monomial masks for RM(r,n). These
# are pure functions of (n, r), so we memoise them to avoid recomputing
# combinatorial structures over and over again.


@lru_cache(maxsize=16)
def get_rows(n: int, r: int) -> list[int]:
    """LRU-cached generator rows for RM(r,n) in punctured order."""
    return rm_generator_rows(n, r)


@lru_cache(maxsize=16)
def get_monom_list(n: int, r: int) -> list[int]:
    """
    LRU-cached monomial masks for RM(r,n) in the same order as get_rows().

    Returns:
        [0] + [mask for mask in 1..(2^n-1) with popcount(mask) <= r]
    """
    if r < 0:
        return []
    out = [0]
    for t in range(1, 1 << n):
        if popcount(t) <= r:
            out.append(t)
    return out
