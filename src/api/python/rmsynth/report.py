from dataclasses import dataclass
from typing import List

# Small result containers used by the optimiser and low-level decoders.
#
# These are intentionally simple dataclasses so they can easily be
# serialised, logged or printed in tests and CLI tools.

@dataclass
class DecodeReport:
    """
    Summary of a single RM decoding run (mainly used in Python-only paths)."""
    n: int
    r: int
    length: int
    dimension: int
    distance: int
    selected_monomials: List[int]
    codeword_bits: int
    ties: int


@dataclass
class OptimizeReport:
    """
    Summary of a full circuit optimisation pass.

    Fields
    n : int
        Number of qubits.
    before_t, after_t : int
        T-count before / after optimisation.
    distance : int
        Hamming distance of the chosen RM codeword (punctured metric).
    r : int
        RM order used (typically n-4 for T-optimisation).
    bitlen : int
        Length of the coefficient vector (2^n-1).
    selected_monomials : List[int]
        Monomials that form the applied correction.
    signature : str
        SHA-256 hash of the optimised coefficient vector; useful for
        regression tests and caching.
    """
    n: int
    before_t: int
    after_t: int
    distance: int
    r: int
    bitlen: int
    selected_monomials: List[int]
    signature: str

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (f"[rmsynth] n={self.n}, r={self.r}, length={self.bitlen}: "
                f"T-count {self.before_t} -> {self.after_t} (distance={self.distance}). "
                f"Signature={self.signature[:16]}...")
