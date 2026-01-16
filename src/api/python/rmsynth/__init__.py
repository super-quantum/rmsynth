"""
rmsynth - Reed-Muller decoding-based optimizer for linear-phase quantum circuits.

Public API exports are lazy-loaded to avoid import cycles.
Type information is provided by __init__.pyi stub file.
"""
from __future__ import annotations

# pyright: reportUnsupportedDunderAll=false
# __all__ is declared in the .pyi stub; at runtime it's set here for help()/dir()
__all__ = [
    "Optimizer",
    "CostModel",
    "Circuit",
    "Gate",
    "optimize_coefficients",
    "extract_phase_coeffs",
    "synthesize_from_coeffs",
    "t_count_of_coeffs",
]


def __getattr__(name: str) -> object:
    """Lazy import to avoid circular dependencies."""
    if name in ("Optimizer", "CostModel"):
        from .optimizer import Optimizer, CostModel
        return Optimizer if name == "Optimizer" else CostModel
    if name in ("Circuit", "Gate", "optimize_coefficients", "extract_phase_coeffs",
                "synthesize_from_coeffs", "t_count_of_coeffs"):
        from . import core
        return getattr(core, name)  # pyright: ignore[reportAny]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

