from ._version import __version__
from .errors import ResourceLimitError, RMSynthError, ValidationError, VerificationError
from .model import CNOT, Circuit, LinearMap, LinearPhaseProgram, Phase, PhasePolynomial
from .optimizer import OptimizationResult, optimize
from .report import OptimizationReport
from .semantics import evaluate_polynomial, evaluate_program, extract_program
from .synthesis import synthesize_program
from .verify import (
    VerificationResult,
    VerificationWitness,
    verify_circuits,
    verify_phase_polynomials,
)

__all__ = [
    "CNOT",
    "Circuit",
    "LinearMap",
    "LinearPhaseProgram",
    "OptimizationReport",
    "OptimizationResult",
    "Phase",
    "PhasePolynomial",
    "RMSynthError",
    "ResourceLimitError",
    "ValidationError",
    "VerificationError",
    "VerificationResult",
    "VerificationWitness",
    "__version__",
    "evaluate_polynomial",
    "evaluate_program",
    "extract_program",
    "optimize",
    "synthesize_program",
    "verify_circuits",
    "verify_phase_polynomials",
]
