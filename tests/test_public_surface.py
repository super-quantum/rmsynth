from __future__ import annotations

from importlib.metadata import version

import rmsynth_reference


def test_public_api_is_explicit() -> None:
    assert rmsynth_reference.__all__ == [
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


def test_package_version_is_consistent() -> None:
    assert version("rmsynth-reference") == rmsynth_reference.__version__
