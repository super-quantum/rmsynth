from __future__ import annotations

from dataclasses import dataclass

from .errors import ValidationError
from .model import CNOT, Circuit, PhasePolynomial


@dataclass(frozen=True, slots=True)
class VerificationWitness:
    """The first basis input on which two objects differ."""

    input_state: int
    expected_state: int
    actual_state: int
    expected_phase: int
    actual_phase: int


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """An equivalence result with an optional counterexample."""

    equivalent: bool
    witness: VerificationWitness | None = None


def run_circuit(circuit: Circuit, input_state: int) -> tuple[int, int]:
    if type(input_state) is not int or not 0 <= input_state < 1 << circuit.qubits:
        raise ValidationError(f"input state must be an integer in [0, {1 << circuit.qubits})")
    state = input_state
    phase = 0
    for operation in circuit.operations:
        if isinstance(operation, CNOT):
            if state >> operation.control & 1:
                state ^= 1 << operation.target
        elif state >> operation.qubit & 1:
            phase = (phase + operation.exponent) % 8
    return state, phase


def verify_circuits(expected: Circuit, actual: Circuit) -> VerificationResult:
    """Compare two circuits on every computational-basis input."""
    if expected.qubits != actual.qubits:
        raise ValidationError("circuits must have the same number of qubits")
    for input_state in range(1 << expected.qubits):
        expected_state, expected_phase = run_circuit(expected, input_state)
        actual_state, actual_phase = run_circuit(actual, input_state)
        if (expected_state, expected_phase) != (actual_state, actual_phase):
            return VerificationResult(
                False,
                VerificationWitness(
                    input_state,
                    expected_state,
                    actual_state,
                    expected_phase,
                    actual_phase,
                ),
            )
    return VerificationResult(True)


def verify_phase_polynomials(
    expected: PhasePolynomial, actual: PhasePolynomial
) -> VerificationResult:
    """Compare two phase polynomials on every input."""
    if expected.qubits != actual.qubits:
        raise ValidationError("phase polynomials must have the same number of qubits")
    for input_state in range(1 << expected.qubits):
        expected_phase = _phase(expected, input_state)
        actual_phase = _phase(actual, input_state)
        if expected_phase != actual_phase:
            return VerificationResult(
                False,
                VerificationWitness(
                    input_state, input_state, input_state, expected_phase, actual_phase
                ),
            )
    return VerificationResult(True)


def _phase(polynomial: PhasePolynomial, state: int) -> int:
    phase = 0
    for mask in range(1, 1 << polynomial.qubits):
        if (mask & state).bit_count() % 2:
            phase += polynomial.coefficients[mask - 1]
    return phase % 8
