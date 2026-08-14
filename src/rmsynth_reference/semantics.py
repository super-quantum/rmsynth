from __future__ import annotations

import hashlib
import json

from .errors import ValidationError
from .model import CNOT, Circuit, LinearMap, LinearPhaseProgram, PhasePolynomial


def extract_program(circuit: Circuit) -> LinearPhaseProgram:
    """Extract the phase polynomial and terminal map from a circuit."""
    forms = [1 << index for index in range(circuit.qubits)]
    coefficients = [0] * ((1 << circuit.qubits) - 1)
    for operation in circuit.operations:
        if isinstance(operation, CNOT):
            forms[operation.target] ^= forms[operation.control]
        else:
            mask = forms[operation.qubit]
            coefficients[mask - 1] = (coefficients[mask - 1] + operation.exponent) % 8
    return LinearPhaseProgram(
        PhasePolynomial(circuit.qubits, tuple(coefficients)), LinearMap(tuple(forms))
    )


def evaluate_polynomial(polynomial: PhasePolynomial, input_state: int) -> int:
    """Evaluate a phase polynomial modulo 8 on one input bit string."""
    _state(polynomial.qubits, input_state)
    return (
        sum(
            coefficient
            for mask, coefficient in enumerate(polynomial.coefficients, start=1)
            if (mask & input_state).bit_count() % 2
        )
        % 8
    )


def evaluate_program(program: LinearPhaseProgram, input_state: int) -> tuple[int, int]:
    """Return the output bit string and phase exponent for one input."""
    _state(program.qubits, input_state)
    output = sum(
        ((row & input_state).bit_count() % 2) << index
        for index, row in enumerate(program.linear_map.rows)
    )
    return output, evaluate_polynomial(program.phase_polynomial, input_state)


def program_digest(program: LinearPhaseProgram) -> str:
    data = {
        "coefficients": program.phase_polynomial.coefficients,
        "linear_map": program.linear_map.rows,
        "qubits": program.qubits,
    }
    encoded = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def circuit_digest(circuit: Circuit) -> str:
    operations = []
    for operation in circuit.operations:
        if isinstance(operation, CNOT):
            operations.append(("cnot", operation.control, operation.target))
        else:
            operations.append(("phase", operation.qubit, operation.exponent))
    data = {"operations": operations, "qubits": circuit.qubits}
    encoded = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _state(qubits: int, input_state: int) -> None:
    if type(input_state) is not int or not 0 <= input_state < 1 << qubits:
        raise ValidationError(f"input state must be an integer in [0, {1 << qubits})")
