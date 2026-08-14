from __future__ import annotations

from . import _native
from .errors import ResourceLimitError, ValidationError
from .limits import MAX_OPTIMIZER_QUBITS
from .model import CNOT, Circuit, LinearMap, LinearPhaseProgram, Operation, Phase


def synthesize_linear_map(linear_map: LinearMap) -> tuple[CNOT, ...]:
    if linear_map.qubits > MAX_OPTIMIZER_QUBITS:
        raise ResourceLimitError(f"synthesis supports at most {MAX_OPTIMIZER_QUBITS} qubits")
    try:
        gates = _native.synthesize_linear_map(list(linear_map.rows))
    except _native.ValidationError as error:
        raise ValidationError(str(error)) from error
    return tuple(CNOT(control, target) for control, target in gates)


def synthesize_program(program: LinearPhaseProgram) -> Circuit:
    """Synthesize a bounded linear-phase program deterministically."""
    if program.qubits > MAX_OPTIMIZER_QUBITS:
        raise ResourceLimitError(f"synthesis supports at most {MAX_OPTIMIZER_QUBITS} qubits")
    operations: list[Operation] = []
    for mask, exponent in enumerate(program.phase_polynomial.coefficients, start=1):
        if exponent == 0:
            continue
        pivot = (mask & -mask).bit_length() - 1
        controls = [
            qubit for qubit in range(program.qubits) if qubit != pivot and mask >> qubit & 1
        ]
        operations.extend(CNOT(control, pivot) for control in controls)
        operations.append(Phase(pivot, exponent))
        operations.extend(CNOT(control, pivot) for control in reversed(controls))
    operations.extend(synthesize_linear_map(program.linear_map))
    return Circuit(program.qubits, tuple(operations))
