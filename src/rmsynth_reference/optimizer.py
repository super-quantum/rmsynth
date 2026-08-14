from __future__ import annotations

from dataclasses import dataclass

from .decoder import decode_exact
from .errors import ResourceLimitError, VerificationError
from .limits import MAX_OPTIMIZER_QUBITS
from .model import Circuit, LinearPhaseProgram, PhasePolynomial
from .report import OptimizationReport
from .semantics import circuit_digest, extract_program, program_digest
from .synthesis import synthesize_program
from .verify import verify_circuits, verify_phase_polynomials


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """An optimized circuit and its deterministic report."""

    circuit: Circuit
    report: OptimizationReport


def optimize(circuit: Circuit) -> OptimizationResult:
    """Return an equivalent circuit only when its T-count is lower."""
    if circuit.qubits > MAX_OPTIMIZER_QUBITS:
        raise ResourceLimitError(f"optimization supports at most {MAX_OPTIMIZER_QUBITS} qubits")

    original = extract_program(circuit)
    received = sum(
        (coefficient % 2) << index
        for index, coefficient in enumerate(original.phase_polynomial.coefficients)
    )
    order = max(-1, circuit.qubits - 4)
    decoded = decode_exact(received, circuit.qubits, order)
    coefficients = tuple(
        (coefficient + (decoded.codeword >> index & 1)) % 8
        for index, coefficient in enumerate(original.phase_polynomial.coefficients)
    )
    candidate_phase = PhasePolynomial(circuit.qubits, coefficients)
    if not verify_phase_polynomials(original.phase_polynomial, candidate_phase).equivalent:
        raise VerificationError("decoder correction changed the phase function")

    candidate_program = LinearPhaseProgram(candidate_phase, original.linear_map)
    candidate = synthesize_program(candidate_program)
    if not verify_circuits(circuit, candidate).equivalent:
        raise VerificationError("synthesized circuit is not equivalent to its input")

    improved = candidate.t_count < circuit.t_count
    output = candidate if improved else circuit
    output_program = candidate_program if improved else original
    report = OptimizationReport(
        status="improved" if improved else "unchanged",
        reason="strict_t_count_reduction" if improved else "no_strict_improvement",
        qubits=circuit.qubits,
        reed_muller_order=order,
        before_t_count=circuit.t_count,
        after_t_count=output.t_count,
        candidates=decoded.candidates,
        ties=decoded.ties,
        input_digest=circuit_digest(circuit),
        output_digest=circuit_digest(output),
        semantic_digest=program_digest(output_program),
        verified=True,
    )
    return OptimizationResult(output, report)
