from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

import rmsynth_reference.optimizer as optimizer_module
from rmsynth_reference import CNOT, Circuit, LinearMap, LinearPhaseProgram, Phase, PhasePolynomial
from rmsynth_reference.decoder import DecodeResult
from rmsynth_reference.errors import ResourceLimitError, VerificationError
from rmsynth_reference.optimizer import optimize
from rmsynth_reference.synthesis import synthesize_program
from rmsynth_reference.verify import verify_circuits


def circuit_from_coefficients(qubits: int, coefficients: tuple[int, ...]) -> Circuit:
    return synthesize_program(
        LinearPhaseProgram(PhasePolynomial(qubits, coefficients), LinearMap.identity(qubits))
    )


def test_terminal_cnot_regression() -> None:
    circuit = Circuit(2, (CNOT(0, 1),))
    result = optimize(circuit)
    assert result.circuit is circuit
    assert result.report.status == "unchanged"
    assert verify_circuits(circuit, result.circuit).equivalent


def test_all_parities_quickstart() -> None:
    circuit = circuit_from_coefficients(4, (1,) * 15)
    result = optimize(circuit)
    assert circuit.t_count == 15
    assert result.circuit.t_count == 0
    assert result.report.status == "improved"
    assert verify_circuits(circuit, result.circuit).equivalent


def test_nonworsening_regression() -> None:
    odd_positions = {0, 13, 16, 22, 27, 28, 29}
    coefficients = tuple(1 if index in odd_positions else 0 for index in range(31))
    circuit = circuit_from_coefficients(5, coefficients)
    result = optimize(circuit)
    assert circuit.t_count == 7
    assert result.circuit.t_count <= 7
    assert verify_circuits(circuit, result.circuit).equivalent


def test_cancellation_is_canonicalized() -> None:
    circuit = Circuit(1, (Phase(0, 1), Phase(0, 1)))
    result = optimize(circuit)
    assert result.report.before_t_count == 2
    assert result.report.after_t_count == 0
    assert verify_circuits(circuit, result.circuit).equivalent


def test_equal_candidate_keeps_original_object() -> None:
    circuit = Circuit(2, (Phase(0, 1), CNOT(0, 1)))
    result = optimize(circuit)
    assert result.circuit is circuit
    assert result.report.input_digest == result.report.output_digest


@st.composite
def small_circuits(draw: st.DrawFn) -> Circuit:
    qubits = draw(st.integers(min_value=1, max_value=5))
    gate_count = draw(st.integers(min_value=0, max_value=14))
    operations = []
    for _ in range(gate_count):
        if qubits > 1 and draw(st.booleans()):
            control = draw(st.integers(min_value=0, max_value=qubits - 1))
            target = draw(st.integers(min_value=0, max_value=qubits - 2))
            if target >= control:
                target += 1
            operations.append(CNOT(control, target))
        else:
            operations.append(
                Phase(
                    draw(st.integers(min_value=0, max_value=qubits - 1)),
                    draw(st.integers(min_value=0, max_value=7)),
                )
            )
    return Circuit(qubits, tuple(operations))


@given(small_circuits())
def test_optimizer_is_equivalent_and_nonworsening(circuit: Circuit) -> None:
    original_operations = circuit.operations
    result = optimize(circuit)
    assert result.circuit.t_count <= circuit.t_count
    assert verify_circuits(circuit, result.circuit).equivalent
    assert circuit.operations == original_operations
    second = optimize(result.circuit)
    assert second.circuit == result.circuit


def test_report_is_deterministic() -> None:
    circuit = circuit_from_coefficients(4, (1,) * 15)
    assert optimize(circuit) == optimize(circuit)


def test_worse_candidate_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        optimizer_module,
        "decode_exact",
        lambda *_: DecodeResult((1 << 15) - 1, (0,), 15, 2, 1),
    )
    circuit = Circuit(4)
    result = optimize(circuit)
    assert result.circuit is circuit
    assert result.report.status == "unchanged"


def test_decoder_equivalence_failure_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        optimizer_module,
        "decode_exact",
        lambda *_: DecodeResult(1, (), 1, 1, 1),
    )
    with pytest.raises(VerificationError, match="decoder correction"):
        optimize(Circuit(4))


def test_synthesis_equivalence_failure_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def wrong_circuit(_: LinearPhaseProgram) -> Circuit:
        return Circuit(1)

    monkeypatch.setattr(optimizer_module, "synthesize_program", wrong_circuit)
    with pytest.raises(VerificationError, match="synthesized circuit"):
        optimize(Circuit(1, (Phase(0, 1),)))


def test_optimizer_size_limit() -> None:
    with pytest.raises(ResourceLimitError, match="at most 5"):
        optimize(Circuit(6))
