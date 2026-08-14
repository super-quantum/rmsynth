from __future__ import annotations

from itertools import product

import pytest
from hypothesis import given
from hypothesis import strategies as st

import rmsynth_reference.semantics as semantics_module
import rmsynth_reference.synthesis as synthesis_module
from rmsynth_reference import (
    CNOT,
    Circuit,
    LinearMap,
    LinearPhaseProgram,
    Phase,
    PhasePolynomial,
    evaluate_program,
    extract_program,
    verify_circuits,
)
from rmsynth_reference.errors import ResourceLimitError, ValidationError
from rmsynth_reference.synthesis import synthesize_linear_map, synthesize_program
from rmsynth_reference.verify import run_circuit as execute_circuit
from rmsynth_reference.verify import verify_phase_polynomials
from tests._oracles import rank, run_circuit


def test_empty_and_phase_semantics() -> None:
    empty = extract_program(Circuit(2))
    assert empty.linear_map == LinearMap.identity(2)
    circuit = Circuit(2, (Phase(0, 3), Phase(0, 7), Phase(1, 2)))
    program = extract_program(circuit)
    assert program.phase_polynomial.coefficients == (2, 2, 0)
    for state in range(4):
        assert evaluate_program(program, state) == run_circuit(circuit, state)


def test_terminal_cnot_and_swap() -> None:
    cnot = Circuit(2, (CNOT(0, 1),))
    assert extract_program(cnot).linear_map.rows == (1, 3)
    swap = Circuit(2, (CNOT(0, 1), CNOT(1, 0), CNOT(0, 1)))
    assert extract_program(swap).linear_map.rows == (2, 1)


def test_mixed_circuit_matches_independent_oracle() -> None:
    circuit = Circuit(
        3,
        (
            Phase(0, 1),
            CNOT(0, 1),
            Phase(1, 3),
            CNOT(2, 0),
            Phase(0, 6),
            CNOT(1, 2),
        ),
    )
    program = extract_program(circuit)
    for state in range(8):
        assert evaluate_program(program, state) == run_circuit(circuit, state)


def test_all_three_qubit_linear_maps() -> None:
    invertible = 0
    for rows in product(range(8), repeat=3):
        if rank(rows, 3) != 3:
            try:
                LinearMap(rows)
            except ValidationError:
                continue
            raise AssertionError("singular map accepted")
        invertible += 1
        linear_map = LinearMap(rows)
        circuit = Circuit(3, synthesize_linear_map(linear_map))
        assert extract_program(circuit).linear_map == linear_map
        for state in range(8):
            expected = sum(((row & state).bit_count() & 1) << bit for bit, row in enumerate(rows))
            assert run_circuit(circuit, state) == (expected, 0)
    assert invertible == 168


@st.composite
def programs(draw: st.DrawFn) -> LinearPhaseProgram:
    qubits = draw(st.integers(min_value=1, max_value=5))
    coefficients = draw(
        st.lists(
            st.integers(min_value=0, max_value=7),
            min_size=(1 << qubits) - 1,
            max_size=(1 << qubits) - 1,
        )
    )
    return LinearPhaseProgram(
        PhasePolynomial(qubits, tuple(coefficients)), LinearMap.identity(qubits)
    )


@given(programs())
def test_phase_program_round_trip(program: LinearPhaseProgram) -> None:
    circuit = synthesize_program(program)
    assert extract_program(circuit) == program
    assert verify_circuits(circuit, synthesize_program(extract_program(circuit))).equivalent


def test_verifier_witnesses() -> None:
    phase_mismatch = verify_circuits(Circuit(1), Circuit(1, (Phase(0, 1),)))
    assert not phase_mismatch.equivalent
    assert phase_mismatch.witness is not None
    assert phase_mismatch.witness.input_state == 1
    map_mismatch = verify_circuits(Circuit(2), Circuit(2, (CNOT(0, 1),)))
    assert not map_mismatch.equivalent
    assert map_mismatch.witness is not None


def test_verifier_validation_and_phase_witness() -> None:
    with pytest.raises(ValidationError, match="same number"):
        verify_circuits(Circuit(1), Circuit(2))
    with pytest.raises(ValidationError, match="input state"):
        execute_circuit(Circuit(1), 2)
    first = PhasePolynomial(1, (0,))
    second = PhasePolynomial(1, (1,))
    result = verify_phase_polynomials(first, second)
    assert not result.equivalent
    assert result.witness is not None
    with pytest.raises(ValidationError, match="same number"):
        verify_phase_polynomials(first, PhasePolynomial(2, (0, 0, 0)))


def test_verifier_does_not_depend_on_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: Circuit) -> None:
        raise AssertionError("extraction should not run")

    monkeypatch.setattr(semantics_module, "extract_program", fail)
    assert verify_circuits(Circuit(1), Circuit(1)).equivalent


def test_synthesis_size_limit() -> None:
    program = LinearPhaseProgram(PhasePolynomial(6, (0,) * 63), LinearMap.identity(6))
    with pytest.raises(ResourceLimitError, match="at most 5"):
        synthesize_program(program)


def test_native_synthesis_error_is_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: list[int]) -> list[tuple[int, int]]:
        raise synthesis_module._native.ValidationError("invalid native map")

    monkeypatch.setattr(synthesis_module._native, "synthesize_linear_map", fail)
    with pytest.raises(ValidationError, match="invalid native map"):
        synthesize_linear_map(LinearMap.identity(2))
