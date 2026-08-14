from __future__ import annotations

import os
import subprocess
import sys

import pytest

from rmsynth_reference import (
    CNOT,
    Circuit,
    LinearMap,
    LinearPhaseProgram,
    Phase,
    PhasePolynomial,
    ValidationError,
)
from rmsynth_reference.errors import ResourceLimitError
from rmsynth_reference.limits import MAX_OPERATIONS


@pytest.mark.parametrize("value", [0, -1, 11, True, 2.0, "2"])
def test_invalid_circuit_size(value: object) -> None:
    with pytest.raises(ValidationError):
        Circuit(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "gate",
    [CNOT(0, 1), Phase(0, 0), Phase(1, 7)],
)
def test_valid_gates(gate: CNOT | Phase) -> None:
    assert Circuit(2, (gate,)).operations == (gate,)


def test_gate_validation() -> None:
    with pytest.raises(ValidationError, match="differ"):
        CNOT(1, 1)
    with pytest.raises(ValidationError, match="outside"):
        Circuit(2, (Phase(2, 1),))
    with pytest.raises(ValidationError, match="between"):
        Phase(0, 8)


def test_phase_polynomial_validation() -> None:
    assert PhasePolynomial(2, (0, 1, 7)).t_count == 2
    with pytest.raises(ValidationError, match="expected 3"):
        PhasePolynomial(2, (0,))
    with pytest.raises(ValidationError):
        PhasePolynomial(1, (True,))


def test_linear_map_validation() -> None:
    assert LinearMap.identity(3).rows == (1, 2, 4)
    assert LinearMap((1, 3)).qubits == 2
    with pytest.raises(ValidationError, match="invertible"):
        LinearMap((1, 1))


def test_operation_limit() -> None:
    operations = (Phase(0, 0),) * (MAX_OPERATIONS + 1)
    with pytest.raises(ResourceLimitError, match="at most"):
        Circuit(1, operations)


def test_container_types_and_dimensions() -> None:
    with pytest.raises(ValidationError, match="tuple"):
        Circuit(1, [])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="only CNOT"):
        Circuit(1, (object(),))  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="tuple"):
        PhasePolynomial(1, [0])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="nonempty"):
        LinearMap(())
    with pytest.raises(ValidationError, match="between"):
        LinearMap((2,))
    with pytest.raises(ValidationError, match="equal dimensions"):
        LinearPhaseProgram(PhasePolynomial(1, (0,)), LinearMap.identity(2))


def test_validation_survives_optimized_python() -> None:
    code = "from rmsynth_reference import CNOT; CNOT(0, 0)"
    environment = {**os.environ, "PYTHONPATH": "src"}
    completed = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=os.getcwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "ValidationError" in completed.stderr
