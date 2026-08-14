from __future__ import annotations

import json
from pathlib import Path

import pytest

from rmsynth_reference import CNOT, Circuit, Phase, ValidationError
from rmsynth_reference.errors import ResourceLimitError
from rmsynth_reference.io import (
    CIRCUIT_SCHEMA,
    circuit_from_data,
    circuit_to_data,
    dumps_json,
    loads_circuit,
    read_circuit,
    write_json,
)
from rmsynth_reference.limits import MAX_INPUT_BYTES


def test_circuit_json_round_trip() -> None:
    circuit = Circuit(2, (Phase(0, 1), CNOT(0, 1)))
    data = circuit_to_data(circuit)
    assert circuit_from_data(data) == circuit
    assert dumps_json(data) == dumps_json(circuit_to_data(circuit_from_data(data)))


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"schema": CIRCUIT_SCHEMA, "qubits": 1, "operations": [], "extra": 1},
        {"schema": "future", "qubits": 1, "operations": []},
        {"schema": CIRCUIT_SCHEMA, "qubits": True, "operations": []},
        {
            "schema": CIRCUIT_SCHEMA,
            "qubits": 1,
            "operations": [{"gate": "phase", "qubit": 0, "exponent": 1, "extra": 0}],
        },
    ],
)
def test_strict_input(data: object) -> None:
    with pytest.raises(ValidationError):
        circuit_from_data(data)


def test_invalid_and_oversized_json() -> None:
    with pytest.raises(ValidationError, match="invalid JSON"):
        loads_circuit("{")
    with pytest.raises(ResourceLimitError, match="exceeds"):
        loads_circuit(b" " * (MAX_INPUT_BYTES + 1))
    with pytest.raises(ValidationError, match="encoding"):
        loads_circuit(b"\xff")


def test_unknown_gate_and_non_array_operations() -> None:
    with pytest.raises(ValidationError, match="array"):
        circuit_from_data({"schema": CIRCUIT_SCHEMA, "qubits": 1, "operations": {}})
    with pytest.raises(ValidationError, match="gate object"):
        circuit_from_data({"schema": CIRCUIT_SCHEMA, "qubits": 1, "operations": [1]})
    with pytest.raises(ValidationError, match="unknown gate"):
        circuit_from_data(
            {
                "schema": CIRCUIT_SCHEMA,
                "qubits": 1,
                "operations": [{"gate": "x", "qubit": 0}],
            }
        )


def test_atomic_write_and_overwrite_refusal(tmp_path: Path) -> None:
    destination = tmp_path / "circuit.json"
    data = circuit_to_data(Circuit(1))
    write_json(destination, data)
    assert read_circuit(destination) == Circuit(1)
    before = destination.read_bytes()
    with pytest.raises(ValidationError, match="already exists"):
        write_json(destination, {"different": True})
    assert destination.read_bytes() == before
    write_json(destination, data, force=True)
    assert not list(tmp_path.glob(".circuit.json.*"))


def test_json_is_canonical() -> None:
    encoded = dumps_json({"z": 1, "a": 2})
    assert encoded == '{\n  "a": 2,\n  "z": 1\n}\n'
    assert json.loads(encoded) == {"a": 2, "z": 1}
