from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .errors import ResourceLimitError, ValidationError
from .limits import MAX_INPUT_BYTES
from .model import CNOT, Circuit, Operation, Phase

CIRCUIT_SCHEMA = "rmsynth-reference/circuit-v1"


def circuit_to_data(circuit: Circuit) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    for operation in circuit.operations:
        if isinstance(operation, CNOT):
            operations.append(
                {"control": operation.control, "gate": "cnot", "target": operation.target}
            )
        else:
            operations.append(
                {"exponent": operation.exponent, "gate": "phase", "qubit": operation.qubit}
            )
    return {"operations": operations, "qubits": circuit.qubits, "schema": CIRCUIT_SCHEMA}


def circuit_from_data(data: object) -> Circuit:
    if not isinstance(data, dict):
        raise ValidationError("circuit JSON must be an object")
    if set(data) != {"schema", "qubits", "operations"}:
        raise ValidationError("circuit object must contain only schema, qubits, and operations")
    if data["schema"] != CIRCUIT_SCHEMA:
        raise ValidationError(f"schema must be {CIRCUIT_SCHEMA!r}")
    raw_operations = data["operations"]
    if not isinstance(raw_operations, list):
        raise ValidationError("operations must be an array")
    operations: list[Operation] = []
    for index, raw in enumerate(raw_operations):
        if not isinstance(raw, dict) or type(raw.get("gate")) is not str:
            raise ValidationError(f"operation {index} must be a gate object")
        if raw["gate"] == "cnot" and set(raw) == {"gate", "control", "target"}:
            operations.append(CNOT(raw["control"], raw["target"]))
        elif raw["gate"] == "phase" and set(raw) == {"gate", "qubit", "exponent"}:
            operations.append(Phase(raw["qubit"], raw["exponent"]))
        else:
            raise ValidationError(f"operation {index} has an unknown gate or invalid fields")
    return Circuit(data["qubits"], tuple(operations))


def loads_circuit(content: str | bytes) -> Circuit:
    size = len(content.encode() if isinstance(content, str) else content)
    if size > MAX_INPUT_BYTES:
        raise ResourceLimitError(f"input exceeds the {MAX_INPUT_BYTES}-byte limit")
    try:
        return circuit_from_data(json.loads(content))
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid JSON: {error.msg}") from error
    except UnicodeDecodeError as error:
        raise ValidationError(f"invalid JSON encoding: {error.reason}") from error


def read_circuit(path: str | Path) -> Circuit:
    if str(path) == "-":
        content = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    else:
        with Path(path).open("rb") as stream:
            content = stream.read(MAX_INPUT_BYTES + 1)
    return loads_circuit(content)


def dumps_json(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def write_json(path: str | Path, data: object, *, force: bool = False) -> None:
    destination = Path(path)
    if destination.exists() and not force:
        raise ValidationError(f"output already exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(dumps_json(data))
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise ValidationError(f"output already exists: {destination}") from error
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
