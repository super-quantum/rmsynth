from __future__ import annotations

import json
from importlib.resources import files

import jsonschema

from rmsynth_reference import Circuit
from rmsynth_reference.io import circuit_to_data
from rmsynth_reference.optimizer import optimize


def schema(name: str) -> object:
    path = files("rmsynth_reference").joinpath("schemas", name)
    return json.loads(path.read_text(encoding="utf-8"))


def test_circuit_schema_accepts_serialized_circuit() -> None:
    circuit_schema = schema("circuit-v1.schema.json")
    jsonschema.Draft202012Validator.check_schema(circuit_schema)
    jsonschema.validate(circuit_to_data(Circuit(2)), circuit_schema)


def test_report_schema_accepts_optimizer_report() -> None:
    report_schema = schema("report-v1.schema.json")
    jsonschema.Draft202012Validator.check_schema(report_schema)
    jsonschema.validate(optimize(Circuit(1)).report.to_dict(), report_schema)
