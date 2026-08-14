from __future__ import annotations

from rmsynth_reference import CNOT, Circuit
from rmsynth_reference.optimizer import optimize
from rmsynth_reference.semantics import extract_program, program_digest


def test_digest_includes_terminal_map() -> None:
    identity = program_digest(extract_program(Circuit(2)))
    cnot = program_digest(extract_program(Circuit(2, (CNOT(0, 1),))))
    assert identity != cnot


def test_report_has_stable_public_fields() -> None:
    report = optimize(Circuit(1)).report.to_dict()
    assert report["schema"] == "rmsynth-reference/report-v1"
    assert report["tool_version"] == "0.1.0rc1"
    assert report["verified"] is True
    assert set(report["limits"]) == {
        "decoder_candidates",
        "input_bytes",
        "operations",
        "optimizer_qubits",
        "verifier_qubits",
    }
    forbidden = {"duration", "elapsed", "host", "path", "runtime", "timestamp"}
    assert forbidden.isdisjoint(report)


def test_report_distinguishes_artifacts_from_semantics() -> None:
    from rmsynth_reference import Phase

    result = optimize(Circuit(1, (Phase(0, 1), Phase(0, 1))))
    report = result.report.to_dict()
    assert report["input_digest"] != report["output_digest"]
    assert len(report["semantic_digest"]) == 64
