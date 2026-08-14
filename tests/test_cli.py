from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import rmsynth_reference.cli as cli_module
from rmsynth_reference import CNOT, Circuit, Phase
from rmsynth_reference.errors import VerificationError
from rmsynth_reference.io import circuit_to_data, dumps_json, write_json


def run_cli(*arguments: object) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": "src"}
    return subprocess.run(
        [sys.executable, "-m", "rmsynth_reference.cli", *(str(value) for value in arguments)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def save(path: Path, circuit: Circuit) -> None:
    write_json(path, circuit_to_data(circuit))


def test_help_and_version() -> None:
    assert run_cli("--help").returncode == 0
    version = run_cli("--version")
    assert version.returncode == 0
    assert version.stdout.strip() == "rmsynth-ref 0.1.0rc1"


def test_inspect(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    save(source, Circuit(2, (Phase(0, 1), CNOT(0, 1))))
    completed = run_cli("inspect", source)
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["qubits"] == 2
    assert result["operations"] == 2
    assert result["t_count"] == 1


def test_inspect_from_standard_input() -> None:
    environment = {**os.environ, "PYTHONPATH": "src"}
    completed = subprocess.run(
        [sys.executable, "-m", "rmsynth_reference.cli", "inspect", "-"],
        env=environment,
        input=dumps_json(circuit_to_data(Circuit(1))),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["qubits"] == 1


def test_optimize(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    report = tmp_path / "report.json"
    save(source, Circuit(1, (Phase(0, 1), Phase(0, 1))))
    completed = run_cli("optimize", source, "--output", output, "--report", report)
    assert completed.returncode == 0
    assert json.loads(report.read_text())["after_t_count"] == 0
    assert output.exists()


def test_overwrite_refusal(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    report = tmp_path / "report.json"
    save(source, Circuit(1))
    output.write_text("keep", encoding="utf-8")
    completed = run_cli("optimize", source, "--output", output, "--report", report)
    assert completed.returncode == 2
    assert output.read_text(encoding="utf-8") == "keep"
    assert not report.exists()


def test_invalid_json_has_no_traceback(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text("{", encoding="utf-8")
    completed = run_cli("inspect", source)
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert completed.stderr.startswith("error: invalid JSON")


def test_optimizer_limit_exit_code(tmp_path: Path) -> None:
    source = tmp_path / "six.json"
    save(source, Circuit(6))
    completed = run_cli(
        "optimize",
        source,
        "--output",
        tmp_path / "out.json",
        "--report",
        tmp_path / "report.json",
    )
    assert completed.returncode == 3
    assert "at most 5 qubits" in completed.stderr


def test_verify_exit_codes(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    phase = tmp_path / "phase.json"
    save(empty, Circuit(1))
    save(phase, Circuit(1, (Phase(0, 1),)))
    equivalent = run_cli("verify", empty, empty)
    assert equivalent.returncode == 0
    assert json.loads(equivalent.stdout) == {"equivalent": True}
    inequivalent = run_cli("verify", empty, phase)
    assert inequivalent.returncode == 4
    assert json.loads(inequivalent.stdout)["witness"]["input_state"] == 1


def test_main_paths_directly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty.json"
    phase = tmp_path / "phase.json"
    output = tmp_path / "output.json"
    report = tmp_path / "report.json"
    save(empty, Circuit(1))
    save(phase, Circuit(1, (Phase(0, 1), Phase(0, 1))))

    assert cli_module.main(["inspect", str(empty)]) == 0
    assert json.loads(capsys.readouterr().out)["qubits"] == 1
    assert (
        cli_module.main(["optimize", str(phase), "--output", str(output), "--report", str(report)])
        == 0
    )
    assert json.loads(report.read_text())["status"] == "improved"
    assert cli_module.main(["verify", str(empty), str(phase)]) == 4
    assert json.loads(capsys.readouterr().out)["equivalent"] is False


def test_main_error_paths_directly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    six = tmp_path / "six.json"
    save(six, Circuit(6))
    assert (
        cli_module.main(
            [
                "optimize",
                str(six),
                "--output",
                str(tmp_path / "out.json"),
                "--report",
                str(tmp_path / "report.json"),
            ]
        )
        == 3
    )
    assert "at most 5" in capsys.readouterr().err

    assert cli_module.main(["inspect", str(tmp_path / "missing.json")]) == 2
    assert "No such file" in capsys.readouterr().err

    def fail(_: Circuit) -> None:
        raise VerificationError("failed")

    monkeypatch.setattr(cli_module, "optimize", fail)
    assert (
        cli_module.main(
            [
                "optimize",
                str(six),
                "--output",
                str(tmp_path / "other.json"),
                "--report",
                str(tmp_path / "other-report.json"),
            ]
        )
        == 4
    )
    assert "failed" in capsys.readouterr().err


def test_output_and_report_must_differ(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "same.json"
    save(source, Circuit(1))
    assert (
        cli_module.main(
            [
                "optimize",
                str(source),
                "--output",
                str(destination),
                "--report",
                str(destination),
            ]
        )
        == 2
    )


def test_report_cannot_replace_input(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    save(source, Circuit(1))
    assert (
        cli_module.main(
            [
                "optimize",
                str(source),
                "--output",
                str(tmp_path / "output.json"),
                "--report",
                str(source),
                "--force",
            ]
        )
        == 2
    )
