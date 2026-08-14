from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_hash_seed_does_not_change_output() -> None:
    code = """
from rmsynth_reference import Circuit, Phase
from rmsynth_reference.optimizer import optimize
from rmsynth_reference.io import dumps_json
print(dumps_json(optimize(Circuit(1, (Phase(0, 1), Phase(0, 1)))).report.to_dict()), end='')
"""
    outputs = []
    for seed in ("1", "91", "random"):
        environment = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"}
        completed = subprocess.run(
            [sys.executable, "-c", code],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(completed.stdout)
    assert len(set(outputs)) == 1


def test_import_does_not_write_files(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(Path.cwd() / "src"),
    }
    subprocess.run(
        [sys.executable, "-c", "import rmsynth_reference"],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    assert list(tmp_path.iterdir()) == []
