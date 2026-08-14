from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("examples/all_parities.py", "T-count: 15 -> 0"),
        ("examples/preserve_terminal_map.py", "unchanged"),
    ],
)
def test_example(path: str, expected: str) -> None:
    completed = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == expected
