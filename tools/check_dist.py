from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN = (
    ".DS_Store",
    ".a",
    ".dll",
    ".dylib",
    ".o",
    ".obj",
    ".pyc",
)
NATIVE_SUFFIXES = (".pyd", ".so")


def _entries(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return archive.getnames()
    raise ValueError(f"unsupported artifact: {path}")


def _check(path: Path) -> list[str]:
    try:
        names = _entries(path)
    except ValueError as error:
        return [str(error)]
    errors = [
        f"forbidden distribution entry: {name}"
        for name in names
        if name.endswith(FORBIDDEN) or "/rmsynth/" in f"/{name}/"
    ]
    if path.suffix != ".whl":
        errors.extend(
            f"compiled file in source distribution: {name}"
            for name in names
            if name.endswith(NATIVE_SUFFIXES)
        )
        sdist_required = (
            "/CITATION.cff",
            "/CODE_OF_CONDUCT.md",
            "/LICENSE",
            "/README.md",
            "/CMakeLists.txt",
            "/cpp/include/rmsynth_reference/core.hpp",
            "/cpp/src/bindings.cpp",
            "/cpp/src/core.cpp",
            "/cpp/tests/core_tests.cpp",
            "/tests/test_optimizer.py",
        )
        errors.extend(
            f"missing source entry: {expected[1:]}"
            for expected in sdist_required
            if not any(name.endswith(expected) for name in names)
        )
        return errors
    wheel_required = (
        "rmsynth_reference/__init__.py",
        "rmsynth_reference/_native.pyi",
        "rmsynth_reference/py.typed",
        "rmsynth_reference/schemas/circuit-v1.schema.json",
        "rmsynth_reference/schemas/report-v1.schema.json",
    )
    errors.extend(
        f"missing wheel entry: {expected}"
        for expected in wheel_required
        if not any(name.endswith(expected) for name in names)
    )
    native = [name for name in names if name.endswith(NATIVE_SUFFIXES)]
    if len(native) != 1 or not native[0].startswith("rmsynth_reference/_native."):
        errors.append("wheel must contain exactly one rmsynth_reference._native extension")
    if any("/tests/" in f"/{name}/" for name in names):
        errors.append("tests must not be included in the wheel")
    return errors


def main(arguments: list[str]) -> int:
    if not arguments:
        print("usage: check_dist.py ARTIFACT...", file=sys.stderr)
        return 2
    errors = [error for argument in arguments for error in _check(Path(argument))]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"checked {len(arguments)} distribution artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
