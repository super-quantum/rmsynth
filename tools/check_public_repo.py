from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "tools/public_manifest.toml").read_text(encoding="utf-8"))
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [Path(line) for line in completed.stdout.splitlines() if line]
    errors: list[str] = []
    allowed = set(config["allowed_top_level"])
    allowed_cpp = set(config["allowed_cpp"])
    allowed_source = set(config["allowed_source"])
    suffixes = tuple(config["forbidden_suffixes"])
    local_home = str(Path.home())
    checked = 0

    for path in paths:
        absolute = root / path
        if not absolute.is_file():
            continue
        checked += 1
        if path.parts[0] not in allowed:
            errors.append(f"unexpected top-level path: {path}")
        if path.parts[0] == "src" and path.as_posix() not in allowed_source:
            errors.append(f"unexpected source path: {path}")
        if path.parts[0] == "cpp" and path.as_posix() not in allowed_cpp:
            errors.append(f"unexpected C++ path: {path}")
        if path.name.endswith(suffixes):
            errors.append(f"forbidden artifact: {path}")
        if absolute.is_file() and absolute.suffix in {
            ".cff",
            ".cpp",
            ".hpp",
            ".json",
            ".md",
            ".py",
            ".toml",
            ".yml",
        }:
            content = absolute.read_text(encoding="utf-8")
            if local_home in content:
                errors.append(f"local home path found in {path}")
        if path.name == "CMakeLists.txt" and local_home in absolute.read_text(encoding="utf-8"):
            errors.append(f"local home path found in {path}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"checked {checked} repository files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
