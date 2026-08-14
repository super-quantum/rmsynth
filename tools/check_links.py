from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

LINK = re.compile(r"\[[^]]*\]\(([^)]+)\)")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    for document in (*root.glob("*.md"), *(root / "docs").glob("*.md")):
        content = document.read_text(encoding="utf-8")
        for target in LINK.findall(content):
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative = unquote(target.split("#", 1)[0])
            if relative and not (document.parent / relative).resolve().exists():
                errors.append(f"broken link in {document.relative_to(root)}: {target}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("local documentation links are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
