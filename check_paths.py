"""Report which analysis scripts still hardcode an absolute filesystem path.

Run from the repository root:

    python check_paths.py

Exit code is 0 if every script is free of absolute paths, 1 otherwise. This is
a diagnostic, not a test: the scripts are published as an analysis record and
are known to contain local paths (see README, "Known limitations").
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Windows drive letters (D:\..., C:/...) and POSIX absolute paths written as
# string literals. Deliberately conservative: matches inside comments too.
ABS_WIN = re.compile(r"\b[A-Za-z]:[\\/][^\s\"')]*")
ABS_POSIX = re.compile(r"[\"'](/(?:home|Users|mnt|data)/[^\s\"']*)[\"']")


def scan(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return hits
    for n, line in enumerate(text.splitlines(), start=1):
        for m in ABS_WIN.finditer(line):
            hits.append((n, m.group(0)))
        for m in ABS_POSIX.finditer(line):
            hits.append((n, m.group(1)))
    return hits


def main() -> int:
    root = Path(__file__).resolve().parent
    code_dir = root / "code"
    if not code_dir.is_dir():
        print(f"no code/ directory under {root}")
        return 2

    total = 0
    dirty = 0
    for script in sorted(code_dir.glob("*.py")):
        hits = scan(script)
        if not hits:
            continue
        dirty += 1
        total += len(hits)
        print(f"{script.name}  ({len(hits)} absolute path{'s' if len(hits) > 1 else ''})")
        for n, s in hits[:5]:
            print(f"    line {n}: {s}")
        if len(hits) > 5:
            print(f"    ... {len(hits) - 5} more")

    scripts = len(list(code_dir.glob("*.py")))
    print()
    print(f"{dirty} of {scripts} scripts contain {total} absolute path(s).")
    if dirty:
        print("Replace them with values from config.py before running on another machine.")
    return 0 if dirty == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
