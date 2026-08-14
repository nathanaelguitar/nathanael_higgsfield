#!/usr/bin/env python3
"""Make the vendored CodeFormer source importable without packaging it."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/CodeFormer", file=sys.stderr)
        return 2
    version = Path(sys.argv[1]) / "basicsr" / "version.py"
    version.parent.mkdir(parents=True, exist_ok=True)
    if not version.exists():
        version.write_text('__gitsha__ = "local"\n__version__ = "0.1.0"\n', encoding="utf-8")
    print(f"Ready {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
