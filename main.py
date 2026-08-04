"""Convenience shim so `python main.py …` keeps working without installation.

The real logic lives in the packaged CLI (`src/cli.py`), also exposed as the
`bsa-report` console script after `pip install -e .`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
