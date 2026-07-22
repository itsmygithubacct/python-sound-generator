#!/usr/bin/env python3
"""Direct CLI entry point, so the generator can be run without -m.

The implementation lives in :mod:`powerup_generator.cli`. This exists so both

    ./powerup_sfx.py render coin --style arcade
    soundgen powerups render coin --style arcade

work, matching the two conventions already in use across this repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # Package import: ``python -m powerup_generator``.
    from .cli import build_parser, main
except ImportError:  # Direct execution: ``./powerup_sfx.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from powerup_generator.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
