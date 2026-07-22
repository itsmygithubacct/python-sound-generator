#!/usr/bin/env python3
"""Direct CLI entry point, so the generator can be run without -m.

The implementation lives in :mod:`ui_game_state_generator.cli`. This exists so
both

    ./ui_sfx.py render menu_accept --style arcade
    soundgen ui render menu_accept --style arcade

work, matching the two conventions already in use across this repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # Package import: ``python -m ui_game_state_generator``.
    from .cli import build_parser, main
except ImportError:  # Direct execution: ``./ui_sfx.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ui_game_state_generator.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
