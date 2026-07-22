#!/usr/bin/env python3
"""Direct CLI wrapper for the human voice generator."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .cli import build_parser, main
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from human_voice_generator.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
