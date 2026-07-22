#!/usr/bin/env python3
"""Compatibility wrapper and direct CLI entry point.

The synthesis implementation lives in :mod:`footstep_generator.engine`.
"""

from __future__ import annotations

try:  # Package import: ``python -m footstep_generator``.
    from .engine import (
        AudioBuffer,
        FOOTWEAR,
        SURFACES,
        build_parser,
        generate_footsteps,
        main,
        write_wav,
    )
except ImportError:  # Direct execution: ``python footstep_generator.py``.
    from engine import (  # type: ignore[no-redef]
        AudioBuffer,
        FOOTWEAR,
        SURFACES,
        build_parser,
        generate_footsteps,
        main,
        write_wav,
    )


__all__ = [
    "AudioBuffer",
    "FOOTWEAR",
    "SURFACES",
    "build_parser",
    "generate_footsteps",
    "main",
    "write_wav",
]


if __name__ == "__main__":
    raise SystemExit(main())
