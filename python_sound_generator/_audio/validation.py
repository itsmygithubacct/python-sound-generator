"""Small dependency-free validators shared by public render controls."""

from __future__ import annotations

import math
from typing import SupportsFloat


def bounded_float(
    value: SupportsFloat,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Return a finite float inside an inclusive range or raise clearly."""

    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(
            f"{name} must be between {minimum:g} and {maximum:g}; got {value}"
        )
    return parsed


__all__ = ["bounded_float"]
