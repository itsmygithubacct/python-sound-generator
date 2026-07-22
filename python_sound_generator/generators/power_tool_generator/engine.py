"""Power-tool Foley selection and conservative shaping."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from ..._audio.foley_numpy import (
    SR,
    darken,
    load_wav,
    master,
    varispeed,
    write_wav,
)


SOURCES = Path(__file__).resolve().parent / "sources"
DRILL_SOURCE = "drill_concrete_cc0.wav"


def drill_loaded(seconds: float, speed: float, brightness: float,
                 intensity: float, rng: random.Random
                 ) -> tuple[list[float], tuple[str, ...]]:
    """A power drill already biting hard material.

    The recording owns the motor, bit chatter, and contact. ``speed`` is honest
    varispeed; ``brightness`` only darkens the approved source, with the default
    deliberately damped after the listening pass.
    """

    if not 0.3 <= seconds <= 4.0:
        raise ValueError("seconds must be between 0.3 and 4")
    if not 0.70 <= speed <= 1.35:
        raise ValueError("speed must be between 0.70 and 1.35")
    if not 0.0 <= brightness <= 1.0:
        raise ValueError("brightness must be between 0 and 1")
    if not 0.05 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")

    source = varispeed(load_wav(SOURCES / DRILL_SOURCE), speed)
    frames = round(seconds * SR)
    if len(source) < frames:
        raise ValueError("drill Foley is shorter than the requested render")
    start = rng.randrange(0, len(source) - frames + 1)
    excerpt = np.asarray(source[start:start + frames], dtype=np.float64)
    cutoff = 3_200.0 + 4_800.0 * brightness
    shaped = darken(excerpt, cutoff)
    shaped *= 0.45 + 0.55 * intensity
    return master(shaped, -10.0), (DRILL_SOURCE,)


__all__ = ["DRILL_SOURCE", "SOURCES", "SR", "drill_loaded", "write_wav"]
