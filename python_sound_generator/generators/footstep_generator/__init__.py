"""Sample-free procedural footstep synthesis with NumPy and SciPy."""

from .footstep_generator import (
    AudioBuffer,
    FOOTWEAR,
    SURFACES,
    generate_footsteps,
    write_wav,
)

__all__ = [
    "AudioBuffer",
    "FOOTWEAR",
    "SURFACES",
    "generate_footsteps",
    "write_wav",
]
