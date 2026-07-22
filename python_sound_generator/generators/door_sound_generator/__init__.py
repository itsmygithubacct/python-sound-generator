"""Sample-driven NumPy/SciPy door Foley generator."""

from .engine import (
    ACTIONS,
    MATERIALS,
    AudioBuffer,
    generate_door_sound,
    write_wav,
)

__all__ = [
    "ACTIONS",
    "MATERIALS",
    "AudioBuffer",
    "generate_door_sound",
    "write_wav",
]
