"""Standard-library-only procedural car engine synthesis."""

from .car_engine_generator import (
    ENGINE_PROFILES,
    MODES,
    AudioBuffer,
    EngineProfile,
    generate_engine_sound,
    write_wav,
)

__all__ = [
    "AudioBuffer",
    "ENGINE_PROFILES",
    "EngineProfile",
    "MODES",
    "generate_engine_sound",
    "write_wav",
]
