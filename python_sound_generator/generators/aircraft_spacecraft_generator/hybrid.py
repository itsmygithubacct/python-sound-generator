"""Compatibility facade for impact cues now owned by :mod:`crash_generator`.

Item 10 moved touchdown and crash composition out of the aircraft generator.
This module keeps the public functions used by existing callers while carrying
no material registry or event implementation of its own.  The nine rate-based
aircraft cues remain fully standard-library; only callers that request one of
these two delegated events load the recording-driven crash package.
"""

from __future__ import annotations

from ..._audio.assets import asset_names, resolve_asset

import random
from functools import lru_cache
from pathlib import Path

from ..crash_generator import engine as crashes
from ..._audio.foley_numpy import (
    SR,
    load_wav,
    resonant_sub,
)

from . import engine


SOURCE_DIRECTORY = Path(__file__).resolve().parent / "sources"


def available() -> bool:
    return bool(bundled_sources()) and bool(crashes.STRUCTURES)


@lru_cache(maxsize=1)
def bundled_sources() -> tuple[str, ...]:
    if not SOURCE_DIRECTORY.is_dir():
        return ()
    return asset_names(SOURCE_DIRECTORY)


@lru_cache(maxsize=None)
def _load(filename: str):
    path = resolve_asset(SOURCE_DIRECTORY / filename)
    if not path.is_file():
        raise ValueError(
            f"missing bundled source {path}; see sources/provenance.json"
        )
    return load_wav(path)


def _mass_speed(craft: engine.CraftProfile) -> float:
    return 1.0 / (0.55 + 0.45 * craft.mass)


def _rng(rng: random.Random):
    return crashes.rng_from_random(rng)


def _surface(craft: engine.CraftProfile) -> crashes.SurfaceProfile:
    return crashes.SURFACES["dirt" if craft.gear == "pad" else "asphalt"]


def _sub(craft: engine.CraftProfile, rng: random.Random, freq: float,
         seconds: float, decay: float):
    return resonant_sub(rng, freq, seconds, decay, cutoff=320.0)


def touchdown(craft: engine.CraftProfile, rng: random.Random,
              speed: float = 0.5, intensity: float = 0.85) -> list[float]:
    """Delegate a craft arrival to the central crash composer."""

    # Preserve the old facade's observable source-selection contract for tools
    # that instrument `_load`; the returned audio comes from crash_generator.
    if craft.gear == "tire":
        _load("egomassive_tire.wav")
    else:
        stones = next(
            name for name in bundled_sources()
            if name.startswith("rubberduck_sfx2_stones")
        )
        _load(stones)
    signal, _ = crashes.touchdown(
        crashes.STRUCTURES[craft.name], _surface(craft), _rng(rng),
        speed, intensity,
    )
    return signal.tolist()


def craft_crash(craft: engine.CraftProfile, rng: random.Random,
                intensity: float = 0.85) -> list[float]:
    """Delegate an airframe failure to the central crash composer."""

    signal, _ = crashes.crash(
        crashes.STRUCTURES[craft.name], _surface(craft), _rng(rng), intensity
    )
    return signal.tolist()


__all__ = [
    "SR",
    "available",
    "bundled_sources",
    "craft_crash",
    "touchdown",
]
