"""Optional CC0 hardware contact for the procedural cyberpunk library."""

from __future__ import annotations

import random
from typing import Sequence

from ..._audio import dsp
from ..._audio.foley_bank import (
    FoleySource,
    inventory,
    sources,
    take,
    unresolved,
)
from .engine import Style


KEY = sources("computer_sounds_generator", "key_cc0.wav")

SERVO = sources(
    "cyberpunk_sound_generator",
    *(f"derjuli_servo_impulse_{index:02d}.wav" for index in range(1, 4)),
)

POOLS: dict[str, tuple[FoleySource, ...]] = {
    "key": KEY,
    "servo": SERVO,
}


_STYLE_SPEED = {"electric": 1.18, "noir": 0.88, "corporate": 1.05}
_STYLE_HIGH = {"electric": 13_500.0, "noir": 5_200.0, "corporate": 9_000.0}
_STYLE_GAIN = {"electric": 0.82, "noir": 1.08, "corporate": 0.72}


def source_inventory() -> tuple[FoleySource, ...]:
    return inventory(POOLS)


def missing_sources() -> tuple[str, ...]:
    return unresolved(source_inventory())


def apply(
    cue_name: str,
    output: list[float],
    style: Style,
    intensity: float,
    mix: float,
    rng: random.Random,
) -> tuple[list[float], tuple[str, ...]]:
    if mix <= 0.0:
        return output, ()

    result = list(output)
    used: list[str] = []

    def add(
        onset: float,
        pool: Sequence[FoleySource],
        gain: float,
        *,
        speed: float,
        low_hz: float,
        high_hz: float,
        decay_seconds: float,
        maximum_seconds: float,
    ) -> None:
        signal, source = take(
            pool,
            rng,
            speed=speed * _STYLE_SPEED[style.name],
            low_hz=low_hz,
            high_hz=min(high_hz, _STYLE_HIGH[style.name]),
            decay_seconds=decay_seconds,
            maximum_seconds=maximum_seconds,
        )
        dsp.add_at(
            result,
            onset,
            signal.tolist(),
            gain * mix * _STYLE_GAIN[style.name],
        )
        used.append(source)

    if cue_name == "terminal_key":
        add(
            0.0,
            KEY,
            0.22 + 0.06 * intensity,
            speed=1.05,
            low_hz=120.0,
            high_hz=11_000.0,
            decay_seconds=0.055,
            maximum_seconds=0.115,
        )
    elif cue_name == "implant_activate":
        add(
            0.0,
            SERVO,
            0.26 + 0.09 * intensity,
            speed=1.0,
            low_hz=70.0,
            high_hz=9_500.0,
            decay_seconds=0.20,
            maximum_seconds=0.40,
        )

    return result, tuple(dict.fromkeys(used))


__all__ = [
    "KEY",
    "POOLS",
    "SERVO",
    "apply",
    "missing_sources",
    "source_inventory",
]
