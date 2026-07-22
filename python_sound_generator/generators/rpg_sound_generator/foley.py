"""Optional CC0 physical layers for the procedural RPG cue library."""

from __future__ import annotations

import random
from typing import Sequence

from ..._audio import dsp
from ..._audio.chiptune import reduce
from ..._audio.foley_bank import (
    FoleySource,
    inventory,
    sources,
    take,
    unresolved,
)
from .engine import Theme


COIN_BAG = sources(
    "rpg_sound_generator",
    *(f"dvdfu_coin_bag_{index:02d}.wav" for index in range(1, 5)),
)

COIN_FLOOR = sources(
    "rpg_sound_generator",
    *(f"dvdfu_coin_floor_{index:02d}.wav" for index in range(1, 4)),
)

CORK = sources(
    "rpg_sound_generator",
    *(f"mfearnley_cork_pop_{index:02d}.wav" for index in range(1, 4)),
)

CHEST = sources(
    "rpg_sound_generator",
    "frisbee_wooden_chest_open.wav",
)

GULP = sources("rpg_sound_generator", "andersholm_gulp.wav")

EQUIP = sources(
    "creature_foley_generator",
    "artisticdude_cloth.wav",
    "artisticdude_cloth_heavy.wav",
    "artisticdude_armor_light.wav",
    "artisticdude_chainmail.wav",
)

WOOD_DROP = sources(
    "wood_noise_generator",
    *(f"rubberduck_wood_hit_{index:02d}.wav" for index in (1, 4, 7)),
)

METAL_DROP = sources(
    "aircraft_spacecraft_generator",
    *(f"kenney_impact_metal_light_{index:03d}.wav" for index in range(5)),
)

LOCK = sources("door_sound_generator", "laleksic_lock.wav")

ITEM_DROP = COIN_FLOOR + WOOD_DROP + METAL_DROP

POOLS: dict[str, tuple[FoleySource, ...]] = {
    "coin_bag": COIN_BAG,
    "coin_floor": COIN_FLOOR,
    "cork": CORK,
    "chest": CHEST,
    "gulp": GULP,
    "equip": EQUIP,
    "wood_drop": WOOD_DROP,
    "metal_drop": METAL_DROP,
    "lock": LOCK,
}


_THEME_SPEED = {"fantasy": 1.00, "dark": 0.90, "retro": 1.18}
_THEME_HIGH = {"fantasy": 12_000.0, "dark": 5_200.0, "retro": 6_200.0}
_THEME_GAIN = {"fantasy": 1.00, "dark": 1.08, "retro": 0.72}


def source_inventory() -> tuple[FoleySource, ...]:
    return inventory(POOLS)


def missing_sources() -> tuple[str, ...]:
    return unresolved(source_inventory())


def _styled_take(
    pool: Sequence[FoleySource],
    theme: Theme,
    rng: random.Random,
    *,
    speed: float,
    low_hz: float,
    high_hz: float,
    decay_seconds: float,
    maximum_seconds: float,
) -> tuple[list[float], str]:
    signal, source = take(
        pool,
        rng,
        speed=speed * _THEME_SPEED[theme.name],
        low_hz=low_hz,
        high_hz=min(high_hz, _THEME_HIGH[theme.name]),
        decay_seconds=decay_seconds,
        maximum_seconds=maximum_seconds,
    )
    samples = signal.tolist()
    if theme.name == "retro":
        samples = reduce(samples, bits=7, hold=4)
    return samples, source


def apply(
    cue_name: str,
    output: list[float],
    theme: Theme,
    intensity: float,
    mix: float,
    rng: random.Random,
) -> tuple[list[float], tuple[str, ...]]:
    """Mix the physical layers for one supported cue and report their sources."""

    if mix <= 0.0:
        return output, ()

    result = list(output)
    used: list[str] = []

    def add(
        onset: float,
        pool: Sequence[FoleySource],
        gain: float,
        *,
        speed: float = 1.0,
        low_hz: float = 35.0,
        high_hz: float = 18_000.0,
        decay_seconds: float = 0.16,
        maximum_seconds: float = 0.40,
    ) -> None:
        signal, source = _styled_take(
            pool,
            theme,
            rng,
            speed=speed,
            low_hz=low_hz,
            high_hz=high_hz,
            decay_seconds=decay_seconds,
            maximum_seconds=maximum_seconds,
        )
        dsp.add_at(
            result,
            onset,
            signal,
            gain * mix * _THEME_GAIN[theme.name],
        )
        used.append(source)

    if cue_name == "item_drop":
        add(
            0.17,
            ITEM_DROP,
            0.32 + 0.14 * intensity,
            speed=1.22,
            low_hz=70.0,
            high_hz=7_500.0,
            decay_seconds=0.10,
            maximum_seconds=0.25,
        )
    elif cue_name == "item_equip":
        add(
            0.0,
            EQUIP,
            0.24 + 0.08 * intensity,
            speed=1.16,
            low_hz=100.0,
            high_hz=7_200.0,
            decay_seconds=0.18,
            maximum_seconds=0.38,
        )
    elif cue_name == "coin_pickup":
        add(
            0.0,
            COIN_BAG,
            0.27 + 0.08 * intensity,
            speed=1.10,
            low_hz=280.0,
            high_hz=13_000.0,
            decay_seconds=0.16,
            maximum_seconds=0.36,
        )
    elif cue_name == "purchase":
        add(
            0.0,
            COIN_BAG,
            0.20,
            speed=1.14,
            low_hz=260.0,
            high_hz=12_000.0,
            decay_seconds=0.17,
            maximum_seconds=0.32,
        )
        add(
            0.08,
            COIN_BAG,
            0.16 + 0.04 * intensity,
            speed=1.06,
            low_hz=260.0,
            high_hz=12_000.0,
            decay_seconds=0.18,
            maximum_seconds=0.34,
        )
    elif cue_name == "potion_drink":
        add(
            0.0,
            CORK,
            0.27,
            speed=1.18,
            low_hz=55.0,
            high_hz=9_000.0,
            decay_seconds=0.16,
            maximum_seconds=0.31,
        )
        add(
            0.24,
            GULP,
            0.21 + 0.07 * intensity,
            speed=1.10,
            low_hz=70.0,
            high_hz=4_800.0,
            decay_seconds=0.23,
            maximum_seconds=0.46,
        )
    elif cue_name == "chest_open":
        add(
            0.0,
            CHEST,
            0.29 + 0.08 * intensity,
            speed=1.30,
            low_hz=38.0,
            high_hz=7_200.0,
            decay_seconds=0.64,
            maximum_seconds=1.10,
        )
    elif cue_name == "door_unlock":
        add(
            0.0,
            LOCK,
            0.31,
            speed=1.04,
            low_hz=75.0,
            high_hz=8_500.0,
            decay_seconds=0.14,
            maximum_seconds=0.31,
        )

    return result, tuple(dict.fromkeys(used))


__all__ = [
    "CHEST",
    "COIN_BAG",
    "COIN_FLOOR",
    "CORK",
    "EQUIP",
    "GULP",
    "ITEM_DROP",
    "LOCK",
    "POOLS",
    "apply",
    "missing_sources",
    "source_inventory",
]
