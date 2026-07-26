"""Optional cartridge-treated CC0 Foley for the quirky RPG library."""

from __future__ import annotations

import random
from typing import Sequence

from ..._audio import dsp
from ..._audio.chiptune import reduce
from ..._audio.foley_bank import (
    PROVIDER_ROOT,
    FoleySource,
    inventory,
    sources,
    take,
    unresolved,
)
from .engine import Style


EQUIP = sources(
    "creature_foley_generator",
    "artisticdude_cloth.wav",
    "artisticdude_cloth_heavy.wav",
    "artisticdude_armor_light.wav",
    "artisticdude_chainmail.wav",
)

BODY = sources(
    "unarmed_foley_generator",
    *(f"independent_nu_hit_{index:02d}.wav" for index in (2, 7, 8, 20, 29, 34)),
)

CREATURE_ATTACK = sources(
    "creature_foley_generator",
    *(f"rubberduck_creature_attack_{index:02d}.wav" for index in (1, 2, 3, 5)),
)

CREATURE_DIE = sources(
    "creature_foley_generator",
    *(f"rubberduck_creature_die_{index:02d}.wav" for index in range(1, 5)),
)

WOOD_HIT = sources(
    "wood_noise_generator",
    *(f"rubberduck_wood_hit_{index:02d}.wav" for index in (1, 4, 7)),
)

WOOD_SCRAPE = sources(
    "wood_noise_generator",
    "lampeight_wood_panel_scrape.wav",
    "rutgermuller_wood_plank_scraping.wav",
)

LOCK = sources("door_sound_generator", "laleksic_lock.wav")

POOLS: dict[str, tuple[FoleySource, ...]] = {
    "equip": EQUIP,
    "body": BODY,
    "creature_attack": CREATURE_ATTACK,
    "creature_die": CREATURE_DIE,
    "wood_hit": WOOD_HIT,
    "wood_scrape": WOOD_SCRAPE,
    "lock": LOCK,
}


_STYLE_SPEED = {"suburban": 1.22, "psychedelic": 1.38, "lofi": 0.94}
_STYLE_HIGH = {"suburban": 5_800.0, "psychedelic": 7_200.0, "lofi": 3_200.0}
_STYLE_GAIN = {"suburban": 0.82, "psychedelic": 0.70, "lofi": 0.92}


def source_inventory() -> tuple[FoleySource, ...]:
    return inventory(POOLS)


def missing_sources() -> tuple[str, ...]:
    return unresolved(source_inventory())


def _styled_take(
    pool: Sequence[FoleySource],
    style: Style,
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
        speed=speed * _STYLE_SPEED[style.name],
        low_hz=low_hz,
        high_hz=min(high_hz, _STYLE_HIGH[style.name]),
        decay_seconds=decay_seconds,
        maximum_seconds=maximum_seconds,
    )
    samples = reduce(
        signal.tolist(),
        bits=max(4, style.bits - 2),
        hold=max(2, style.hold * 2),
    )
    return samples, source


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

    def duck(amount: float = 0.90) -> None:
        nonlocal result
        result = dsp.scale(result, max(0.10, 1.0 - amount * mix))

    def add_signal(
        onset: float,
        signal: Sequence[float],
        gain: float,
    ) -> None:
        treated = reduce(
            list(signal),
            bits=max(5, style.bits - 1),
            hold=max(2, style.hold),
        )
        dsp.add_at(result, onset, treated, gain * mix)

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
        signal, source = _styled_take(
            pool,
            style,
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
            gain * mix * _STYLE_GAIN[style.name],
        )
        used.append(source)

    if cue_name in {"attack_swing", "attack_miss"}:
        from ..combat_generators.lib import cues as combat
        from ..combat_generators.lib.weapons import MACE

        swing = combat.render(
            "swing",
            combat.Options(weapon=MACE),
            rng,
        )
        duck(0.94)
        add_signal(0.0, swing, 0.94 if cue_name == "attack_swing" else 0.80)
        add(
            0.0,
            EQUIP,
            0.12,
            speed=1.22,
            low_hz=140.0,
            high_hz=5_800.0,
            decay_seconds=0.11,
            maximum_seconds=0.24,
        )
        used.append("generator:combat_generators/swing")
    elif cue_name == "enemy_attack":
        duck(0.94)
        add(
            0.0,
            CREATURE_ATTACK,
            0.61 + 0.13 * intensity,
            speed=1.16,
            low_hz=80.0,
            high_hz=6_400.0,
            decay_seconds=0.28,
            maximum_seconds=0.58,
        )
    elif cue_name == "enemy_defeat":
        duck(0.92)
        add(
            0.0,
            CREATURE_DIE,
            0.62 + 0.12 * intensity,
            speed=1.10,
            low_hz=60.0,
            high_hz=6_200.0,
            decay_seconds=0.40,
            maximum_seconds=0.70,
        )
    elif cue_name == "party_down":
        from ..combat_generators.lib import cues as combat

        fall = combat.render(
            "body_fall",
            combat.Options(armored=False),
            rng,
        )
        duck(0.86)
        add_signal(0.0, fall, 0.68)
        add(
            0.08,
            BODY,
            0.42 + 0.10 * intensity,
            speed=0.74,
            low_hz=35.0,
            high_hz=4_800.0,
            decay_seconds=0.22,
            maximum_seconds=0.48,
        )
        used.append("generator:combat_generators/body_fall")
    elif cue_name == "equip":
        add(
            0.0,
            EQUIP,
            0.23 + 0.07 * intensity,
            speed=1.12,
            low_hz=130.0,
            high_hz=6_000.0,
            decay_seconds=0.14,
            maximum_seconds=0.32,
        )
    elif cue_name == "player_hit":
        duck(0.84)
        add(
            0.0,
            BODY,
            0.52 + 0.10 * intensity,
            speed=0.94,
            low_hz=45.0,
            high_hz=5_800.0,
            decay_seconds=0.075,
            maximum_seconds=0.25,
        )
    elif cue_name == "enemy_hit":
        duck(0.90)
        add(
            0.0,
            BODY,
            0.46 + 0.10 * intensity,
            speed=1.12,
            low_hz=70.0,
            high_hz=6_200.0,
            decay_seconds=0.09,
            maximum_seconds=0.28,
        )
        add(
            0.0,
            WOOD_HIT,
            0.34 + 0.08 * intensity,
            speed=1.18,
            low_hz=100.0,
            high_hz=5_500.0,
            decay_seconds=0.08,
            maximum_seconds=0.24,
        )
    elif cue_name == "critical_hit":
        duck(0.94)
        add(
            0.0,
            BODY,
            0.58 + 0.12 * intensity,
            speed=0.78,
            low_hz=38.0,
            high_hz=6_500.0,
            decay_seconds=0.15,
            maximum_seconds=0.39,
        )
        add(
            0.0,
            WOOD_HIT,
            0.45 + 0.10 * intensity,
            speed=0.92,
            low_hz=70.0,
            high_hz=5_200.0,
            decay_seconds=0.14,
            maximum_seconds=0.34,
        )
    elif cue_name == "door_transition":
        add(
            0.0,
            WOOD_SCRAPE,
            0.19 + 0.05 * intensity,
            speed=1.35,
            low_hz=80.0,
            high_hz=4_800.0,
            decay_seconds=0.20,
            maximum_seconds=0.36,
        )
        add(
            0.27,
            LOCK,
            0.15,
            speed=1.30,
            low_hz=110.0,
            high_hz=5_500.0,
            decay_seconds=0.07,
            maximum_seconds=0.17,
        )

    return result, tuple(dict.fromkeys(used))


__all__ = [
    "BODY",
    "CREATURE_ATTACK",
    "CREATURE_DIE",
    "EQUIP",
    "LOCK",
    "POOLS",
    "PROVIDER_ROOT",
    "WOOD_HIT",
    "WOOD_SCRAPE",
    "apply",
    "missing_sources",
    "source_inventory",
]
