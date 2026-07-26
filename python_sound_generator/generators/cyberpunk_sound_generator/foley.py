"""Physical and mechanical layers for the procedural cyberpunk library.

The cyberpunk provider owns the musical and electronic identity of its cues.
Firearms, body impacts, and weapon movement are delegated to the specialist
providers that model those events physically.  Keeping this bridge behind the
optional Foley mix preserves the dependency-free procedural path while giving
games a production path that does not turn gunshots into pitched UI sweeps.
"""

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

BODY = sources(
    "unarmed_foley_generator",
    *(f"independent_nu_hit_{index:02d}.wav" for index in (2, 7, 8, 20, 29, 34)),
)

MEDIUM_METAL = sources(
    "projectile_sounds",
    *(f"kenney_impact_metal_medium_{index:03d}.wav" for index in range(5)),
    *(f"rubberduck_metal_hit_{index:02d}.wav" for index in range(1, 6)),
)

SMALL_METAL = sources(
    "projectile_sounds",
    *(f"kenney_impact_metal_light_{index:03d}.wav" for index in range(5)),
    *(f"kenney_impact_tin_medium_{index:03d}.wav" for index in range(5)),
)

STONE = (
    sources(
        "stone_noise_generator",
        *(f"rubberduck_sfx2_stones_{index:02d}.wav" for index in range(1, 4)),
    )
    + sources(
        "stone_noise_generator",
        *(f"kenney_impact_mining_{index:03d}.wav" for index in range(5)),
    )
)

GUN_ACTION = sources(
    "gunshot_sounds_generator",
    "bolt_rifle.wav",
    "magazine_out_ar15.wav",
    "magazine_in_ar15.wav",
    "charging_handle_ar15.wav",
)

METAL = MEDIUM_METAL

POOLS: dict[str, tuple[FoleySource, ...]] = {
    "key": KEY,
    "servo": SERVO,
    "body": BODY,
    # The specialist gunshot provider consumes these approved contact pools
    # for actions, ejection, casings, and concrete-floor contact.  Listing the
    # delegated inventory here keeps `cyberpunk sources` complete.
    "medium_metal": MEDIUM_METAL,
    "small_metal": SMALL_METAL,
    "stone": STONE,
    "gun_action": GUN_ACTION,
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

    def duck(amount: float = 0.92) -> None:
        """Move the generic identity layer behind a specialist physical cue."""

        nonlocal result
        result = dsp.scale(result, max(0.06, 1.0 - amount * mix))

    def add_signal(
        onset: float,
        signal: Sequence[float],
        gain: float,
    ) -> None:
        dsp.add_at(result, onset, list(signal), gain * mix)

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

    firearm = {
        "pistol_shot": ("shot", "pistol", 1),
        "smg_burst": ("burst", "smg", 4),
        "rifle_shot": ("shot", "rifle", 1),
        "shotgun_blast": ("shot", "shotgun", 1),
    }.get(cue_name)

    if firearm is not None:
        from ..gunshot_sounds_generator import cues as gunshot
        from ..gunshot_sounds_generator import engine as gunshot_engine

        rendered = gunshot.render(
            firearm[0],
            gunshot.Options(
                weapon=gunshot_engine.FIREARMS[firearm[1]],
                space=gunshot_engine.SPACES["dry"],
                surface=gunshot_engine.SURFACES["concrete"],
                shots=firearm[2],
                intensity=intensity,
            ),
            rng,
        )
        duck(0.96)
        add_signal(0.0, rendered.samples, 1.0)
        used.extend(rendered.sources)
    elif cue_name == "reload":
        from ..gunshot_sounds_generator import cues as gunshot
        from ..gunshot_sounds_generator import engine as gunshot_engine

        rendered = gunshot.render(
            "reload",
            gunshot.Options(
                weapon=gunshot_engine.FIREARMS["pistol"],
                intensity=intensity,
            ),
            rng,
        )
        duck(0.98)
        add_signal(0.0, rendered.samples, 0.94)
        used.extend(rendered.sources)
    elif cue_name == "dry_fire":
        from ..gunshot_sounds_generator import cues as gunshot

        rendered = gunshot.render(
            "dry_fire",
            gunshot.Options(intensity=intensity),
            rng,
        )
        duck(0.98)
        add_signal(0.0, rendered.samples, 0.88)
        used.extend(rendered.sources)
    elif cue_name == "melee_swing":
        from ..combat_generators.lib import cues as combat
        from ..combat_generators.lib.weapons import MACE

        swing = combat.render(
            "swing",
            combat.Options(weapon=MACE),
            rng,
        )
        duck(0.94)
        add_signal(0.0, swing, 0.92)
        used.append("generator:combat_generators/swing")
    elif cue_name in {"melee_hit", "runner_downed"}:
        duck(0.88)
        add(
            0.0,
            BODY,
            0.62 + 0.16 * intensity,
            speed=0.78 if cue_name == "runner_downed" else 0.92,
            low_hz=40.0,
            high_hz=7_000.0,
            decay_seconds=0.18 if cue_name == "runner_downed" else 0.10,
            maximum_seconds=0.42 if cue_name == "runner_downed" else 0.30,
        )
    elif cue_name == "armor_hit":
        duck(0.90)
        add(
            0.0,
            METAL,
            0.58 + 0.14 * intensity,
            speed=0.90,
            low_hz=100.0,
            high_hz=11_000.0,
            decay_seconds=0.16,
            maximum_seconds=0.38,
        )
    elif cue_name == "ricochet":
        from ..weapons_destruction_generator import cues as weapons

        deflection = weapons.render(
            "ricochet",
            weapons.Options(intensity=intensity),
            rng,
        )
        duck(0.94)
        add_signal(0.0, deflection, 0.92)
        used.append("generator:weapons_destruction_generator/ricochet")
    elif cue_name == "terminal_key":
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
    "BODY",
    "GUN_ACTION",
    "KEY",
    "METAL",
    "MEDIUM_METAL",
    "POOLS",
    "SERVO",
    "SMALL_METAL",
    "STONE",
    "apply",
    "missing_sources",
    "source_inventory",
]
