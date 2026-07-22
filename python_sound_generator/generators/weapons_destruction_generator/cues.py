"""Cue and pack registries for the weapons/destruction generator."""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from ..._audio.models import VariationFilenames

import random
from dataclasses import dataclass
from typing import Callable

from ..._audio import dsp

from . import engine


@dataclass(frozen=True)
class Options:
    """Every supported render axis; cues ignore fields they do not honour."""

    weapon: engine.FirearmProfile = engine.FIREARMS["rifle"]
    projectile: engine.ProjectileProfile = engine.PROJECTILES["bullet"]
    explosive: engine.ExplosiveProfile = engine.EXPLOSIVES["grenade"]
    material: engine.ImpactMaterial = engine.MATERIALS["metal"]
    shots: int = 4
    intensity: float = 0.85


@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Callable[[random.Random, Options], list[float]]
    peak_db: float
    honours: Honours | str | tuple[str, ...] = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


CUES: dict[str, Cue] = {
    "fire": Cue(
        "fire", "weapons", "one conventional firearm, cannon, or artillery shot",
        lambda rng, opts: engine.fire_weapon(opts.weapon, rng, opts.intensity),
        -1.0, "weapon, intensity",
    ),
    "burst": Cue(
        "burst", "weapons", "a cadence-correct sequence of conventional shots",
        lambda rng, opts: engine.fire_burst(
            opts.weapon, rng, opts.shots, opts.intensity
        ),
        -1.0, "weapon, shots, intensity",
    ),
    "laser": Cue(
        "laser", "weapons", "fast descending coherent-energy report",
        lambda rng, opts: engine.fire_laser(rng, opts.intensity),
        -3.0, "intensity",
    ),
    "plasma": Cue(
        "plasma", "weapons", "charged unstable-energy discharge",
        lambda rng, opts: engine.fire_plasma(rng, opts.intensity),
        -2.0, "intensity",
    ),
    "launch": Cue(
        "launch", "projectiles", "projectile separation, ignition, and propulsion",
        lambda rng, opts: engine.launch_projectile(
            opts.projectile, rng, opts.intensity
        ),
        -2.0, "projectile, intensity",
    ),
    "flyby": Cue(
        "flyby", "projectiles", "close projectile pass with a Doppler descent",
        lambda rng, opts: engine.projectile_flyby(
            opts.projectile, rng, opts.intensity
        ),
        -6.0, "projectile, intensity",
    ),
    "ricochet": Cue(
        "ricochet", "projectiles", "contact, deflection ping, and receding tail",
        lambda rng, opts: engine.ricochet(
            opts.projectile, opts.material, rng, opts.intensity
        ),
        -4.0, "projectile, material, intensity",
    ),
    "explosion": Cue(
        "explosion", "destruction", "shock, pressure body, rumble, and fragments",
        lambda rng, opts: engine.explosion(
            opts.explosive, rng, opts.intensity
        ),
        -0.5, "explosive, intensity",
    ),
}


# `impacts` is gone as a group with the cue that filled it. `debris` was the
# other half of that family and left with it; `explosion` keeps its own
# fragments, which are part of the blast rather than a material's answer to
# being hit.
GROUPS = ("weapons", "projectiles", "destruction")


def render(cue_name: str, options: Options, rng: random.Random) -> list[float]:
    """Render and master a named cue."""

    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    return dsp.master(cue.render(rng, options), cue.peak_db)


def stem_for(cue_name: str, options: Options) -> str:
    """Build a descriptive default artifact name from honoured options."""

    if cue_name in ("fire", "burst"):
        return f"{cue_name}_{options.weapon.name}"
    if cue_name in ("launch", "flyby"):
        return f"{cue_name}_{options.projectile.name}"
    if cue_name == "ricochet":
        return f"ricochet_{options.projectile.name}_{options.material.name}"
    if cue_name == "explosion":
        return f"explosion_{options.explosive.name}"
    return cue_name


@dataclass(frozen=True)
class PackEntry:
    filename: str
    cue: str
    count: int
    options: Options


@dataclass(frozen=True)
class Pack(VariationFilenames):
    name: str
    description: str
    entries: tuple[PackEntry, ...]



DEFAULTS = Options()

SHOWCASE = Pack(
    "showcase",
    "one representative render of every cue",
    (
        PackEntry("fire_rifle", "fire", 1, DEFAULTS),
        PackEntry("burst_rifle", "burst", 1, DEFAULTS),
        PackEntry("laser", "laser", 1, DEFAULTS),
        PackEntry("plasma", "plasma", 1, DEFAULTS),
        PackEntry(
            "launch_rocket", "launch", 1,
            Options(projectile=engine.PROJECTILES["rocket"]),
        ),
        PackEntry(
            "flyby_shell", "flyby", 1,
            Options(projectile=engine.PROJECTILES["shell"]),
        ),
        PackEntry("ricochet_bullet_metal", "ricochet", 1, DEFAULTS),
        PackEntry(
            "explosion_artillery", "explosion", 1,
            Options(explosive=engine.EXPLOSIVES["artillery"]),
        ),
    ),
)


ARSENAL = Pack(
    "arsenal",
    "variation bank covering conventional and energy weapons",
    tuple(
        PackEntry(
            f"fire_{name}", "fire", 2 if name in ("cannon", "artillery") else 3,
            Options(weapon=profile),
        )
        for name, profile in engine.FIREARMS.items()
    )
    + (
        PackEntry(
            "burst_smg", "burst", 3,
            Options(weapon=engine.FIREARMS["smg"], shots=5),
        ),
        PackEntry(
            "burst_machine_gun", "burst", 3,
            Options(weapon=engine.FIREARMS["machine_gun"], shots=6),
        ),
        PackEntry("laser", "laser", 3, DEFAULTS),
        PackEntry("plasma", "plasma", 3, DEFAULTS),
    ),
)


DESTRUCTION = Pack(
    "destruction",
    "launches, flybys, ricochets, and every explosion scale",
    tuple(
        PackEntry(
            f"launch_{name}", "launch", 2,
            Options(projectile=profile),
        )
        for name, profile in engine.PROJECTILES.items()
        if name in ("shell", "rocket", "missile", "grenade", "bomb")
    )
    + tuple(
        PackEntry(
            f"flyby_{name}", "flyby", 2,
            Options(projectile=profile),
        )
        for name, profile in engine.PROJECTILES.items()
    )
    + tuple(
        PackEntry(
            f"explosion_{name}", "explosion", 2,
            Options(explosive=profile),
        )
        for name, profile in engine.EXPLOSIVES.items()
    )
    + tuple(
        # Ricochet is the only surviving cue that reads a material, so it is
        # the only place the material matrix still shows up in a pack.
        PackEntry(
            f"ricochet_bullet_{name}", "ricochet", 2,
            Options(material=material),
        )
        for name, material in engine.MATERIALS.items()
    ),
)


PACKS = {pack.name: pack for pack in (SHOWCASE, ARSENAL, DESTRUCTION)}
