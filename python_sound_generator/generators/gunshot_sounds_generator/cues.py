"""Cue and pack registries for firearm context and mechanism Foley."""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from ..._audio.models import VariationFilenames

from ..._audio.models import RenderResult as Render

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import engine


@dataclass(frozen=True)
class Options:
    """All supported axes; each cue declares the subset it honours."""

    weapon: engine.FirearmProfile = engine.FIREARMS["rifle"]
    space: engine.SpaceProfile = engine.SPACES["outdoor"]
    suppression: engine.SuppressionProfile = engine.SUPPRESSIONS["unsuppressed"]
    surface: engine.SurfaceProfile = engine.SURFACES["concrete"]
    shots: int = 5
    intensity: float = 0.85




@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Callable[[random.Random, Options], tuple[np.ndarray, list[str]]]
    peak_db: float
    honours: Honours | str | tuple[str, ...] = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


CUES: dict[str, Cue] = {
    "shot": Cue(
        "shot", "gunshot",
        "report, acoustic space, action cycle, ejection, and casing landing",
        lambda rng, opts: engine.shot(
            opts.weapon, opts.space, opts.suppression, opts.surface, rng,
            opts.intensity,
        ),
        -1.0, "weapon, space, suppression, surface, intensity",
    ),
    "burst": Cue(
        "burst", "gunshot", "complete shots at the weapon's own cadence",
        lambda rng, opts: engine.burst(
            opts.weapon, opts.space, opts.suppression, opts.surface, rng,
            opts.shots, opts.intensity,
        ),
        -1.0, "weapon, space, suppression, surface, shots, intensity",
    ),
    "tail": Cue(
        "tail", "gunshot", "reflected field without the direct report",
        lambda rng, opts: engine.tail(
            opts.weapon, opts.space, opts.suppression, rng, opts.intensity,
        ),
        -8.0, "weapon, space, suppression, intensity",
    ),
    "bolt": Cue(
        "bolt", "mechanism", "bolt or slide cycling back and returning home",
        lambda rng, opts: engine.bolt(
            rng, opts.intensity, heavy=opts.weapon.category == "ordnance",
            recorded=opts.weapon.name == "rifle",
        ),
        -8.0, "weapon, intensity",
    ),
    "casing_eject": Cue(
        "casing_eject", "mechanism", "empty casing striking the action on exit",
        lambda rng, opts: engine.casing_eject(rng, opts.intensity),
        -11.0, "intensity",
    ),
    "casing_land": Cue(
        "casing_land", "mechanism", "empty casing landing, bouncing, and settling",
        lambda rng, opts: engine.casing_land(
            opts.surface, rng, opts.intensity
        ),
        -10.0, "surface, intensity",
    ),
    "magazine_in": Cue(
        "magazine_in", "mechanism", "magazine body seating and catch engaging",
        lambda rng, opts: engine.magazine_in(rng, opts.intensity),
        -8.5, "intensity",
    ),
    "magazine_out": Cue(
        "magazine_out", "mechanism", "magazine catch releasing and body clearing",
        lambda rng, opts: engine.magazine_out(rng, opts.intensity),
        -9.5, "intensity",
    ),
    "charging_handle": Cue(
        "charging_handle", "mechanism", "handle drawn back and action released",
        lambda rng, opts: engine.charging_handle(
            rng, opts.intensity, heavy=opts.weapon.category == "ordnance"
        ),
        -8.0, "weapon, intensity",
    ),
    "safety": Cue(
        "safety", "mechanism", "selector or safety detent moving between stops",
        lambda rng, opts: engine.safety(rng, opts.intensity),
        -12.0, "intensity",
    ),
    "trigger": Cue(
        "trigger", "mechanism", "trigger linkage click without the striker",
        lambda rng, opts: engine.trigger(rng, opts.intensity),
        -14.0, "intensity",
    ),
    "dry_fire": Cue(
        "dry_fire", "mechanism", "trigger and striker on an empty chamber",
        lambda rng, opts: engine.dry_fire(rng, opts.intensity),
        -10.5, "intensity",
    ),
    "reload": Cue(
        "reload", "sequence",
        "weapon-specific magazine, shell, cylinder, or breech sequence",
        lambda rng, opts: engine.reload(opts.weapon, rng, opts.intensity),
        -5.0, "weapon, intensity",
    ),
}

GROUPS = ("gunshot", "mechanism", "sequence")


def render(cue_name: str, options: Options, rng: random.Random) -> Render:
    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    signal, sources = cue.render(rng, options)
    return Render(engine.master(signal, cue.peak_db), tuple(dict.fromkeys(sources)))


def stem_for(cue_name: str, options: Options) -> str:
    if cue_name in {"shot", "burst"}:
        suppressed = "_suppressed" if options.suppression.name == "suppressed" else ""
        return f"{cue_name}_{options.weapon.name}_{options.space.name}{suppressed}"
    if cue_name == "tail":
        return f"tail_{options.weapon.name}_{options.space.name}"
    if cue_name == "casing_land":
        return f"casing_land_{options.surface.name}"
    if cue_name in {"reload", "bolt", "charging_handle"}:
        return f"{cue_name}_{options.weapon.name}"
    return cue_name


@dataclass(frozen=True)
class PackEntry:
    filename: str
    cue: str
    count: int
    options: Options = Options()


@dataclass(frozen=True)
class Pack(VariationFilenames):
    name: str
    description: str
    entries: tuple[PackEntry, ...]



def _options(*, weapon: str = "rifle", space: str = "outdoor",
             suppression: str = "unsuppressed", surface: str = "concrete",
             shots: int = 5, intensity: float = 0.85) -> Options:
    return Options(
        weapon=engine.FIREARMS[weapon],
        space=engine.SPACES[space],
        suppression=engine.SUPPRESSIONS[suppression],
        surface=engine.SURFACES[surface],
        shots=shots,
        intensity=intensity,
    )


SHOWCASE = Pack(
    "showcase", "one representative render of every cue",
    tuple(PackEntry(stem_for(name, Options()), name, 1) for name in CUES),
)


MECHANISMS = Pack(
    "mechanisms", "accepted fixed Foley plus variation banks for stochastic mechanisms",
    tuple(
        PackEntry(
            name, name,
            1 if name in {
                "bolt", "magazine_in", "magazine_out", "charging_handle", "reload"
            } else 3,
        )
        for name in CUES
        if CUES[name].group in {"mechanism", "sequence"}
    )
    + tuple(
        PackEntry(f"casing_land_{surface}", "casing_land", 2,
                  _options(surface=surface))
        for surface in engine.SURFACES
        if surface != "concrete"
    ),
)


CONTEXTS = Pack(
    "contexts", "same rifle report across spaces and suppression balances",
    tuple(
        PackEntry(
            f"shot_rifle_{space}{'_suppressed' if suppression == 'suppressed' else ''}",
            "shot", 1, _options(space=space, suppression=suppression),
        )
        for space in engine.SPACES
        for suppression in engine.SUPPRESSIONS
    ),
)


ARSENAL = Pack(
    "arsenal", "complete shots and reloads for every conventional firearm",
    tuple(
        PackEntry(f"shot_{weapon}", "shot", 2, _options(weapon=weapon))
        for weapon in engine.FIREARMS
    )
    + tuple(
        PackEntry(f"reload_{weapon}", "reload", 1, _options(weapon=weapon))
        for weapon in engine.FIREARMS
    )
    + (
        PackEntry("burst_smg", "burst", 2, _options(weapon="smg", shots=5)),
        PackEntry("burst_machine_gun", "burst", 2,
                  _options(weapon="machine_gun", shots=6)),
    ),
)


PACKS = {pack.name: pack for pack in (SHOWCASE, MECHANISMS, CONTEXTS, ARSENAL)}


__all__ = [
    "CUES", "GROUPS", "PACKS", "Cue", "Options", "Pack", "PackEntry",
    "Render", "render", "stem_for",
]
