"""Cue and exact-filename pack registry for power-tool Foley."""

from __future__ import annotations

from ..._audio.models import VariationFilenames

from ..._audio.models import RenderResult as Render

import random
from dataclasses import dataclass

from . import engine


@dataclass(frozen=True)
class Options:
    seconds: float = 1.2
    speed: float = 1.0
    brightness: float = 0.35
    intensity: float = 0.75




@dataclass(frozen=True)
class Cue:
    name: str
    description: str


CUES = {
    "drill_loaded": Cue(
        "drill_loaded", "power drill motor and bit chatter under hard contact"
    ),
}


def render(cue_name: str, options: Options, rng: random.Random) -> Render:
    if cue_name not in CUES:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        )
    samples, sources = engine.drill_loaded(
        options.seconds, options.speed, options.brightness,
        options.intensity, rng,
    )
    return Render(samples, sources)


@dataclass(frozen=True)
class PackEntry:
    filename: str
    cue: str
    options: Options = Options()


@dataclass(frozen=True)
class Pack(VariationFilenames):
    name: str
    description: str
    entries: tuple[PackEntry, ...]


SHOWCASE = Pack(
    "showcase", "one damped loaded-drill render",
    (PackEntry("drill_loaded", "drill_loaded"),),
)
ARTILLERY_TERRAIN = Pack(
    "artillery-terrain", "exact drill filename loaded by Artillery Terrain",
    (PackEntry("drill", "drill_loaded"),),
)
PACKS = {pack.name: pack for pack in (SHOWCASE, ARTILLERY_TERRAIN)}


__all__ = [
    "CUES", "PACKS", "Cue", "Options", "Pack", "PackEntry", "Render",
    "render",
]
