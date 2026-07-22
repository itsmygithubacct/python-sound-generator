"""Cue and pack registry for structure-failure event composition."""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from ..._audio.models import VariationFilenames

from ..._audio.models import RenderResult as Render

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import engine


@dataclass(frozen=True)
class Options:
    structure: engine.StructureProfile = engine.STRUCTURES["sedan"]
    surface: engine.SurfaceProfile = engine.SURFACES["asphalt"]
    speed: float = 0.5
    intensity: float = 0.85




@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    renderer: Callable[
        [engine.StructureProfile, engine.SurfaceProfile, np.random.Generator,
         float, float],
        tuple[np.ndarray, list[str]],
    ]
    peak_db: float
    honours: Honours | str | tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


def _collision(structure, surface, rng, speed, intensity):
    return engine.collision(structure, surface, rng, intensity)


def _debris(structure, surface, rng, speed, intensity):
    return engine.debris(structure, surface, rng, intensity)


def _crash(structure, surface, rng, speed, intensity):
    return engine.crash(structure, surface, rng, intensity)


def _touchdown(structure, surface, rng, speed, intensity):
    return engine.touchdown(structure, surface, rng, speed, intensity)


CUES: dict[str, Cue] = {
    "collision": Cue(
        "collision", "failure", "one structural contact and deformation",
        _collision, -1.0, "structure, surface, intensity",
    ),
    "crash": Cue(
        "crash", "failure",
        "primary collision, decelerating secondaries, and aftermath",
        _crash, -0.5, "structure, surface, intensity",
    ),
    "debris": Cue(
        "debris", "aftermath", "fragments scattering and settling",
        _debris, -5.0, "structure, surface, intensity",
    ),
    "touchdown": Cue(
        "touchdown", "arrival", "controlled contact, gear load, and settling",
        _touchdown, -1.5, "structure, surface, speed, intensity",
    ),
}

GROUPS = ("failure", "aftermath", "arrival")


def render(cue_name: str, options: Options,
           rng: np.random.Generator) -> Render:
    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    signal, sources = cue.renderer(
        options.structure, options.surface, rng, options.speed,
        options.intensity,
    )
    return Render(engine.master(signal, cue.peak_db), tuple(sources))


def stem_for(cue_name: str, options: Options) -> str:
    return f"{cue_name}_{options.structure.name}_{options.surface.name}"


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



def _at(structure: str = "sedan", surface: str = "asphalt",
        **kwargs) -> Options:
    return Options(
        structure=engine.STRUCTURES[structure],
        surface=engine.SURFACES[surface],
        **kwargs,
    )


SHOWCASE = Pack(
    "showcase", "one representative render of every event",
    (
        PackEntry("collision_sedan_asphalt", "collision", 1, _at()),
        PackEntry("crash_mixed_concrete", "crash", 1,
                  _at("mixed", "concrete", intensity=1.0)),
        PackEntry("debris_timber_stone", "debris", 1,
                  _at("timber", "stone")),
        PackEntry("touchdown_trainer_asphalt", "touchdown", 1,
                  _at("trainer", speed=0.6)),
    ),
)


ARCADE_DRIVER = Pack(
    "arcade_driver", "the collision files formerly emitted by vehicle foley",
    (
        PackEntry("traffic_impact", "collision", 3,
                  _at(intensity=0.7)),
        PackEntry("crash", "crash", 2, _at(intensity=1.0)),
        PackEntry("crash_debris", "debris", 2, _at()),
    ),
)


OFFROAD = Pack(
    "offroad", "the crash files for a buggy on loose ground",
    (
        PackEntry("crash", "crash", 2,
                  _at("buggy", "dirt", intensity=1.0)),
        PackEntry("crash_debris", "debris", 2,
                  _at("buggy", "dirt")),
    ),
)


FLIGHT = Pack(
    "flight", "trainer touchdown and crash files formerly owned by aircraft",
    (
        PackEntry("touchdown", "touchdown", 2,
                  _at("trainer", speed=0.5)),
        PackEntry("touchdown_fast", "touchdown", 2,
                  _at("trainer", speed=0.9)),
        PackEntry("crash", "crash", 2,
                  _at("trainer", intensity=1.0)),
    ),
)


LANDER = Pack(
    "lander", "footpad arrivals and hull crashes for a small lander",
    (
        PackEntry("touchdown_soft", "touchdown", 2,
                  _at("lander", "dirt", speed=0.2, intensity=0.5)),
        PackEntry("touchdown_hard", "touchdown", 2,
                  _at("lander", "dirt", speed=0.6, intensity=1.0)),
        PackEntry("crash", "crash", 2,
                  _at("lander", "dirt", intensity=1.0)),
    ),
)


LUNAR_LANDER = Pack(
    "lunar-lander", "game-named landing and crash stems",
    (
        PackEntry("landing", "touchdown", 3,
                  _at("lander", "dirt", speed=0.2, intensity=0.5)),
        PackEntry("crash", "crash", 4,
                  _at("lander", "dirt", intensity=1.0)),
    ),
)


PACKS = {
    pack.name: pack
    for pack in (SHOWCASE, ARCADE_DRIVER, OFFROAD, FLIGHT, LANDER,
                 LUNAR_LANDER)
}


__all__ = [
    "CUES",
    "GROUPS",
    "PACKS",
    "Cue",
    "Options",
    "Pack",
    "PackEntry",
    "Render",
    "render",
    "stem_for",
]
