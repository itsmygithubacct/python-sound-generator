"""Cue and exact-filename pack registries for fantasy energy."""

from __future__ import annotations

from ..._audio.models import VariationFilenames

from ..._audio.models import RenderResult as Render

import random
from dataclasses import dataclass
from typing import Callable

from . import engine


@dataclass(frozen=True)
class Options:
    intensity: float = 0.9
    duration: float | None = None




@dataclass(frozen=True)
class Cue:
    name: str
    description: str
    render: Callable[..., list[float]]
    duration: float
    peak_db: float


CUES: dict[str, Cue] = {
    "beam": Cue(
        "beam",
        "enchanted blade snap into a compact airy crescent",
        engine.beam,
        0.48,
        -5.0,
    ),
    "dissolve": Cue(
        "dissolve",
        "descending magical life-energy motes and breath",
        engine.dissolve,
        1.08,
        -9.0,
    ),
}


def render(cue_name: str, options: Options,
           rng: random.Random) -> Render:
    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    if not 0.05 <= options.intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    duration = cue.duration if options.duration is None else options.duration
    if not 0.12 <= duration <= 4.0:
        raise ValueError("duration must be between 0.12 and 4 seconds")
    raw = cue.render(rng, duration, options.intensity)
    return Render(engine.master(raw, cue.peak_db))


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



FANTASY_ADVENTURE = Pack(
    "fantasy-adventure",
    "fantasy sword beams under Fantasy Adventure's exact combat namespace",
    (PackEntry("combat/beam", "beam", 5),),
)

SHOWCASE = Pack(
    "showcase",
    "one representative render of every fantasy-energy cue",
    tuple(PackEntry(name, name, 1) for name in CUES),
)

PACKS = {pack.name: pack for pack in (FANTASY_ADVENTURE, SHOWCASE)}
