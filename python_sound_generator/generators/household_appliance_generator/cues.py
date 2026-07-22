"""Cue and pack registry for household appliance Foley."""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from ..._audio.models import VariationFilenames

from ..._audio.models import RenderResult as Render

import random
from dataclasses import dataclass

from . import engine


MIN_SECONDS = 1.0
MAX_SECONDS = 60.0


@dataclass(frozen=True)
class Options:
    """Playback controls shared by every appliance recording."""

    seconds: float | None = None
    speed: float = 1.0
    intensity: float = 0.75




@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    appliance: engine.Appliance
    loop: bool
    seconds: float
    honours: Honours | str | tuple[str, ...] = "seconds, speed, intensity"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


CUES: dict[str, Cue] = {
    name: Cue(
        name=name,
        group="loop" if appliance.loop else "mechanism",
        description=appliance.description,
        appliance=appliance,
        loop=appliance.loop,
        seconds=appliance.seconds,
    )
    for name, appliance in engine.APPLIANCES.items()
}

GROUPS = ("loop", "mechanism")
LOOPS = tuple(name for name, cue in CUES.items() if cue.loop)
ONESHOTS = tuple(name for name, cue in CUES.items() if not cue.loop)


def duration_for(cue_name: str, options: Options) -> float:
    cue = CUES[cue_name]
    return cue.seconds if options.seconds is None else options.seconds


def render(cue_name: str, options: Options, rng: random.Random) -> Render:
    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    if not 0.05 <= options.intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    if not 0.70 <= options.speed <= 1.35:
        raise ValueError("speed must be between 0.70 and 1.35")
    seconds = duration_for(cue_name, options)
    upper = MAX_SECONDS if cue.loop else 8.0
    if not MIN_SECONDS <= seconds <= upper:
        raise ValueError(
            f"seconds for {cue_name} must be between {MIN_SECONDS:g} and "
            f"{upper:g}; got {seconds}"
        )
    samples, sources = engine.render_appliance(
        cue.appliance, seconds, options.speed, options.intensity, rng
    )
    return Render(samples, sources)


@dataclass(frozen=True)
class PackEntry:
    filename: str
    cue: str
    count: int = 1
    options: Options = Options()


@dataclass(frozen=True)
class Pack(VariationFilenames):
    name: str
    description: str
    entries: tuple[PackEntry, ...]



SHOWCASE = Pack(
    "showcase",
    "one representative render of every recorded appliance",
    tuple(PackEntry(name, name) for name in CUES),
)

KITCHEN = Pack(
    "kitchen",
    "the four recorded countertop and ventilation appliances",
    (
        PackEntry("air_fryer_loop", "air_fryer"),
        PackEntry("microwave_loop", "microwave"),
        PackEntry("range_hood_loop", "range_hood"),
        PackEntry("can_opener", "can_opener", count=3),
    ),
)

WHOLE_HOUSE = Pack(
    "whole-house",
    "ready-to-loop beds for every continuous appliance plus can opener takes",
    tuple(
        PackEntry(f"{name}_loop" if cue.loop else name, name,
                  count=1 if cue.loop else 3)
        for name, cue in CUES.items()
    ),
)

PACKS = {pack.name: pack for pack in (SHOWCASE, KITCHEN, WHOLE_HOUSE)}


__all__ = [
    "CUES", "GROUPS", "LOOPS", "MAX_SECONDS", "MIN_SECONDS", "ONESHOTS",
    "PACKS", "Cue", "Options", "Pack", "PackEntry", "Render",
    "duration_for", "render",
]
