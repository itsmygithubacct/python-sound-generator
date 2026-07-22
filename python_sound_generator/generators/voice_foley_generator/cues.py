"""Cue and exact-filename pack registries for human voice Foley."""

from __future__ import annotations

from ..._audio.models import VariationFilenames

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import engine


@dataclass(frozen=True)
class Options:
    voice: engine.VoiceProfile = engine.VOICES["neutral"]
    intensity: float = 0.85
    duration: float | None = None
    body: bool = True


@dataclass(frozen=True)
class Render:
    samples: list[float]
    sources: tuple[str, ...]
    layers: tuple[str, ...]


@dataclass(frozen=True)
class Cue:
    name: str
    description: str
    render: Callable[..., tuple[np.ndarray, list[str], list[str]]]
    duration: float
    peak_db: float
    honours_body: bool = False


CUES: dict[str, Cue] = {
    "pain": Cue(
        "pain", "brief nonverbal human hurt reaction with optional body hit",
        engine.pain, 0.44, -7.0, True,
    ),
    "gasp": Cue(
        "gasp", "short breathy shock or defeat intake", engine.gasp, 0.52, -10.0,
    ),
    "exertion": Cue(
        "exertion", "compact effort grunt for a jump, shove, or strike",
        engine.exertion, 0.46, -9.0,
    ),
    "death": Cue(
        "death", "human gasp, soft fall, and descending magical dissolve",
        engine.death, 1.28, -7.0,
    ),
}


def render(cue_name: str, options: Options,
           rng: np.random.Generator) -> Render:
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
    raw, sources, layers = cue.render(
        options.voice, rng, duration, options.intensity,
        body=options.body if cue.honours_body else False,
    )
    return Render(
        engine.master(raw, cue.peak_db),
        tuple(dict.fromkeys(sources)),
        tuple(dict.fromkeys(layers)),
    )


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



def _options(voice: str, **overrides) -> Options:
    return Options(voice=engine.VOICES[voice], **overrides)


FANTASY_ADVENTURE = Pack(
    "fantasy-adventure",
    "human hurt and layered defeat cues under Fantasy Adventure's player namespace",
    (
        PackEntry("player/hurt", "pain", 5, _options("masculine", body=True)),
        PackEntry("player/death", "death", 4, _options("masculine")),
    ),
)

SHOWCASE = Pack(
    "showcase",
    "every voice cue across the recorded voice profiles",
    tuple(
        PackEntry(f"{cue}_{voice}", cue, 1, _options(voice))
        for voice in ("masculine", "feminine")
        for cue in CUES
    ),
)

PACKS = {pack.name: pack for pack in (FANTASY_ADVENTURE, SHOWCASE)}
