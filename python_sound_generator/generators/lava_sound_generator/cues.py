"""Cue and pack registries for the lava generator."""

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
    """Every supported render axis; cues ignore fields they do not honour."""

    scale: engine.Scale = engine.SCALES["lake"]
    intensity: float = 0.9
    seconds: float | None = None




@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Callable[..., tuple[np.ndarray, list[str]]]
    peak_db: float
    loop: bool
    seconds: float
    honours: Honours | str | tuple[str, ...] = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


# Loops sit below the events on purpose: the bed plays under everything else, and
# a lake that arrives at the same level as the splash in front of it leaves the
# mix nowhere to go. The bed is lower still than fire's, because lava is a
# background hazard the player stands near, not a fire they are looking at.
_LOOP_PEAK = -15.0

CUES: dict[str, Cue] = {
    "molten": Cue(
        "molten", "loop", "a lava bed: the roar, heaving, with sparse bubbles",
        engine.molten, _LOOP_PEAK, True, 12.0, "scale, seconds",
    ),
    "lava_bed": Cue(
        "lava_bed", "loop", "the molten bed under everything (alias of molten)",
        engine.molten, _LOOP_PEAK, True, 12.0, "scale, seconds",
    ),
    "flow": Cue(
        "flow", "loop", "a slow surge of moving lava: a wave passing through",
        engine.flow, -14.0, True, 10.0, "scale, seconds",
    ),
    "lava": Cue(
        "lava", "event", "the game hazard: a molten splash and surge",
        engine.lava, -10.0, False, 0.0, "scale, intensity",
    ),
    "bubble": Cue(
        "bubble", "event", "a single molten bloop: one bubble surfacing",
        engine.bubble, -12.0, False, 0.0, "scale, intensity",
    ),
    "burst": Cue(
        "burst", "event", "a cluster of bubbles erupting at once",
        engine.burst, -9.0, False, 0.0, "scale, intensity",
    ),
    "crust_crack": Cue(
        "crust_crack", "event", "hot rock cracking: a bright crack over the bed",
        engine.crust_crack, -10.0, False, 0.0, "scale, intensity",
    ),
}

GROUPS = ("loop", "event")

MIN_SECONDS = 2.0
MAX_SECONDS = 60.0


def duration_for(cue_name: str, options: Options) -> float:
    cue = CUES[cue_name]
    if not cue.loop:
        return 0.0
    return cue.seconds if options.seconds is None else options.seconds


def render(cue_name: str, options: Options,
           rng: np.random.Generator) -> Render:
    """Render one cue, mastered and ready to write."""

    if cue_name not in CUES:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        )
    cue = CUES[cue_name]
    if not 0.0 < options.intensity <= 1.0:
        raise ValueError(f"intensity must be in (0, 1]; got {options.intensity}")

    if cue.loop:
        seconds = duration_for(cue_name, options)
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise ValueError(
                f"seconds must be between {MIN_SECONDS} and {MAX_SECONDS}; "
                f"got {seconds}"
            )
        signal, sources = cue.render(options.scale, seconds, rng)
    else:
        signal, sources = cue.render(options.scale, rng, options.intensity)
    return Render(engine.master(signal, cue.peak_db, cue.loop), tuple(sources))


# --------------------------------------------------------------------------
# packs
# --------------------------------------------------------------------------

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



def _options(scale: str, **overrides) -> Options:
    return Options(scale=engine.SCALES[scale], **overrides)


# Flying Joust loads sfx/lava.wav as a hazard one-shot and looks for up to eight
# _vNN variants, stopping at the first missing file; this fills exactly the four
# its src/sound.c will find -- lava.wav, lava_v02.wav, lava_v03.wav,
# lava_v04.wav -- each a different seed, so four genuinely different stretches.
# The game is never touched; the pack writes to output/.
FLYING_JOUST = Pack(
    name="flying-joust",
    description="the lava hazard flying-joust loads, four variants",
    entries=(
        PackEntry("lava", "lava", 4, _options("lake")),
    ),
)

SHOWCASE = Pack(
    name="showcase",
    description="one of everything, for auditioning",
    entries=tuple(
        PackEntry(name, name, 1, Options(seconds=6.0 if CUES[name].loop else None))
        for name in CUES
    ),
)

SCALES_PACK = Pack(
    name="scales",
    description="the molten bed at every scale, vent through eruption",
    entries=tuple(
        PackEntry(f"molten_{name}", "molten", 1,
                  _options(name, seconds=8.0))
        for name in engine.SCALES
    ),
)

PACKS = {pack.name: pack for pack in (FLYING_JOUST, SHOWCASE, SCALES_PACK)}
