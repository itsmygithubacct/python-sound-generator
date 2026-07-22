"""Cue and pack registries for the stone generator."""

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
    """Every supported render axis; cues ignore fields they do not honour.

    `stone` and `scale` are separate axes rather than one combined profile
    because they are separately true: a granite pebble and a granite cliff are
    the same material at two masses, and a sandstone cliff is a different
    material at the same mass. Together they form a material x mass x severity
    matrix; `intensity` is the severity.
    """

    stone: engine.Stone = engine.STONES["limestone"]
    scale: engine.Scale = engine.SCALES["boulder"]
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


# Loops sit below the events on purpose: a bed plays under everything else, and
# a rockfall arriving at the same level as the shatter in front of it leaves the
# mix nowhere to go. The same reasoning, and the same numbers, as
# fire_sound_generator.
_LOOP_PEAK = -14.0

CUES: dict[str, Cue] = {
    "strike": Cue(
        "strike", "event", "stone struck, and it holds: a pick, a hammer, a round",
        engine.strike, -9.0, False, 0.0, "stone, scale, intensity",
    ),
    "crack": Cue(
        "crack", "event", "stone failing but holding together: a fissure",
        engine.crack, -9.0, False, 0.0, "stone, scale, intensity",
    ),
    "shatter": Cue(
        "shatter", "event", "stone coming apart: contact, fracture, debris",
        engine.shatter, -8.0, False, 0.0, "stone, scale, intensity",
    ),
    "fall": Cue(
        "fall", "event", "one stone landing: arrival, bounces, settle",
        engine.fall, -9.5, False, 0.0, "stone, scale, intensity",
    ),
    "scatter": Cue(
        "scatter", "event", "rubble arriving, with nothing in front of it",
        engine.scatter, -10.0, False, 0.0, "stone, scale, intensity",
    ),
    "grind": Cue(
        "grind", "loop", "stone dragged over stone: millstone, lid, shift",
        engine.grind, _LOOP_PEAK, True, 8.0, "stone, scale, intensity, seconds",
    ),
    "rockfall": Cue(
        "rockfall", "loop", "a slope coming down and not stopping",
        engine.rockfall, _LOOP_PEAK, True, 10.0, "stone, scale, intensity, seconds",
    ),
}

GROUPS = ("event", "loop")

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
        signal, sources = cue.render(options.stone, options.scale, seconds, rng,
                                     options.intensity)
    else:
        signal, sources = cue.render(options.stone, options.scale, rng,
                                     options.intensity)
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



def _options(stone: str, scale: str, **overrides) -> Options:
    return Options(stone=engine.STONES[stone], scale=engine.SCALES[scale],
                   **overrides)


# There are no game packs here, and their absence is deliberate rather than
# unfinished.
#
# Two reasons. The first is that nothing in this generator has been signed off by
# ear yet, and every generator in this repository that shipped a game pack
# shipped it *after* the audition, not before -- item 8's arcade-fighter pack is
# dated to the day its cues were approved. Shipping a bank ahead of the listening
# would invert the one practice this roadmap says has paid every time.
#
# The second is that the consumers are not games. Material providers are the
# foundation and event providers are the consumers: a material generator owns
# what stone *is*, and an event generator composes it. Checked
# against the games as they actually stand -- kitty-breakout ships no WAVs at all
# (nine built-in procedural cues), arena-fps-engine has one 200 ms proof PCM, and
# fantasy-adventure's assets/sfx/environment/ is four door files -- there is no stone slot
# in any of them waiting on a filename. Inventing one would be guessing at a bank
# that does not exist.
#
# The packs below are for auditioning, which is what this generator needs next.

SHOWCASE = Pack(
    name="showcase",
    description="one of every cue, at the default limestone boulder",
    entries=tuple(
        PackEntry(name, name, 1,
                  Options(seconds=6.0 if CUES[name].loop else None))
        for name in CUES
    ),
)

# The material axis, heard on the cue that shows it most. A shatter is contact,
# fracture and debris at once, so it is the one place all four Stone fields are
# audible in a single file.
STONES_PACK = Pack(
    name="stones",
    description="shatter and grind across every stone, for the material axis",
    entries=tuple(
        PackEntry(f"shatter_{name}", "shatter", 1, _options(name, "boulder"))
        for name in engine.STONES
    ) + tuple(
        PackEntry(f"grind_{name}", "grind", 1,
                  _options(name, "boulder", seconds=6.0))
        for name in engine.STONES
    ),
)

# The mass axis. strike is the cleanest read on varispeed because nothing else
# is happening in it; shatter is there because debris scale is the part most
# likely to be wrong.
SCALES_PACK = Pack(
    name="scales",
    description="strike and shatter at every scale, pebble through cliff",
    entries=tuple(
        PackEntry(f"strike_{name}", "strike", 1, _options("limestone", name))
        for name in engine.SCALES
    ) + tuple(
        PackEntry(f"shatter_{name}", "shatter", 1, _options("limestone", name))
        for name in engine.SCALES
    ),
)

PACKS = {pack.name: pack for pack in (SHOWCASE, STONES_PACK, SCALES_PACK)}
