"""Cue and pack registries for the magic generator.

The shape mirrors ``environment_ambience_generator``: each cue is its own entry
carrying a renderer, a group, a peak level and a ``loop`` flag, and the flag
decides mastering -- a loop goes through the fade-free master because a fade at
the ends is a seam, a one-shot takes the ordinary master and wants its fades.

The two options every cue honours are the element (``style``) and how hard the
spell lands (``intensity``); the one loop cue also honours ``seconds``. Nothing
here declares an axis it does not read.
"""

from __future__ import annotations

from ..._audio.models import VariationFilenames

import random
from dataclasses import dataclass
from typing import Callable


from . import engine
from .engine import STYLES, Style


# A cue is handed the rng, the length to render, and the options; it takes what
# it honours and ignores the rest. One-shots ignore the length -- a beam lasts
# as long as a beam lasts -- and only the ward loop reads it.
Render = Callable[[random.Random, float, "Options"], list[float]]


@dataclass(frozen=True)
class Options:
    """Every supported render axis; cues ignore fields they do not honour."""

    style: Style = STYLES["arcane"]
    intensity: float = 0.85
    seconds: float | None = None


@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Render
    peak_db: float
    loop: bool
    seconds: float
    layers: tuple[str, ...] = ()
    intensity_note: str = ""

    @property
    def honours(self) -> tuple[str, ...]:
        """``seconds`` is honoured exactly when the cue loops."""

        base = ("style", "intensity")
        return base + (("seconds",) if self.loop else ())

    def duration(self, options: Options) -> float:
        """One-shots keep their own length; loops take the requested one."""

        if not self.loop or options.seconds is None:
            return self.seconds
        return options.seconds


# Loops sit far below the events: a ward hum plays under everything and must
# not arrive at the level of the beam in front of it.
_LOOP_PEAK = -14.0


CUES: dict[str, Cue] = {
    # -- projectile ---------------------------------------------------------
    "beam": Cue(
        "beam", "projectile",
        "enchanted-sword launch: steel snap, magical crescent, airy streak",
        lambda r, sec, o: engine.beam(r, o.style, o.intensity),
        -2.0, False, 0.48,
        ("bright steel modal snap", "detuned crescent that sweeps up and lets go",
         "focused airy energy streak"),
        "how hard the launch lands: sweep top, streak level",
    ),
    "bolt": Cue(
        "bolt", "projectile",
        "a short offensive discharge: a zap",
        lambda r, sec, o: engine.bolt(r, o.style, o.intensity),
        -2.0, False, 0.30,
        ("crack", "descending detuned tone", "electric buzz"),
        "zap ferocity and buzz",
    ),
    "teleport": Cue(
        "teleport", "projectile",
        "a warp: a whoosh with a pitch discontinuity",
        lambda r, sec, o: engine.teleport(r, o.style, o.intensity),
        -3.0, False, 0.55,
        ("rushing whoosh", "pitched tone that cuts to a new frequency",
         "transient on the cut"),
        "whoosh weight",
    ),
    # -- cast ---------------------------------------------------------------
    "cast": Cue(
        "cast", "cast",
        "a spell being cast: rising shimmer to a soft pop",
        lambda r, sec, o: engine.cast(r, o.style, o.intensity),
        -4.0, False, 0.60,
        ("rising detuned shimmer", "sparkle grains", "soft landing pop"),
        "sparkle density and how bright it climbs",
    ),
    "charge": Cue(
        "charge", "cast",
        "energy gathering: a rising harmonic tremolo that ends at its loudest",
        lambda r, sec, o: engine.charge(r, o.style, o.intensity),
        -3.5, False, 0.72,
        ("in-drawn air, band climbing", "harmonic stack sweeping up",
         "building sparkle"),
        "how much energy is drawn in",
    ),
    "heal": Cue(
        "heal", "cast",
        "a gentle ascending chime-shimmer: restoration",
        lambda r, sec, o: engine.heal(r, o.style, o.intensity),
        -5.0, False, 0.90,
        ("ascending consonant chimes", "soft attacks", "fine dust overhead"),
        "dust density",
    ),
    "sparkle": Cue(
        "sparkle", "cast",
        "a sprinkle of tiny pitched grains: magic dust",
        lambda r, sec, o: engine.sparkle(r, o.style, o.intensity),
        -6.0, False, 0.60,
        ("many tuned grains", "pentatonic pitch set", "no sustained tone"),
        "how much dust falls",
    ),
    # -- state --------------------------------------------------------------
    "shield": Cue(
        "shield", "state",
        "a magical ward forming: a ring that settles into a hum",
        lambda r, sec, o: engine.shield(r, o.style, o.intensity),
        -4.0, False, 1.00,
        ("forming shimmer snap", "consonant bell ring",
         "low hum fading up as the ring falls"),
        "how present the sustained hum is",
    ),
    "shield_loop": Cue(
        "shield_loop", "state",
        "the steady ward hum, as a seamless loop",
        lambda r, sec, o: engine.shield_loop(r, sec, o.style, o.intensity),
        _LOOP_PEAK, True, 8.0,
        ("detuned harmonic drone, quantised to loop", "slow shimmer",
         "airy high bed, primed circularly"),
        "shimmer depth and airiness",
    ),
    "dissolve": Cue(
        "dissolve", "state",
        "a short descending magical life-energy dissolve",
        lambda r, sec, o: engine.dissolve(r, o.style, o.intensity),
        -5.0, False, 0.72,
        ("downward detuned shimmer", "airy band sliding down",
         "thinning dust, fading to nothing"),
        "how much dust trails the fade",
    ),
    "curse": Cue(
        "curse", "state",
        "a dark descending detune: a debuff",
        lambda r, sec, o: engine.curse(r, o.style, o.intensity),
        -3.5, False, 0.85,
        ("two low voices beating as they fall", "dissonant upper partial",
         "dark noise wash, lowpassed"),
        "wash weight",
    ),
}


GROUPS = ("projectile", "cast", "state")

LOOPS = tuple(name for name, cue in CUES.items() if cue.loop)
ONESHOTS = tuple(name for name, cue in CUES.items() if not cue.loop)

MIN_SECONDS = engine.MIN_SECONDS
MAX_SECONDS = engine.MAX_SECONDS


def render(cue_name: str, options: Options, rng: random.Random) -> list[float]:
    """Render and master a named cue."""

    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    if not 0.05 <= options.intensity <= 1.0:
        raise ValueError(
            f"intensity must be between 0.05 and 1; got {options.intensity}"
        )
    signal = cue.render(rng, cue.duration(options), options)
    return engine.master(signal, cue.peak_db, cue.loop)


def stem_for(cue_name: str, options: Options) -> str:
    """A descriptive default artifact name: cue, element, and a loop's length."""

    cue = CUES[cue_name]
    stem = f"{cue_name}_{options.style.name}"
    if cue.loop:
        stem = f"{stem}_{cue.duration(options):g}s".replace(".", "_")
    return stem


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



def _opts(style: str = "arcane", **overrides) -> Options:
    return Options(style=STYLES[style], **overrides)


# Fantasy Adventure loads its magic samples by subdirectory: ``sfx/combat/beam.wav`` with
# five variants, and ``sfx/player/death.wav``. This pack emits that tree under
# its own output/ root (it never touches the game) so a designer can drop
# ``combat/`` straight into the game's ``sfx/``.
#
# ``beam`` is the enchanted sword projectile, five variants, one per seed --
# exactly the bank sound.c's SND_BEAM already expects.
#
# ``dissolve`` is contributed as ``player/death_dissolve``: the "magical
# life-energy dissolve" layer player/death asks for. It is deliberately a
# separately named layer and not ``death.wav`` -- the game already ships a full
# death bank, and this is the magic layer to mix into it, four variants to match
# death's four. Overwriting the game's file is not this generator's job.
FANTASY_ADVENTURE = Pack(
    "fantasy-adventure",
    "Fantasy Adventure's magic bank: combat/beam x5 and the player/death dissolve layer",
    (
        PackEntry("combat/beam", "beam", 5, _opts("arcane", intensity=0.9)),
        PackEntry("player/death_dissolve", "dissolve", 4,
                  _opts("arcane", intensity=0.7)),
    ),
)


SHOWCASE = Pack(
    "showcase",
    "one representative render of every cue, for auditioning",
    tuple(
        PackEntry(
            name, name, 1,
            Options(seconds=6.0) if CUES[name].loop else Options(),
        )
        for name in CUES
    ),
)


# The element axis, made auditable: two signature cues rendered in all four
# elements so the character difference can be heard side by side.
ELEMENTS = Pack(
    "elements",
    "charge and bolt in every element, to audition the style axis",
    tuple(
        PackEntry(f"charge_{name}", "charge", 1, _opts(name))
        for name in STYLES
    )
    + tuple(
        PackEntry(f"bolt_{name}", "bolt", 1, _opts(name))
        for name in STYLES
    ),
)


PACKS = {pack.name: pack for pack in (FANTASY_ADVENTURE, SHOWCASE, ELEMENTS)}
