"""Cue and pack registries for the water generator."""

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

    mass: engine.Mass = engine.MASSES["body"]
    intensity: float = 0.8
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
    replaces: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


# Loops sit below the events, as in items 9 and 13: a bed plays under everything
# else. `aquarium` sits lower still -- it is a room tone that runs for the whole
# session, and Aquarium plays it under gameplay that does not exist yet.
_LOOP_PEAK = -14.0
_ROOM_PEAK = -18.0

CUES: dict[str, Cue] = {
    "splash": Cue(
        "splash", "event", "something entering water; mass picks the recording",
        engine.splash, -8.5, False, 0.0, "mass, intensity",
        replaces="environment_ambience_generator splash",
    ),
    "bubble": Cue(
        "bubble", "event", "bubbles as an event: a pop, or a burst of them",
        engine.bubble, -11.0, False, 0.0, "intensity",
    ),
    "pour": Cue(
        "pour", "event", "water poured from a vessel into more water",
        engine.pour, -11.0, False, 0.0, "intensity",
    ),
    "wade": Cue(
        "wade", "event", "one step through shallow water: in, through, out",
        engine.wade, -10.0, False, 0.0, "mass, intensity",
    ),
    "aquarium": Cue(
        "aquarium", "loop", "inside a fish tank: water body, airstone, pump",
        engine.aquarium, _ROOM_PEAK, True, 12.0, "intensity, seconds",
        replaces="environment_ambience_generator aquarium",
    ),
    "bubbler": Cue(
        "bubbler", "loop", "an intermittent aquarium bubbler: bursts, then quiet",
        engine.bubbler, _ROOM_PEAK, True, 30.0, "intensity, seconds",
    ),
    "bubbles": Cue(
        "bubbles", "loop", "a close rising column of bubbles",
        engine.bubbles, _LOOP_PEAK, True, 8.0, "intensity, seconds",
        replaces="environment_ambience_generator bubbles",
    ),
    "rain": Cue(
        "rain", "loop", "rain, from a drizzle to a downpour",
        engine.rain, _LOOP_PEAK, True, 12.0, "intensity, seconds",
        replaces="environment_ambience_generator rain",
    ),
}

GROUPS = ("event", "loop")

MIN_SECONDS = 2.0
MAX_SECONDS = 120.0

# What item 14 asks for and this generator does not make. Named here rather than
# left as silence, because "the source does not exist" and "nobody got to it" look
# identical from outside and only one of them is a decision.
#
#   reel_in    -- a fishing reel is a ratchet: a mechanism, not water. There is no
#                 CC0 recording of one on this machine and no water in the cue at
#                 all. It belongs to whatever generator owns mechanisms, and
#                 door_sound_generator is the closest thing that exists.
#   hook_cast  -- a line whistling through air. Also not water; combat_generators'
#                 swing is air and cloth and is the nearest working thing.
#   hook_strike -- is `splash --mass fish`, and shipping it under a second name
#                 would be the `punch`/`swing` mistake item 7 avoided.
#
# Aquarium now has an event mixer, but these remain absent for the material
# reasons below rather than because of a playback limitation.
# Cues that are deliberately absent, with the reason, so a gap does not read as an
# oversight and get re-attempted blind. Three of these were built and rejected by
# ear on 2026-07-15; three were never built because their subject is not water.
NOT_MADE: dict[str, str] = {
    "drip": "built and rejected 2026-07-15: no CC0 drip source exists, and a "
            "retuned splash reads as a tick. Absent rather than approximated -- "
            "the fireball/impacts precedent. See engine.py",
    "surface": "built and rejected 2026-07-15 as 'nothing, vague': water disturbed "
               "with no entry has no event to hold onto. What reads is `splash` "
               "at a light mass. See engine.py",
    "swim": "built and rejected 2026-07-15 as 'sparse, vague': entries on a rate "
            "read as things thrown in, not a body moving through. Gait is a "
            "cadence this does not model; `wade` is the one-step scope. See engine.py",
    "reel_in": "a ratchet mechanism, not water; no CC0 source on this machine",
    "hook_cast": "a line through air, not water; combat_generators' swing is nearer",
    "hook_strike": "this is `splash --mass fish` under another name",
}


def duration_for(cue_name: str, options: Options) -> float:
    cue = CUES[cue_name]
    if not cue.loop:
        return 0.0
    return cue.seconds if options.seconds is None else options.seconds


def render(cue_name: str, options: Options,
           rng: np.random.Generator) -> Render:
    """Render one cue, mastered and ready to write."""

    if cue_name not in CUES:
        hint = ""
        if cue_name in NOT_MADE:
            hint = f"; not made here: {NOT_MADE[cue_name]}"
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}{hint}"
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
        signal, sources = cue.render(options.mass, seconds, rng,
                                     options.intensity)
    else:
        signal, sources = cue.render(options.mass, rng, options.intensity)
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



def _options(mass: str, **overrides) -> Options:
    return Options(mass=engine.MASSES[mass], **overrides)


SHOWCASE = Pack(
    name="showcase",
    description="one of every cue, for auditioning",
    entries=tuple(
        PackEntry(name, name, 1,
                  Options(seconds=8.0 if CUES[name].loop else None))
        for name in CUES
    ),
)

# The mass axis, on the cue that shows it. splash is where `Mass` picks a
# different *recording* rather than resampling one, so it is the only place the
# registry's central claim is audible.
MASSES_PACK = Pack(
    name="masses",
    description="splash at every mass, droplet through shell",
    entries=tuple(
        PackEntry(f"splash_{name}", "splash", 1, _options(name))
        for name in engine.MASSES
    ),
)

# The five cues taken over from item 4, each against the version that was
# rejected. This is the pack that decides whether the takeover was justified.
REPLACEMENTS = Pack(
    name="replacements",
    description="the five cues taken from item 4 after its versions were rejected",
    entries=(
        PackEntry("splash", "splash", 3, _options("body")),
        PackEntry("aquarium", "aquarium", 1, _options("fish", seconds=12.0)),
        PackEntry("bubbles", "bubbles", 1, _options("fish", seconds=8.0)),
        PackEntry("rain", "rain", 1, _options("droplet", seconds=10.0)),
    ),
)
# drip is the fifth cue taken from item 4, and unlike the other four it is not
# here: it was rejected by ear a second time and is absent by decision. Four of
# five replacements survive; the fifth says the source does not exist. See
# engine.py and NOT_MADE.

# Aquarium's ambience runtime name is assets/audio/aquarium_ambience.wav.
# One file, named exactly, so promoting it is a copy rather than a decision about
# names. 45 s because it is a room tone that runs for a whole session and the
# beds behind it are 3.4-7.3 s -- a short loop of tiled material would be heard as
# a loop within a minute.
AQUARIUM = Pack(
    name="aquarium",
    description="the approved intermittent ambience loop for Aquarium",
    entries=(
        # An intermittent bubbler, not the busy `aquarium`: signed off by ear for
        # this game on 2026-07-15 as what a quiet tank in the corner sounds like.
        # 45 s so the intermittency reads and the loop is not noticed under
        # gameplay; the in-process mixer loops it. intensity 0.4 = sparse bursts.
        PackEntry("aquarium_ambience", "bubbler", 1,
                  _options("fish", seconds=45.0, intensity=0.4)),
    ),
)

AQUARIUM_EVENTS = Pack(
    name="aquarium-events",
    description="the feed and catch splashes used by Aquarium's event mixer",
    entries=(
        PackEntry("feed_splash", "splash", 1,
                  _options("droplet", intensity=0.35)),
        PackEntry("catch_splash", "splash", 1,
                  _options("fish", intensity=0.75)),
    ),
)

PACKS = {pack.name: pack
         for pack in (SHOWCASE, MASSES_PACK, REPLACEMENTS, AQUARIUM,
                      AQUARIUM_EVENTS)}
