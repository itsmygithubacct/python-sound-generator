"""Cue and pack registries for the environment ambience generator.

Unlike the weapons bank there is no action/noun split to model here. A firearm
cue is a verb (`fire`) applied to a profile (`rifle`), but nothing is *done* to
an ambience -- the texture is the sound. So each texture is its own cue, and
the two options every cue shares are how long the loop runs and how much is
happening inside it.

The registry carries `loop`, and that flag decides mastering: a bed goes
through the fade-free loop master because a fade would put a hole at the loop
point, while a one-shot takes the ordinary `dsp.master` and wants its fades.
"""

from __future__ import annotations

from ..._audio.models import VariationFilenames

import random
from dataclasses import dataclass
from typing import Callable

from ..._audio import dsp

from ..._audio import loops as loopdsp
from . import bells, engine, horn, ocean, wind


# A cue is handed the rng, the length it should render, and the options; it
# takes what it honours and ignores the rest. Adapting happens in the registry
# rather than in the engines, so a renderer keeps an explicit signature that
# reads well from the Python API and from a test.
Render = Callable[[random.Random, float, "Options"], list[float]]


@dataclass(frozen=True)
class Options:
    """Every supported render axis; cues ignore fields they do not honour.

    `seconds` of None means "this cue's own default loop length", which is not
    one shared number: a machine repeats in six seconds, a cave wants twelve,
    and an ocean needs ninety before its wave sets stop repeating audibly.
    """

    seconds: float | None = None
    intensity: float = 0.6
    foliage: float = 0.5
    pattern: str = "underway"
    quarter: int = 4
    hour: int = 0


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
    extra: tuple[str, ...] = ()

    @property
    def honours(self) -> tuple[str, ...]:
        """Length is derived, not declared: a one-shot's length is not yours.

        A thunderclap lasts as long as a thunderclap lasts. Only a loop has a
        length worth asking for, so `--seconds` is honoured exactly when `loop` is.
        Anything beyond that is per-cue and declared in `extra`.
        """

        base = ("seconds", "intensity") if self.loop else ("intensity",)
        return base + self.extra

    def duration(self, options: Options) -> float:
        """One-shots set their own length from the sound they represent."""

        if not self.loop or options.seconds is None:
            return self.seconds
        return options.seconds


# Beds master well below the one-shots on purpose. Ambience is the floor of a
# mix, not an event in it, and a bed normalised to the same peak as a thunder
# clap will bury every cue a game lays on top of it.
_LOOP_PEAK = -12.0
_QUIET_LOOP_PEAK = -15.0


CUES: dict[str, Cue] = {
    "wind": Cue(
        "wind", "weather", "gusting wind over an exposed edge",
        lambda r, sec, o: wind.blow(r, sec, o.intensity, o.foliage),
        _LOOP_PEAK, True, 20.0,
        ("brown sub-rumble", "body: dual-lowpass crossfade on one noise",
         "gated hiss", "foliage: broadleaf + conifer", "aeolian wire tone"),
        "Beaufort: 0.1 breeze, 0.5 fresh, 1.0 near gale",
        ("foliage",),
    ),
    "ocean": Cue(
        "ocean", "weather", "surf: wave sets, breaking crests, and spray",
        lambda r, sec, o: ocean.surf(r, sec, o.intensity),
        _LOOP_PEAK, True, 90.0,
        ("deep swell rumble", "wave body, cutoff tracking height",
         "surf hiss delayed behind the crest", "foam fizz on breaks",
         "background wash"),
        "sea state: calm lapping to rough surf",
    ),
    "cave": Cue(
        "cave", "interior", "a large wet cave with reverberant drips",
        lambda r, sec, o: engine.cave(r, sec, o.intensity), _QUIET_LOOP_PEAK, True, 12.0,
        ("dark rumble", "air movement", "drips", "long reverb"),
        "drip rate and air movement",
    ),
    "dungeon": Cue(
        "dungeon", "interior", "a built cave: drone, iron, and a squarer room",
        lambda r, sec, o: engine.dungeon(r, sec, o.intensity), _QUIET_LOOP_PEAK, True, 12.0,
        ("dark rumble", "minor-second drone", "structural groan", "drips",
         "iron clanks", "long reverb"),
        "event rate and structural groan",
    ),
    "forest": Cue(
        "forest", "life", "daytime woodland: leaves, insects, and birds",
        lambda r, sec, o: engine.forest(r, sec, o.intensity, o.foliage),
        _LOOP_PEAK, True, 20.0,
        ("foliage bed", "low under-wind", "insect band",
         "birdsong: real species at range"),
        "bird activity",
        ("foliage",),
    ),
    "crowd": Cue(
        "crowd", "life", "massed indistinct conversation",
        lambda r, sec, o: engine.crowd(r, sec, o.intensity), _LOOP_PEAK, True, 8.0,
        ("syllable grains", "two formant banks", "murmur", "room"),
        "empty room to packed",
    ),
    "mechanical": Cue(
        "mechanical", "machine", "machinery running under load",
        lambda r, sec, o: engine.mechanical(r, sec, o.intensity), _LOOP_PEAK, True, 6.0,
        ("rotation-locked hum", "blade-pass thrum", "bearing whine",
         "broadband", "rattle"),
        "whine, rattle, and broadband",
    ),
    "space": Cue(
        "space", "machine", "a ship's interior: life support and hull",
        lambda r, sec, o: engine.space(r, sec, o.intensity), _QUIET_LOOP_PEAK, True, 12.0,
        ("detuned sub drone", "life-support air", "shimmer", "hull groan"),
        "air handling level",
    ),
    "bell": Cue(
        "bell", "oneshot", "a single church bell",
        lambda r, sec, o: bells.single(r, sec, o.intensity),
        -3.0, False, 10.0,
        ("10 inharmonic partials", "split modes beating",
         "clapper impact and bounce", "warble"),
        "proximity: distant and warm to close and bright",
    ),
    "bell_chime": Cue(
        "bell_chime", "oneshot", "a Westminster quarter, and the hour tolled",
        lambda r, sec, o: bells.chime(r, sec, o.intensity, o.quarter, o.hour),
        -3.0, False, 30.0,
        ("Westminster phrases", "hour bell an octave down", "overlapping tolls"),
        "proximity",
        ("quarter", "hour"),
    ),
    "foghorn": Cue(
        "foghorn", "oneshot", "a vessel's fog signal",
        lambda r, sec, o: horn.signal(r, sec, o.intensity, o.pattern),
        -1.0, False, 6.5,
        ("10 harmonics, 1/n rolloff", "attack chuff and pitch rise",
         "pressure tremolo", "release sag", "diaphone grunt", "bell formant"),
        "proximity",
        ("pattern",),
    ),
    "thunder": Cue(
        "thunder", "oneshot", "thunder, from a distant roll to a close crack",
        lambda r, sec, o: engine.thunder(r, sec, o.intensity), -0.5, False, 6.5,
        ("crack", "smeared rolls", "sub", "sky diffusion"),
        "proximity, not volume",
    ),
    "rockfall": Cue(
        "rockfall", "oneshot", "rock and gravel coming down",
        lambda r, sec, o: engine.rockfall(r, sec, o.intensity), -2.0, False, 3.4,
        ("initial crack", "size-skewed stone impacts", "grit", "rumble"),
        "how much rock",
    ),
}


GROUPS = ("weather", "interior", "life", "machine", "oneshot")

LOOPS = tuple(name for name, cue in CUES.items() if cue.loop)
ONESHOTS = tuple(name for name, cue in CUES.items() if not cue.loop)


def render(cue_name: str, options: Options, rng: random.Random) -> list[float]:
    """Render and master a named cue."""

    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    signal = cue.render(rng, cue.duration(options), options)
    if cue.loop:
        return loopdsp.master(signal, cue.peak_db)
    return dsp.master(signal, cue.peak_db)


def stem_for(cue_name: str, options: Options) -> str:
    """Build a default artifact name; loop length is part of a loop's identity."""

    cue = CUES[cue_name]
    if not cue.loop:
        return cue_name
    return f"{cue_name}_{cue.duration(options):g}s".replace(".", "_")


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



DEFAULTS = Options()

SHOWCASE = Pack(
    "showcase",
    "one representative render of every cue",
    tuple(
        PackEntry(name, name, 1, DEFAULTS)
        for name in CUES
    ),
)


AMBIENCE = Pack(
    "ambience",
    "every loop at two intensities, plus one-shot variations",
    tuple(
        PackEntry(f"{name}_calm", name, 1, Options(intensity=0.3))
        for name in LOOPS
    )
    + tuple(
        PackEntry(f"{name}_strong", name, 1, Options(intensity=0.95))
        for name in LOOPS
    )
    + tuple(
        PackEntry(name, name, 3, DEFAULTS)
        for name in ONESHOTS
    ),
)


# Rain and water events are owned by water_sounds_generator. This pack keeps
# the dry outdoor half here so callers cannot accidentally ship two models for
# the same cue.
WEATHER = Pack(
    "weather",
    "wind, thunder, and rockfall for outdoor games",
    (
        PackEntry("wind_calm", "wind", 1, Options(intensity=0.25)),
        PackEntry("wind_storm", "wind", 1, Options(intensity=1.0)),
        PackEntry("thunder_distant", "thunder", 2, Options(intensity=0.2)),
        PackEntry("thunder_close", "thunder", 2, Options(intensity=0.95)),
        PackEntry("rockfall", "rockfall", 2, Options(intensity=0.7)),
    ),
)


# town_shore.py in the older project is a composition: ocean under sparse
# birdsong, with the church clock and a barge horn arriving on their own
# schedules. A generator cannot ship that as one file -- the bells are on a
# 15-minute clock and the horn on a 25-minute one, and no loop holds either --
# so it ships the parts and a game lays them out. The loops are long because
# that is what the source's slow structure needs: the surf beat alone runs to
# 180 s.
TOWN_SHORE = Pack(
    "town_shore",
    "the town shore soundscape: surf, birds, bells, and a barge horn",
    (
        PackEntry("shore_surf", "ocean", 1, Options(seconds=90.0,
                                                    intensity=0.3)),
        PackEntry("shore_surf_rough", "ocean", 1, Options(seconds=90.0,
                                                          intensity=0.85)),
        PackEntry("shore_wind", "wind", 1, Options(seconds=30.0, intensity=0.4,
                                                   foliage=0.1)),
        PackEntry("birdsong", "forest", 1, Options(seconds=30.0, intensity=0.2,
                                                   foliage=0.6)),
        PackEntry("bell", "bell", 2, Options(intensity=0.5)),
        PackEntry("quarter_1", "bell_chime", 1, Options(intensity=0.5, quarter=1)),
        PackEntry("quarter_4_hour", "bell_chime", 1,
                  Options(intensity=0.5, quarter=4, hour=12)),
        PackEntry("horn_underway", "foghorn", 2,
                  Options(intensity=0.3, pattern="underway")),
        PackEntry("horn_towing", "foghorn", 1,
                  Options(intensity=0.3, pattern="towing")),
    ),
)


PACKS = {pack.name: pack for pack in
         (SHOWCASE, AMBIENCE, WEATHER, TOWN_SHORE)}
