"""Cue and pack registries for the creature Foley generator."""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from ..._audio.models import VariationFilenames

from ..._audio.models import RenderResult as Render

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import engine


@dataclass(frozen=True)
class Options:
    """Every supported render axis; cues ignore fields they do not honour."""

    size: engine.SizeProfile = engine.SIZES["medium"]
    character: engine.CharacterProfile = engine.CHARACTERS["beast"]
    intensity: float = 0.85
    beats: int = 3
    room: float = 0.06
    duration: float | None = None




@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Callable[..., tuple[np.ndarray, list[str]]]
    duration: float
    peak_db: float
    honours: Honours | str | tuple[str, ...] = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


CUES: dict[str, Cue] = {
    "chirp": Cue(
        "chirp", "voice", "short contact call, peep, or chitter",
        engine.chirp, 0.55, -3.0, "size, character, intensity, room",
    ),
    "purr": Cue(
        "purr", "voice", "sustained contentment rumble at a body-scaled rate",
        engine.purr, 2.20, -9.0, "size, character, intensity, room",
    ),
    "growl": Cue(
        "growl", "voice", "low threat vocalization",
        engine.growl, 1.10, -4.0, "size, character, intensity, room",
    ),
    "roar": Cue(
        "roar", "voice", "full-power call, stacked for large bodies",
        engine.roar, 1.60, -1.0, "size, character, intensity, room",
    ),
    "pain": Cue(
        "pain", "voice", "hurt reaction",
        engine.pain, 0.80, -2.5, "size, character, intensity, room",
    ),
    "defeat": Cue(
        "defeat", "voice", "death cry trailing into a final breath",
        engine.defeat, 1.50, -2.0, "size, character, intensity, room",
    ),
    "bite": Cue(
        "bite", "combat", "jaw snap with flesh contact",
        engine.bite, 0.45, -3.0, "size, character, intensity, room",
    ),
    "claw": Cue(
        "claw", "combat", "multi-stroke claw swipe with limb movement",
        engine.claw, 0.60, -5.0, "size, character, intensity, room",
    ),
    "wing": Cue(
        "wing", "movement", "wing beats re-timed to the creature's wing rate",
        engine.wing, 0.0, -6.0, "size, character, intensity, beats, room",
    ),
    "move": Cue(
        "move", "movement", "body movement, hide, and heavy footfall",
        engine.move, 0.70, -9.0, "size, character, intensity, room",
    ),
    "egg_crack": Cue(
        "egg_crack", "life cycle", "shell fracturing and falling fragments",
        engine.egg_crack, 0.90, -4.0, "size, character, intensity, room",
    ),
    "hatch": Cue(
        "hatch", "life cycle", "shell gives way and the hatchling calls",
        engine.hatch, 2.00, -3.0, "size, character, intensity, room",
    ),
}


GROUPS = ("voice", "combat", "movement", "life cycle")


def duration_for(cue_name: str, options: Options) -> float:
    """Natural length of a cue for this creature, before any override."""

    if options.duration is not None:
        return options.duration
    if cue_name == "wing":
        interval = 1.0 / max(0.4, options.size.wing_rate)
        return interval * max(0, options.beats - 1) + 0.55 + 0.30 * options.size.duration_scale
    return CUES[cue_name].duration * options.size.duration_scale


def render(cue_name: str, options: Options, rng: np.random.Generator) -> Render:
    """Render and master a named cue."""

    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    if not 0.05 <= options.intensity <= 1.5:
        raise ValueError("intensity must be between 0.05 and 1.5")
    if not 1 <= options.beats <= 8:
        raise ValueError("beats must be between 1 and 8")
    if not 0.0 <= options.room <= 1.0:
        raise ValueError("room must be between 0 and 1")

    duration = duration_for(cue_name, options)
    if not 0.10 <= duration <= 12.0:
        raise ValueError("duration must be between 0.1 and 12 seconds")
    frames = max(1, round(duration * engine.SR))

    if cue_name == "wing":
        raw, used = cue.render(
            options.size, options.character, frames, options.intensity, rng,
            options.beats,
        )
    else:
        raw, used = cue.render(
            options.size, options.character, frames, options.intensity, rng
        )

    raw = engine.early_reflections(raw, options.room, rng)
    # Mastering normalizes to the target, so a gain applied here would simply be
    # undone. To change the delivered level, intensity has to move the target.
    peak_db = cue.peak_db + 20.0 * math.log10(
        0.70 + 0.30 * min(1.0, options.intensity)
    )
    return Render(engine.master(raw, peak_db), tuple(dict.fromkeys(used)))


def stem_for(cue_name: str, options: Options) -> str:
    """Build a descriptive default artifact name from honoured options."""

    return f"{cue_name}_{options.character.name}_{options.size.name}"


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


def _options(character: str, size: str, **overrides) -> Options:
    return Options(
        size=engine.SIZES[size],
        character=engine.CHARACTERS[character],
        **overrides,
    )


SHOWCASE = Pack(
    "showcase",
    "one representative render of every cue",
    (
        PackEntry("chirp_bird_small", "chirp", 1, _options("bird", "small")),
        PackEntry("purr_beast_medium", "purr", 1, DEFAULTS),
        PackEntry("growl_beast_medium", "growl", 1, DEFAULTS),
        PackEntry("roar_beast_huge", "roar", 1, _options("beast", "huge")),
        PackEntry("pain_beast_medium", "pain", 1, DEFAULTS),
        PackEntry("defeat_beast_medium", "defeat", 1, DEFAULTS),
        PackEntry("bite_beast_large", "bite", 1, _options("beast", "large")),
        PackEntry("claw_beast_medium", "claw", 1, DEFAULTS),
        PackEntry("wing_bird_small", "wing", 1, _options("bird", "small")),
        PackEntry("move_beast_large", "move", 1, _options("beast", "large")),
        PackEntry("egg_crack_reptile_small", "egg_crack", 1, _options("reptile", "small")),
        PackEntry("hatch_reptile_small", "hatch", 1, _options("reptile", "small")),
    ),
)


BESTIARY = Pack(
    "bestiary",
    "voice banks for every character at its most typical size",
    tuple(
        PackEntry(f"{cue}_{character}_{size}", cue, 2, _options(character, size))
        for character, size in (
            ("beast", "medium"), ("beast", "huge"), ("bird", "small"),
            ("insect", "tiny"), ("reptile", "large"), ("slime", "medium"),
            ("spectral", "medium"),
        )
        for cue in ("chirp", "growl", "roar", "pain", "defeat")
    ),
)


FOLEY = Pack(
    "foley",
    "non-vocal creature Foley across the size range",
    tuple(
        PackEntry(f"wing_{character}_{size}", "wing", 2,
                  _options(character, size, beats=beats))
        for character, size, beats in (
            ("bird", "tiny", 4), ("bird", "small", 3), ("beast", "large", 3),
            ("beast", "huge", 2), ("insect", "tiny", 4),
        )
    )
    + tuple(
        PackEntry(f"{cue}_beast_{size}", cue, 2, _options("beast", size))
        for size in ("small", "medium", "large", "huge")
        for cue in ("bite", "claw", "move")
    )
    + tuple(
        PackEntry(f"{cue}_{character}_{size}", cue, 2, _options(character, size))
        for character, size in (("reptile", "small"), ("bird", "tiny"), ("beast", "large"))
        for cue in ("egg_crack", "hatch")
    ),
)


FLYING_JOUST = Pack(
    "flying-joust",
    "wing flaps, mount calls, rider pain, and eggs for Flying Joust",
    (
        PackEntry("wing_flap", "wing", 3, _options("bird", "medium", beats=1)),
        PackEntry("wing_climb", "wing", 3, _options("bird", "medium", beats=4)),
        PackEntry("mount_call", "chirp", 3, _options("bird", "medium")),
        PackEntry("mount_screech", "roar", 3, _options("bird", "medium")),
        PackEntry("bird_pain", "pain", 3, _options("bird", "small")),
        PackEntry("bird_defeat", "defeat", 3, _options("bird", "small")),
        PackEntry("egg_crack", "egg_crack", 3, _options("bird", "small")),
        PackEntry("egg_hatch", "hatch", 3, _options("bird", "small")),
        PackEntry("lava_beast", "growl", 2, _options("beast", "huge")),
    ),
)


RANCHER = Pack(
    "rancher",
    "creature expressions and battle reactions for Creature Rancher",
    tuple(
        PackEntry(f"{cue}_{size}", cue, 3, _options("beast", size))
        for size in ("small", "medium", "large")
        for cue in ("chirp", "purr", "growl", "pain")
    )
    + (
        PackEntry("battle_bite", "bite", 3, _options("beast", "medium")),
        PackEntry("battle_claw", "claw", 3, _options("beast", "medium")),
        PackEntry("battle_roar", "roar", 3, _options("beast", "large")),
        PackEntry("defeat", "defeat", 3, _options("beast", "medium")),
        PackEntry("idle_move", "move", 3, _options("beast", "medium")),
    ),
)


AQUARIUM = Pack(
    "aquarium",
    "small feeding bite and large shark bite for Aquarium's event mixer",
    (
        PackEntry("fish_bite", "bite", 1,
                  _options("beast", "tiny", intensity=0.55, room=0.02)),
        PackEntry("shark_bite", "bite", 1,
                  _options("beast", "large", intensity=1.0, room=0.03)),
    ),
)


PACKS = {pack.name: pack
         for pack in (SHOWCASE, BESTIARY, FOLEY, FLYING_JOUST, RANCHER,
                      AQUARIUM)}
