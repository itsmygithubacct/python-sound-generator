"""Cue and pack registries for the human voice generator."""

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

    body: engine.Body = engine.BODIES["adult"]
    voice: engine.Voice = engine.VOICES["male"]
    intensity: float = 0.85




@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Callable[..., tuple[np.ndarray, list[str]]] | None
    peak_db: float
    honours: Honours | str | tuple[str, ...] = ""
    voices: tuple[str, ...] = ("male",)
    absent_reason: str = ""

    @property
    def absent(self) -> bool:
        return self.render is None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


CUES: dict[str, Cue] = {
    "hurt": Cue(
        "hurt", "damage", "a brief nonverbal pain grunt (Fantasy Adventure player/hurt)",
        engine.hurt, -3.5, "body, voice, intensity", ("male", "female"),
    ),
    "death": Cue(
        "death", "damage", "a human death gasp/groan (the gasp layer of player/death)",
        engine.death, -5.0, "body, intensity", ("male",),
    ),
    "effort": Cue(
        "effort", "exertion", "a swing/lift exertion grunt, not pain",
        engine.effort, -3.5, "body, voice, intensity", ("male", "female"),
    ),
    "shout": Cue(
        "shout", "exertion", "a short vocal call: a rally, a challenge, a cry",
        engine.shout, -2.5, "body, intensity", ("male",),
    ),
    # Registered but unsourced: visible in `list`, honest at render time.
    "breath": Cue(
        "breath", "absent", "an inhale/exhale (ABSENT: no CC0 breath bundled)",
        None, -8.0, "-", (), engine.ABSENT["breath"],
    ),
}

GROUPS = ("damage", "exertion", "absent")


def render(cue_name: str, options: Options,
           rng: np.random.Generator) -> Render:
    """Render one cue, mastered and ready to write."""

    if cue_name not in CUES:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        )
    cue = CUES[cue_name]
    if cue.absent or cue.render is None:
        raise ValueError(
            f"cue {cue_name!r} is ABSENT -- no recording is bundled for it. "
            f"{cue.absent_reason}"
        )
    if not 0.0 <= options.intensity <= 1.0:
        raise ValueError(f"intensity must be in [0, 1]; got {options.intensity}")
    if options.voice.name not in cue.voices:
        raise ValueError(
            f"cue {cue_name!r} has no {options.voice.name} recording; it is "
            f"bundled for: {', '.join(cue.voices)}"
        )
    signal, sources = cue.render(options.body, options.voice, rng,
                                 options.intensity)
    return Render(engine.master(signal, cue.peak_db), tuple(sources))


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



def _options(body: str = "adult", voice: str = "male", **kw) -> Options:
    return Options(body=engine.BODIES[body], voice=engine.VOICES[voice], **kw)


# Fantasy Adventure's human-voice bank, named exactly as src/tools/compile_audio.py loads
# it: sfx/player/hurt.wav x5 and sfx/player/death.wav x4, first variant with no
# suffix. The hero is a "small fantasy hero", so a youth body rather than an
# adult. death is the gasp only -- the body fall and the dissolve are the
# game's to layer (combat_generators, magic_sound_generator).
FANTASY_ADVENTURE = Pack(
    name="fantasy-adventure",
    description="Fantasy Adventure's player/hurt x5 and player/death gasp x4",
    entries=(
        PackEntry("player/hurt", "hurt", 5, _options("youth")),
        PackEntry("player/death", "death", 4, _options("youth", intensity=0.7)),
    ),
)

SHOWCASE = Pack(
    name="showcase",
    description="one of everything renderable, both voices, a range of bodies",
    entries=(
        PackEntry("hurt", "hurt", 1, _options("adult")),
        PackEntry("death", "death", 1, _options("adult", intensity=0.7)),
        PackEntry("effort", "effort", 1, _options("adult")),
        PackEntry("shout", "shout", 1, _options("adult", intensity=0.95)),
        PackEntry("hurt_female", "hurt", 1, _options("adult", "female")),
        PackEntry("effort_female", "effort", 1, _options("adult", "female")),
        PackEntry("hurt_child", "hurt", 1, _options("child")),
        PackEntry("hurt_giant", "hurt", 1, _options("giant")),
    ),
)

# The size axis, front and centre: one hurt grunt at every body.
BODIES_PACK = Pack(
    name="bodies",
    description="a hurt grunt at every body, child through giant",
    entries=tuple(
        PackEntry(f"hurt_{name}", "hurt", 1, _options(name))
        for name in engine.BODIES
    ),
)

PACKS = {pack.name: pack for pack in (FANTASY_ADVENTURE, SHOWCASE, BODIES_PACK)}
