"""Cue and pack registries for the fire generator."""

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

    scale: engine.Scale = engine.SCALES["campfire"]
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
# a fire that arrives at the same level as the explosion in front of it leaves
# the mix nowhere to go.
_LOOP_PEAK = -14.0

CUES: dict[str, Cue] = {
    "burning": Cue(
        "burning", "loop", "a fire already alight: roar, hiss, and crackle",
        engine.burning, _LOOP_PEAK, True, 12.0, "scale, seconds",
    ),
    "torch": Cue(
        "torch", "loop", "a handheld flame, dragged through air and fluttering",
        engine.torch, _LOOP_PEAK, True, 8.0, "scale, seconds",
    ),
    "flame": Cue(
        "flame", "loop", "a pressurised jet: flamethrower, burner, breath",
        engine.flame, -12.0, True, 8.0, "scale, seconds",
    ),
    "lava_flow": Cue(
        "lava_flow", "loop",
        "molten rock as a bed: slow, low, constant rumble and molten bubbles",
        engine.lava_flow, -15.0, True, 8.0, "scale, seconds",
    ),
    "ignite": Cue(
        "ignite", "event", "catching light: the whoomp, and the fire that stays",
        engine.ignite, -10.0, False, 0.0, "scale, intensity",
    ),
    "fireball": Cue(
        "fireball", "event", "a flame projectile: thrown, and going somewhere",
        engine.fireball, -9.0, False, 0.0, "scale, intensity",
    ),
    "fire_explosion": Cue(
        "fire_explosion", "event", "fuel going up: a blast, then everything burns",
        engine.fire_explosion, -8.0, False, 0.0, "scale, intensity",
    ),
    "extinguish": Cue(
        "extinguish", "event", "going out: the roar collapses, the steam outlives it",
        engine.extinguish, -12.0, False, 0.0, "scale, intensity",
    ),
    "lava": Cue(
        "lava", "event", "viscous molten splash, sizzle, armour hiss, and bubbles",
        engine.lava, -7.0, False, 0.0, "scale, intensity",
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


# Arcade Fighter's fireball, the file that started this generator. Four variants,
# matching the bank its src/sound.c already loads. combat_generators' pack
# leaves this slot empty and its energy_release stays a phaser blast.
ARCADE_FIGHTER = Pack(
    name="arcade-fighter",
    description="the fireball arcade-fighter has been missing",
    entries=(
        PackEntry("fireball", "fireball", 4, _options("campfire")),
    ),
)

FLYING_JOUST = Pack(
    name="flying-joust",
    description="the four compact molten-lava impacts Flying Joust loads by name",
    entries=(
        PackEntry("lava", "lava", 4, _options("inferno")),
    ),
)

VOLCANO_SIMULATOR = Pack(
    name="volcano-simulator",
    description="the seamless molten crater bed for a volcano simulator",
    entries=(
        PackEntry(
            "lava_flow", "lava_flow", 1,
            _options("bonfire", seconds=8.0),
        ),
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
    description="burning at every scale, ember through inferno",
    entries=tuple(
        PackEntry(f"burning_{name}", "burning", 1,
                  _options(name, seconds=8.0))
        for name in engine.SCALES
    ),
)

PACKS = {
    pack.name: pack
    for pack in (
        ARCADE_FIGHTER, FLYING_JOUST, VOLCANO_SIMULATOR, SHOWCASE, SCALES_PACK
    )
}
