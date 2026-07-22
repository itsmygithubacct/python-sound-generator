"""Cue and pack registries for the unarmed Foley generator."""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from ..._audio.models import VariationFilenames

from ..._audio.models import RenderResult as Render
from ..._audio.runtime import derive_seed

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..combat_generators.lib import cues as combat

from . import engine


@dataclass(frozen=True)
class Options:
    """Every supported render axis; cues ignore fields they do not honour."""

    weight: engine.Weight = engine.WEIGHTS["medium"]
    intensity: float = 0.9




@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Callable[..., tuple[np.ndarray, list[str]]]
    peak_db: float
    honours: Honours | str | tuple[str, ...] = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


CUES: dict[str, Cue] = {
    "hit": Cue(
        "hit", "impact", "a fist landing on a body; deep, blunt, no pitch",
        engine.hit, -9.0, "weight, intensity",
    ),
    "block": Cue(
        "block", "impact", "a blow caught on a braced forearm; sharp, stops dead",
        engine.block, -10.0, "weight, intensity",
    ),
    "knockout": Cue(
        "knockout", "impact", "the hit that ends it, and the body after it",
        engine.knockout, -8.5, "weight, intensity",
    ),
}

GROUPS = ("impact",)


def render(cue_name: str, options: Options,
           rng: np.random.Generator) -> Render:
    """Render one cue, mastered and ready to write."""

    if cue_name not in CUES:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        )
    cue = CUES[cue_name]
    signal, sources = cue.render(options.weight, rng, options.intensity)
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
    """A named bank. `entries` are made here; `borrows` are not.

    A game's bank does not respect the seam between a recording and a
    synthesizer, so a pack that only shipped the Foley half would leave every
    caller running two commands and stitching the result. `borrows` names files
    taken verbatim from a combat_generators pack -- the swings and the energy
    cues, which are better synthesized and were signed off there. They are
    rendered through that generator's own registry using the same stable seed
    identity, so the same seed yields the same bytes, and the two CLIs default
    to the same seed on purpose. There is one implementation of a punch, and it
    is not here.
    """

    name: str
    description: str
    entries: tuple[PackEntry, ...]
    borrows: tuple[str, ...] = ()
    combat_pack: str | None = None


    def borrowed_entries(self) -> tuple[combat.PackEntry, ...]:
        if not self.borrows:
            return ()
        source = combat.PACKS[self.combat_pack]
        by_name = {e.filename: e for e in source.entries}
        missing = [name for name in self.borrows if name not in by_name]
        if missing:
            raise KeyError(
                f"{self.name} borrows {missing} from combat_generators pack "
                f"{self.combat_pack!r}, which no longer ships them"
            )
        return tuple(by_name[name] for name in self.borrows)


def _options(weight: str, **overrides) -> Options:
    return Options(weight=engine.WEIGHTS[weight], **overrides)


# Arcade Fighter's bank, less the files nothing can honestly make yet. hit and
# block play on contact (src/game.c:279, 290) and ko on the finish (game.c:688);
# those three are the recordings. punch, kick and special are borrowed from
# combat_generators, which fires them the moment a move starts, before anything
# has been touched. Variant counts match the banks already in assets/sfx/.
#
# Four of the game's 29 combat files are not here, and are not borrowed either:
#   fireball x4 -- fire_sound_generator makes these. Not borrowed, because that
#                  generator is the fireball's home and borrowing across two
#                  hybrids to save one command would couple them for cosmetics.
#                  Run `soundgen fire pack arcade-fighter` for
#                  the other four. (combat_generators' energy_release filled the
#                  slot until it was heard: it is a phaser blast, not a
#                  fireball.)
# The UI cues -- menu/select/round/win -- belong to ui_game_state_generator.
# The game falls back to its own built-in synth for anything absent,
# so a partial bank is a working bank. Shipping a wrong sound would not be.
ARCADE_FIGHTER = Pack(
    name="arcade-fighter",
    description="arcade-fighter combat bank: impacts and swings "
                "(no fireball, no UI cues)",
    entries=(
        PackEntry("hit", "hit", 5, _options("medium")),
        PackEntry("block", "block", 4, _options("medium")),
        PackEntry("ko", "knockout", 3, _options("brutal")),
    ),
    borrows=("punch", "kick", "special"),
    combat_pack="arcade-fighter",
)

# Rancher is a monster battler: its hit (six variants, src/game.c:1235) is the
# battle impact this bank differentiates. Its train cue is not here -- that
# fires when a drill starts (game.c:621), so it is a UI confirm and not a blow --
# and the creature's own voice belongs to creature_foley_generator.
CREATURE_RANCHER = Pack(
    name="creature-rancher",
    description="battle impacts for creature-rancher",
    entries=(
        PackEntry("hit", "hit", 6, _options("heavy")),
    ),
)

SHOWCASE = Pack(
    name="showcase",
    description="one of everything, at every weight, for auditioning",
    entries=tuple(
        PackEntry(f"{name}_{weight}", name, 1, _options(weight))
        for name in CUES
        for weight in engine.WEIGHTS
    ),
)

PACKS = {pack.name: pack for pack in (ARCADE_FIGHTER, CREATURE_RANCHER, SHOWCASE)}


def combat_seed(entry: combat.PackEntry, index: int, seed: int) -> int:
    """Return the source provider's stable seed for a borrowed artifact."""

    provider = combat.__package__.removesuffix(".lib")
    return derive_seed(
        seed,
        provider=provider,
        cue=entry.cue,
        filename=entry.filename,
        variation=index + 1,
    )


def combat_render(entry: combat.PackEntry, index: int,
                  seed: int) -> list[float]:
    """Render one borrowed file through combat_generators' own registry.

    The stable seed identity matches that project's ``cmd_pack``, so a borrowed
    punch is the same punch, to the byte, as the one its own CLI writes.
    """
    rng = random.Random(combat_seed(entry, index, seed))
    return combat.render(entry.cue, entry.opts, rng)
