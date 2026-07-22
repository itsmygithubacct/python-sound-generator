"""The cue registry: every sound this project can render, by name.

A cue is a named renderer plus the options it honours. Everything the CLI can
produce is in CUES or PACKS, so `combat_sfx.py list` is generated from the same
tables the renderer uses and cannot drift out of date.
"""

from __future__ import annotations

from ...._audio.models import Honours, normalize_honours

import random
from dataclasses import dataclass
from typing import Callable

from ...._audio import dsp
from . import energy, events, ranged, strikes
from .materials import BLADE, TARGETS, Target
from .weapons import BOOT, FIST, SWORD, WEAPONS, Weapon


@dataclass
class Options:
    """Everything a cue might want. Cues ignore what they do not use."""
    weapon: Weapon = SWORD
    target: Target = BLADE
    surface: str = "stone"
    armored: bool = True


@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Callable[[random.Random, Options], list[float]]
    # None means "take the level from the target being hit"
    peak_db: float | None = None
    honours: Honours | str | tuple[str, ...] = ""

    def level(self, opts: Options) -> float:
        return opts.target.peak_db if self.peak_db is None else self.peak_db

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


CUES: dict[str, Cue] = {
    "strike": Cue(
        "strike", "melee", "weapon meets target, no swing (immediate impact)",
        lambda rng, o: strikes.strike(o.weapon, o.target, rng),
        None, "weapon, target"),
    "swing": Cue(
        "swing", "melee", "swing through empty air, a miss",
        lambda rng, o: strikes.swing(o.weapon, rng),
        -10.5, "weapon"),
    "full": Cue(
        "full", "melee", "a swing that lands: whoosh into impact",
        lambda rng, o: strikes.full_strike(o.weapon, o.target, rng),
        None, "weapon, target"),
    "parry": Cue(
        "parry", "melee", "blades meet and slide, parting late",
        lambda rng, o: events.parry(o.weapon, rng),
        -9.5, "weapon"),
    "draw": Cue(
        "draw", "handling", "sword leaving a scabbard, scrape then zing",
        lambda rng, o: events.draw(o.weapon, rng),
        -11.0, "weapon"),
    "sheathe": Cue(
        "sheathe", "handling", "blade going back in, ending on the stop",
        lambda rng, o: events.sheathe(o.weapon, rng),
        -11.5, "weapon"),
    "rustle": Cue(
        "rustle", "handling", "mail moving",
        lambda rng, o: events.rustle(rng),
        -13.0, ""),
    "footstep": Cue(
        "footstep", "handling", "a boot landing",
        lambda rng, o: events.footstep(rng, o.surface, o.armored),
        -12.0, "surface, armored"),
    "body_fall": Cue(
        "body_fall", "handling", "a body going down, and its weapon after it",
        lambda rng, o: events.body_fall(rng, o.armored),
        -9.0, "armored"),
    "bow_draw": Cue(
        "bow_draw", "ranged", "limbs loading up, creak and stretch",
        lambda rng, o: ranged.bow_draw(rng),
        -13.0, ""),
    "bow_release": Cue(
        "bow_release", "ranged", "loose: string, limbs, arrow away",
        lambda rng, o: ranged.bow_release(rng),
        -9.5, ""),
    "arrow_flight": Cue(
        "arrow_flight", "ranged", "an arrow going past, fletching spinning",
        lambda rng, o: ranged.arrow_flight(rng),
        -12.0, ""),
    "arrow_hit_wood": Cue(
        "arrow_hit_wood", "ranged", "thunk into a post, shaft wobbling",
        lambda rng, o: ranged.arrow_hit_wood(rng),
        -9.5, ""),
    "arrow_hit_armor": Cue(
        "arrow_hit_armor", "ranged", "a deflection, bright and brief",
        lambda rng, o: ranged.arrow_hit_armor(rng),
        -9.5, ""),
    # There is no "punch" or "block" cue, and that is the design rather than an
    # omission. A punch is `swing` with a fist, a block is `strike` against a
    # guard, a kick is `swing` with a boot: unarmed combat is the same six
    # layers landing on the same targets, so it needed strikers and targets,
    # not a parallel set of cues.
    #
    # There is no "knockout" either, but that one is a deletion rather than a
    # design. It existed, it was listened to on 2026-07-15, and it was cut: it
    # was built on `strike` against the head, so it inherited the same
    # synthesized-flesh problem that moved every impact to
    # unarmed_foley_generator. Knockouts come from there now.
    "energy_charge": Cue(
        "energy_charge", "energy", "power gathering, ending unresolved",
        lambda rng, o: energy.energy_charge(rng),
        -11.0, ""),
    # Named for what it is rather than what it was first asked to be. This was
    # arcade-fighter's `fireball` until someone listened to it: it is too
    # mechanical to be one, and it is a phaser or laser blast. A fireball is
    # *fire* -- combustion, roar, no clean pitch to slide -- and belongs to the
    # dedicated fire provider. This cue does not pretend to be it.
    "energy_release": Cue(
        "energy_release", "energy",
        "a phaser or laser blast leaving; not a fireball",
        lambda rng, o: energy.energy_release(rng),
        -8.5, ""),
}

GROUPS = ("melee", "handling", "ranged", "energy")


def render(cue_name: str, opts: Options, rng: random.Random) -> list[float]:
    """Render one cue, mastered and ready to write."""
    cue = CUES[cue_name]
    return dsp.master(cue.render(rng, opts), cue.level(opts))


# --------------------------------------------------------------------------
# packs: named sets of artifacts for a specific consumer
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PackEntry:
    filename: str          # base name, no extension
    cue: str
    count: int
    opts: Options


@dataclass(frozen=True)
class Pack:
    name: str
    description: str
    entries: tuple[PackEntry, ...]
    # plain -> name_01.wav | v_suffix -> name.wav, name_v02.wav
    naming: str = "plain"

    def filename(self, entry: PackEntry, index: int) -> str:
        if self.naming == "v_suffix":
            # The house convention: board-duel, arcade-fighter and creature-rancher
            # all ship the same variant_filename() in their src/sound.c, and it
            # loads variants until one is missing. Do not drift from it.
            return (f"{entry.filename}.wav" if index == 0
                    else f"{entry.filename}_v{index + 1:02d}.wav")
        if entry.count == 1:
            return f"{entry.filename}.wav"
        return f"{entry.filename}_{index + 1:02d}.wav"


# Board Duel is a battle-chess game: a capture is two pieces fighting. The
# variant counts match the banks src/sound.c already loads, and the filenames
# match its naming, so this set is drop-in. capture_clank uses `strike` and not
# `full` on purpose: a swing first would delay the impact past the moment the
# player sees the capture land.
BOARD_DUEL = Pack(
    name="board-duel",
    description="drop-in bank for the board-duel combat cues",
    naming="v_suffix",
    entries=(
        PackEntry("capture_clank", "strike", 5, Options(weapon=SWORD,
                                                        target=BLADE)),
        PackEntry("fall_thud", "body_fall", 4, Options(armored=True)),
        PackEntry("move_step", "footstep", 6, Options(surface="stone",
                                                      armored=False)),
    ),
)

# Arcade Fighter fires these where its src/game.c says, which is what decides the
# cue behind each one. punch, kick and special play the moment the move starts
# (game.c:335-337), before anything has been touched -- so they are swings, not
# impacts, and a punch that opened with a thud would land before the fighter
# had moved. fireball is the projectile leaving.
#
# This pack makes the three cues that are *not* flesh impacts, and that split is
# a finding rather than a tidiness choice. A swing is air and cloth, and both
# are noise, which is what synthesis is good at; all three were signed off by
# ear on 2026-07-15. The impacts it used to ship -- hit, block, ko -- were
# listened to and rejected, and they now come from unarmed_foley_generator,
# which is built on real recordings. That generator ships the bank by borrowing
# these, so a drop-in bank is still one command; it is just not this one.
#
# `fireball` is absent, and it is absent rather than approximated. This pack
# filled it with energy_release until that was listened to: the cue is a good
# phaser blast and not a fireball, so shipping it under that name would have
# been the `hit` mistake again -- a rejected sound going out because the slot
# wanted filling. Fire comes from fire_sound_generator. menu, select, round and
# win are absent for a separate reason: they are UI cues and belong to
# ui_game_state_generator.
ARCADE_FIGHTER = Pack(
    name="arcade-fighter",
    description="arcade-fighter swings and the special charge (impacts: see "
                "unarmed_foley_generator; fireball: pending a fire generator)",
    naming="v_suffix",
    entries=(
        PackEntry("punch", "swing", 5, Options(weapon=FIST)),
        PackEntry("kick", "swing", 5, Options(weapon=BOOT)),
        PackEntry("special", "energy_charge", 3, Options()),
    ),
)

SHOWCASE = Pack(
    name="showcase",
    description="one of everything, for auditioning",
    entries=tuple(PackEntry(name, name, 1, Options()) for name in CUES),
)

PACKS = {p.name: p for p in (BOARD_DUEL, ARCADE_FIGHTER, SHOWCASE)}


def weapon_named(name: str) -> Weapon:
    if name not in WEAPONS:
        raise ValueError(f"unknown weapon {name!r}; "
                         f"expected one of {sorted(WEAPONS)}")
    return WEAPONS[name]


def target_named(name: str) -> Target:
    if name not in TARGETS:
        raise ValueError(f"unknown target {name!r}; "
                         f"expected one of {sorted(TARGETS)}")
    return TARGETS[name]
