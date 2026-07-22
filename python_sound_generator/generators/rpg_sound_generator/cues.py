"""Cue, theme, and pack registries for the RPG sound library."""

from __future__ import annotations

from ..._audio.models import RenderResult, VariationFilenames
from ..._audio.validation import bounded_float

import random
from dataclasses import dataclass
from typing import Callable

from ..._audio import dsp
from . import engine
from .engine import THEMES, Theme


Renderer = Callable[[Theme, random.Random, float], list[float]]


@dataclass(frozen=True, slots=True)
class Options:
    """Render controls shared by every RPG cue."""

    theme: Theme = THEMES["fantasy"]
    intensity: float = 0.85
    foley: float = 0.0


@dataclass(frozen=True, slots=True)
class Cue:
    name: str
    group: str
    description: str
    renderer: Renderer
    peak_db: float
    seconds: float
    layers: tuple[str, ...]
    foley_kinds: tuple[str, ...] = ()


def _add(out: list[float], start: float, signal: list[float],
         gain: float = 1.0) -> None:
    dsp.add_at(out, start, signal, gain)


# Inventory -----------------------------------------------------------------

def _inventory_open(theme: Theme, rng: random.Random,
                    intensity: float) -> list[float]:
    out = dsp.silence(0.48)
    _add(out, 0.00, engine.click(0.07, theme, rng), 0.28)
    _add(out, 0.01, engine.whoosh(0.30, 280.0, 1_900.0, theme, rng),
         0.55 + 0.25 * intensity)
    _add(out, 0.08, engine.arpeggio((60, 67), 0.34, theme, rng,
                                    gap=0.10, note_duration=0.28), 0.55)
    return out


def _inventory_close(theme: Theme, rng: random.Random,
                     intensity: float) -> list[float]:
    out = dsp.silence(0.42)
    _add(out, 0.00, engine.arpeggio((67, 60), 0.30, theme, rng,
                                    gap=0.09, note_duration=0.24), 0.50)
    _add(out, 0.02, engine.whoosh(0.28, 1_700.0, 240.0, theme, rng),
         0.45 + 0.20 * intensity)
    _add(out, 0.25, engine.click(0.06, theme, rng, bright=False), 0.35)
    return out


def _item_pickup(theme: Theme, rng: random.Random,
                 intensity: float) -> list[float]:
    out = engine.arpeggio((72, 76, 79), 0.38, theme, rng,
                          gap=0.065, note_duration=0.25)
    _add(out, 0.015, engine.click(0.05, theme, rng), 0.24)
    _add(out, 0.04, engine.sparkle(0.25, theme, rng,
                                   density=3 + round(4 * intensity)), 0.30)
    return out


def _item_drop(theme: Theme, rng: random.Random,
               intensity: float) -> list[float]:
    out = dsp.silence(0.42)
    _add(out, 0.00, engine.sweep(72, 48, 0.26, theme, rng,
                                 decay=0.18, curve=1.25), 0.45)
    _add(out, 0.19, engine.thump(0.20, theme, rng),
         0.55 + 0.35 * intensity)
    return out


def _item_equip(theme: Theme, rng: random.Random,
                intensity: float) -> list[float]:
    out = dsp.silence(0.52)
    _add(out, 0.00, engine.click(0.055, theme, rng), 0.42)
    _add(out, 0.015, engine.chime(57, 0.42, theme, rng),
         0.65 + 0.20 * intensity)
    _add(out, 0.13, engine.chime(64, 0.30, theme, rng), 0.34)
    _add(out, 0.02, engine.thump(0.16, theme, rng), 0.22)
    return out


# Items and economy ----------------------------------------------------------

def _coin_pickup(theme: Theme, rng: random.Random,
                 intensity: float) -> list[float]:
    out = dsp.silence(0.44)
    _add(out, 0.00, engine.chime(88, 0.36, theme, rng), 0.70)
    _add(out, 0.065, engine.chime(95, 0.30, theme, rng),
         0.45 + 0.20 * intensity)
    _add(out, 0.00, engine.click(0.035, theme, rng), 0.20)
    return out


def _purchase(theme: Theme, rng: random.Random,
              intensity: float) -> list[float]:
    out = dsp.silence(0.62)
    _add(out, 0.00, engine.chime(84, 0.28, theme, rng), 0.38)
    _add(out, 0.075, engine.chime(88, 0.30, theme, rng), 0.45)
    _add(out, 0.18, engine.arpeggio((67, 72), 0.38, theme, rng,
                                    gap=0.10, note_duration=0.30),
         0.48 + 0.16 * intensity)
    return out


def _potion_drink(theme: Theme, rng: random.Random,
                  intensity: float) -> list[float]:
    out = dsp.silence(0.92)
    # Cork pop.
    _add(out, 0.00, engine.click(0.08, theme, rng, bright=False), 0.46)
    _add(out, 0.00, engine.thump(0.11, theme, rng), 0.27)
    # Three falling bubble chirps make the liquid action readable.
    for index, onset in enumerate((0.13, 0.25, 0.38)):
        _add(out, onset, engine.sweep(75 - index * 2, 64 - index * 2,
                                     0.16, theme, rng, decay=0.08),
             0.28 + 0.08 * intensity)
    # A quiet restorative finish, distinct from the drink itself.
    _add(out, 0.50, engine.arpeggio((60, 64, 67), 0.38, theme, rng,
                                    gap=0.075, note_duration=0.28), 0.34)
    return out


def _chest_open(theme: Theme, rng: random.Random,
                intensity: float) -> list[float]:
    out = dsp.silence(1.18)
    _add(out, 0.00, engine.click(0.09, theme, rng, bright=False), 0.55)
    _add(out, 0.07, engine.creak(0.45, theme, rng),
         0.34 + 0.18 * intensity)
    _add(out, 0.34, engine.arpeggio((60, 64, 67, 72), 0.72, theme, rng,
                                    gap=0.12, note_duration=0.48), 0.58)
    _add(out, 0.40, engine.sparkle(0.62, theme, rng,
                                   density=6 + round(6 * intensity)), 0.36)
    return out


def _door_unlock(theme: Theme, rng: random.Random,
                 intensity: float) -> list[float]:
    out = dsp.silence(0.58)
    for index, onset in enumerate((0.00, 0.065, 0.14)):
        _add(out, onset, engine.click(0.055, theme, rng,
                                      bright=index == 2), 0.32 + index * 0.08)
    _add(out, 0.20, engine.sweep(55, 67, 0.28, theme, rng, decay=0.19),
         0.42 + 0.16 * intensity)
    return out


# Progression ----------------------------------------------------------------

def _experience_gain(theme: Theme, rng: random.Random,
                     intensity: float) -> list[float]:
    out = engine.arpeggio((69, 72), 0.34, theme, rng,
                          gap=0.075, note_duration=0.26)
    _add(out, 0.04, engine.sparkle(0.24, theme, rng,
                                   density=2 + round(3 * intensity)), 0.24)
    return out


def _quest_accept(theme: Theme, rng: random.Random,
                  intensity: float) -> list[float]:
    out = dsp.silence(0.82)
    _add(out, 0.00, engine.arpeggio((55, 60, 64), 0.64, theme, rng,
                                    gap=0.14, note_duration=0.42), 0.58)
    _add(out, 0.31, engine.chime(67, 0.42, theme, rng),
         0.35 + 0.12 * intensity)
    return out


def _quest_complete(theme: Theme, rng: random.Random,
                    intensity: float) -> list[float]:
    out = dsp.silence(1.22)
    _add(out, 0.00, engine.arpeggio((60, 64, 67, 72), 0.92, theme, rng,
                                    gap=0.13, note_duration=0.58), 0.58)
    for note in (60, 64, 67, 72):
        _add(out, 0.48, engine.chime(note, 0.65, theme, rng), 0.18)
    _add(out, 0.25, engine.sparkle(0.78, theme, rng,
                                   density=7 + round(8 * intensity)), 0.36)
    return out


def _level_up(theme: Theme, rng: random.Random,
              intensity: float) -> list[float]:
    out = dsp.silence(1.36)
    _add(out, 0.00, engine.arpeggio((60, 62, 64, 67, 72), 0.98, theme, rng,
                                    gap=0.095, note_duration=0.58), 0.64)
    for note in (60, 64, 67, 72):
        _add(out, 0.48, engine.chime(note, 0.76, theme, rng), 0.19)
    _add(out, 0.16, engine.whoosh(0.52, 360.0, 4_200.0, theme, rng),
         0.24 + 0.22 * intensity)
    _add(out, 0.30, engine.sparkle(0.82, theme, rng,
                                   density=8 + round(10 * intensity)), 0.40)
    return out


# World and system -----------------------------------------------------------

def _dialogue_open(theme: Theme, rng: random.Random,
                   intensity: float) -> list[float]:
    out = dsp.silence(0.36)
    # A page-like flutter, then one neutral voice marker. No reward interval.
    flutter = engine.whoosh(0.20, 1_900.0, 850.0, theme, rng)
    _add(out, 0.00, flutter, 0.30 + 0.14 * intensity)
    _add(out, 0.07, engine.tone(67, 0.22, theme, rng,
                                decay=0.085, attack=0.001), 0.42)
    return out


def _secret_found(theme: Theme, rng: random.Random,
                  intensity: float) -> list[float]:
    out = dsp.silence(1.04)
    # An asymmetric original motif so it reads as discovery, not generic gain.
    _add(out, 0.00, engine.arpeggio((60, 63, 69, 77), 0.76, theme, rng,
                                    gap=0.115, note_duration=0.48), 0.60)
    _add(out, 0.34, engine.sparkle(0.58, theme, rng,
                                   density=5 + round(7 * intensity)), 0.38)
    return out


def _encounter(theme: Theme, rng: random.Random,
               intensity: float) -> list[float]:
    out = dsp.silence(0.82)
    _add(out, 0.00, engine.thump(0.48, theme, rng),
         0.58 + 0.30 * intensity)
    _add(out, 0.04, engine.sweep(48, 42, 0.52, theme, rng,
                                 decay=0.35, curve=0.75), 0.52)
    _add(out, 0.16, engine.chime(54, 0.50, theme, rng), 0.30)
    _add(out, 0.16, engine.chime(60, 0.50, theme, rng), 0.25)
    return out


def _save_game(theme: Theme, rng: random.Random,
               intensity: float) -> list[float]:
    out = dsp.silence(0.94)
    _add(out, 0.00, engine.arpeggio((72, 67, 64, 60), 0.76, theme, rng,
                                    gap=0.10, note_duration=0.42), 0.48)
    _add(out, 0.34, engine.chime(60, 0.48, theme, rng),
         0.28 + 0.10 * intensity)
    return out


def _rest(theme: Theme, rng: random.Random,
          intensity: float) -> list[float]:
    out = dsp.silence(1.34)
    _add(out, 0.00, engine.arpeggio((72, 69, 65, 60), 1.12, theme, rng,
                                    gap=0.18, note_duration=0.64), 0.42)
    _add(out, 0.18, engine.whoosh(0.72, 1_200.0, 180.0, theme, rng),
         0.18 + 0.10 * intensity)
    return out


GROUPS = ("inventory", "items", "progression", "world", "system")


def _cue(name: str, group: str, description: str, renderer: Renderer,
         peak_db: float, seconds: float, *layers: str,
         foley_kinds: tuple[str, ...] = ()) -> Cue:
    return Cue(
        name,
        group,
        description,
        renderer,
        peak_db,
        seconds,
        layers,
        foley_kinds,
    )


CUES: dict[str, Cue] = {
    cue.name: cue
    for cue in (
        _cue("inventory_open", "inventory", "open an inventory or character sheet",
             _inventory_open, -5.0, 0.48, "latch", "rising page sweep", "open chord"),
        _cue("inventory_close", "inventory", "close an inventory or character sheet",
             _inventory_close, -6.0, 0.42, "falling chord", "closing page sweep", "latch"),
        _cue("item_pickup", "inventory", "collect a general inventory item",
             _item_pickup, -4.0, 0.38, "ascending chimes", "pickup sparkle"),
        _cue("item_drop", "inventory", "drop or discard an item",
             _item_drop, -4.0, 0.42, "falling glide", "soft impact",
             foley_kinds=("material contact",)),
        _cue("item_equip", "inventory", "equip a weapon, tool, or armor item",
             _item_equip, -3.5, 0.52, "fastener click", "material ring", "body",
             foley_kinds=("cloth or carried gear",)),
        _cue("coin_pickup", "items", "collect a coin or small currency reward",
             _coin_pickup, -4.0, 0.44, "two bright coin rings", "contact tick",
             foley_kinds=("coin into bag",)),
        _cue("purchase", "items", "complete a shop transaction",
             _purchase, -4.5, 0.62, "coin pair", "confirmation cadence",
             foley_kinds=("coin transaction",)),
        _cue("potion_drink", "items", "uncork and drink a restorative potion",
             _potion_drink, -5.0, 0.92, "cork pop", "bubble chirps", "restoration chord",
             foley_kinds=("cork", "swallow")),
        _cue("chest_open", "items", "unlock and reveal a treasure chest",
             _chest_open, -3.5, 1.18, "latch", "wood creak", "reward fanfare", "sparkle",
             foley_kinds=("wooden chest",)),
        _cue("door_unlock", "items", "turn a lock and confirm an unlocked route",
             _door_unlock, -5.0, 0.58, "mechanism clicks", "rising confirmation",
             foley_kinds=("lock mechanism",)),
        _cue("experience_gain", "progression", "gain a small amount of experience",
             _experience_gain, -6.0, 0.34, "short rising pair", "light motes"),
        _cue("quest_accept", "progression", "accept a new objective",
             _quest_accept, -5.0, 0.82, "grounded cadence", "objective marker"),
        _cue("quest_complete", "progression", "complete an objective and claim its reward",
             _quest_complete, -3.5, 1.22, "ascending fanfare", "final chord", "sparkle"),
        _cue("level_up", "progression", "advance a character level",
             _level_up, -2.5, 1.36, "scale ascent", "rising air", "final chord", "motes"),
        _cue("dialogue_open", "world", "begin a conversation or show a dialogue box",
             _dialogue_open, -7.0, 0.36, "page flutter", "neutral voice marker"),
        _cue("secret_found", "world", "reveal a hidden room, object, or route",
             _secret_found, -4.0, 1.04, "discovery motif", "sparkle trail"),
        _cue("encounter", "world", "signal a dangerous encounter",
             _encounter, -2.5, 0.82, "low impact", "falling tension", "dissonant stab"),
        _cue("save_game", "system", "confirm that game progress was saved",
             _save_game, -6.0, 0.94, "settling cadence", "tonic ring"),
        _cue("rest", "system", "rest, camp, or pass time safely",
             _rest, -8.0, 1.34, "descending lullaby", "soft exhale"),
    )
}


def _render_result(cue_name: str, options: Options,
                   rng: random.Random) -> RenderResult[list[float]]:
    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    if options.theme.name not in THEMES or THEMES[options.theme.name] != options.theme:
        raise ValueError(f"unregistered theme {options.theme.name!r}")
    bounded_float(options.intensity, "intensity", 0.05, 1.0)
    bounded_float(options.foley, "foley", 0.0, 1.0)

    signal = cue.renderer(options.theme, rng, options.intensity)
    sources: tuple[str, ...] = ()
    if options.foley > 0.0 and cue.foley_kinds:
        from . import foley

        signal, sources = foley.apply(
            cue_name,
            signal,
            options.theme,
            options.intensity,
            options.foley,
            rng,
        )
    return RenderResult(engine.master(signal, cue.peak_db), sources)


def render(cue_name: str, options: Options,
           rng: random.Random) -> list[float]:
    """Render and master one named cue, preserving the historical return type."""

    return _render_result(cue_name, options, rng).samples


def render_with_sources(cue_name: str, options: Options,
                        rng: random.Random) -> RenderResult[list[float]]:
    """Render one cue together with every logical recording it consumed."""

    return _render_result(cue_name, options, rng)


def generate(cue_name: str, *, theme: str = "fantasy",
             intensity: float = 0.85, foley_mix: float = 0.0,
             seed: int = 20260721) -> list[float]:
    """Convenience API returning deterministic mono float samples."""

    options = Options(
        theme=engine.theme_named(theme),
        intensity=intensity,
        foley=foley_mix,
    )
    return render(cue_name, options, random.Random(seed))


def generate_with_sources(
    cue_name: str,
    *,
    theme: str = "fantasy",
    intensity: float = 0.85,
    foley_mix: float = 0.0,
    seed: int = 20260721,
) -> RenderResult[list[float]]:
    """Return samples and logical Foley sources for one deterministic cue."""

    options = Options(
        theme=engine.theme_named(theme),
        intensity=intensity,
        foley=foley_mix,
    )
    return render_with_sources(cue_name, options, random.Random(seed))


def stem_for(cue_name: str, options: Options) -> str:
    if cue_name not in CUES:
        raise ValueError(f"unknown cue {cue_name!r}")
    return f"{cue_name}_{options.theme.name}"


@dataclass(frozen=True, slots=True)
class PackEntry:
    filename: str
    cue: str
    count: int = 1


@dataclass(frozen=True, slots=True)
class Pack(VariationFilenames):
    name: str
    description: str
    entries: tuple[PackEntry, ...]



CORE = Pack(
    "core",
    "one organised render of every RPG cue",
    tuple(PackEntry(f"{cue.group}/{cue.name}", cue.name) for cue in CUES.values()),
)

INVENTORY = Pack(
    "inventory",
    "three variations of inventory, loot, equipment, and shop interactions",
    tuple(
        PackEntry(name, name, 3)
        for name in (
            "inventory_open",
            "inventory_close",
            "item_pickup",
            "item_drop",
            "item_equip",
            "coin_pickup",
            "purchase",
            "potion_drink",
            "chest_open",
            "door_unlock",
        )
    ),
)

PROGRESSION = Pack(
    "progression",
    "two variations of quest, experience, level, secret, and save feedback",
    tuple(
        PackEntry(name, name, 2)
        for name in (
            "experience_gain",
            "quest_accept",
            "quest_complete",
            "level_up",
            "secret_found",
            "save_game",
        )
    ),
)

PACKS = {pack.name: pack for pack in (CORE, INVENTORY, PROGRESSION)}


__all__ = [
    "CUES",
    "GROUPS",
    "PACKS",
    "Cue",
    "Options",
    "Pack",
    "PackEntry",
    "generate",
    "generate_with_sources",
    "render",
    "render_with_sources",
    "stem_for",
]
