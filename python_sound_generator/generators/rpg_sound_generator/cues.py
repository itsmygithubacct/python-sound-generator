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


def _production_renderer(kind: str, signature: int,
                         seconds: float) -> Renderer:
    """Build a deterministic, theme-aware cue for a production semantic."""

    def render(theme: Theme, rng: random.Random,
               intensity: float) -> list[float]:
        out = dsp.silence(seconds)
        base = 42 + signature % 25
        high = 70 + signature % 17
        weight = 0.70 + 0.30 * intensity

        if kind == "ui":
            notes = (base + 12, base + 19) if signature % 2 == 0 else (
                base + 19, base + 12
            )
            _add(out, 0.0, engine.arpeggio(
                notes, min(seconds, 0.34), theme, rng,
                gap=0.065, note_duration=min(0.18, seconds),
            ), 0.55 * weight)
            _add(out, 0.0, engine.click(
                min(seconds, 0.045), theme, rng,
                bright=signature % 3 != 0,
            ), 0.20)
        elif kind == "impact":
            _add(out, 0.0, engine.thump(
                min(seconds, 0.36), theme, rng,
            ), 0.55 * weight)
            _add(out, 0.0, engine.click(
                min(seconds, 0.10), theme, rng,
                bright=signature % 3 != 1,
            ), 0.28 * weight)
            _add(out, 0.01, engine.sweep(
                high, base, min(seconds, 0.30), theme, rng,
                decay=max(0.03, min(seconds, 0.30) * 0.42),
                curve=1.10 + signature % 4 * 0.18,
            ), 0.22)
        elif kind == "motion":
            rising = signature % 2 == 0
            start = base if rising else high + 7
            end = high + 7 if rising else base
            _add(out, 0.0, engine.whoosh(
                min(seconds * 0.82, 0.68),
                220.0 if rising else 4_200.0,
                4_200.0 if rising else 220.0,
                theme, rng,
            ), 0.34 * weight)
            _add(out, 0.0, engine.sweep(
                start, end, min(seconds * 0.72, 0.52), theme, rng,
                curve=0.70 if rising else 1.42,
            ), 0.34)
        elif kind == "alert":
            notes = tuple(
                high if index % 2 == 0 else high - 6 - signature % 3
                for index in range(3 + signature % 3)
            )
            _add(out, 0.0, engine.arpeggio(
                notes, min(seconds, 0.78), theme, rng,
                gap=0.095, note_duration=0.17,
            ), 0.53 * weight)
            _add(out, 0.0, engine.thump(
                min(seconds, 0.30), theme, rng,
            ), 0.24)
        elif kind == "positive":
            notes = (base, base + 4 + signature % 3, base + 7, base + 12)
            _add(out, 0.0, engine.arpeggio(
                notes, min(seconds, 0.90), theme, rng,
                gap=max(0.065, min(0.12, seconds / 7.0)),
                note_duration=min(0.34, seconds * 0.58),
            ), 0.53 * weight)
            _add(out, min(seconds * 0.28, 0.22), engine.sparkle(
                min(seconds * 0.58, 0.52), theme, rng,
                density=3 + signature % 6,
            ), 0.24)
        elif kind == "negative":
            _add(out, 0.0, engine.sweep(
                high + 5, base - 5, min(seconds * 0.76, 0.68),
                theme, rng, curve=1.28 + signature % 4 * 0.16,
            ), 0.54 * weight)
            _add(out, min(0.06, seconds * 0.16), engine.whoosh(
                min(seconds * 0.68, 0.52), 2_800.0, 130.0,
                theme, rng,
            ), 0.24)
        elif kind == "magic":
            rising = signature % 4 != 3
            _add(out, 0.0, engine.whoosh(
                min(seconds * 0.70, 0.58),
                180.0 if rising else 5_200.0,
                5_200.0 if rising else 180.0,
                theme, rng,
            ), 0.28 * weight)
            _add(out, 0.02, engine.sweep(
                base, high + signature % 7, min(seconds * 0.62, 0.50),
                theme, rng, curve=0.72 + signature % 3 * 0.18,
            ), 0.36)
            _add(out, min(seconds * 0.30, 0.22), engine.sparkle(
                min(seconds * 0.58, 0.48), theme, rng,
                density=4 + signature % 7,
            ), 0.30)
            if signature % 3 == 0:
                _add(out, min(seconds * 0.44, 0.31), engine.thump(
                    min(seconds * 0.34, 0.28), theme, rng,
                ), 0.26)
        elif kind == "vehicle":
            pulse_seconds = min(seconds, 0.24)
            for index in range(2 + signature % 3):
                _add(out, index * 0.12, engine.thump(
                    pulse_seconds, theme, rng,
                ), (0.24 + index * 0.04) * weight)
            _add(out, 0.0, engine.sweep(
                base - 6, base + 7 if signature % 2 == 0 else base - 12,
                min(seconds * 0.86, 0.72), theme, rng,
                curve=0.76 + signature % 4 * 0.20,
            ), 0.28)
        else:
            raise ValueError(f"unknown production renderer kind: {kind}")
        return out

    return render


GROUPS = (
    "interface", "inventory", "items", "battle", "magic", "progression",
    "summon", "world", "system",
)


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
        _cue("cursor_move", "interface", "move a cursor through a menu",
             _production_renderer("ui", 1, 0.10), -11.0, 0.10,
             "crystal cursor tick"),
        _cue("menu_confirm", "interface", "accept a menu command",
             _production_renderer("ui", 2, 0.24), -7.0, 0.24,
             "rising menu interval", "latch click"),
        _cue("menu_cancel", "interface", "back out of a menu command",
             _production_renderer("ui", 3, 0.24), -8.0, 0.24,
             "falling menu interval", "release click"),
        _cue("menu_error", "interface", "reject an unavailable command",
             _production_renderer("negative", 4, 0.34), -5.0, 0.34,
             "denial fall", "muted rush"),
        _cue("text_blip", "interface", "one repeatable dialogue character",
             _production_renderer("ui", 5, 0.09), -12.0, 0.09,
             "compact speech tone"),
        _cue("battle_start", "battle", "transition into active battle",
             _production_renderer("alert", 6, 0.82), -2.5, 0.82,
             "battle warning", "low strike"),
        _cue("atb_ready", "battle", "a party member is ready to act",
             _production_renderer("ui", 7, 0.30), -7.0, 0.30,
             "readiness chime", "turn marker"),
        _cue("attack_swing", "battle", "weapon or limb cuts through the air",
             _production_renderer("motion", 8, 0.30), -4.0, 0.30,
             "weapon air cut"),
        _cue("attack_hit", "battle", "ordinary physical attack connects",
             _production_renderer("impact", 9, 0.40), -2.0, 0.40,
             "contact crack", "body impact"),
        _cue("attack_miss", "battle", "physical attack misses its target",
             _production_renderer("motion", 10, 0.38), -5.0, 0.38,
             "empty swing", "miss fall"),
        _cue("critical_hit", "battle", "high-damage physical strike lands",
             _production_renderer("impact", 11, 0.68), -1.0, 0.68,
             "heavy crack", "critical collapse"),
        _cue("player_hurt", "battle", "party member takes damage",
             _production_renderer("impact", 12, 0.42), -2.5, 0.42,
             "body hit", "pain fall"),
        _cue("enemy_defeat", "battle", "one enemy is removed from battle",
             _production_renderer("negative", 13, 0.72), -3.0, 0.72,
             "enemy collapse", "spirit release"),
        _cue("defend", "battle", "party member braces against damage",
             _production_renderer("positive", 14, 0.48), -6.0, 0.48,
             "guard chord", "brace tick"),
        _cue("counter", "battle", "counterattack is triggered",
             _production_renderer("impact", 15, 0.50), -2.0, 0.50,
             "counter flash", "returning blow"),
        _cue("row_change", "battle", "character changes battle row",
             _production_renderer("motion", 16, 0.42), -6.0, 0.42,
             "formation sweep", "position chime"),
        _cue("run_success", "battle", "party escapes battle",
             _production_renderer("positive", 17, 0.70), -4.0, 0.70,
             "escape ascent", "release sparkle"),
        _cue("run_failure", "battle", "escape attempt fails",
             _production_renderer("negative", 18, 0.58), -3.5, 0.58,
             "blocked retreat", "failure rush"),
        _cue("victory", "battle", "battle ends in victory",
             _production_renderer("positive", 19, 1.18), -2.0, 1.18,
             "victory cadence", "reward sparkle"),
        _cue("defeat", "battle", "party is defeated",
             _production_renderer("negative", 20, 1.16), -2.0, 1.16,
             "party collapse", "lost cadence"),
        _cue("magic_cast", "magic", "generic spell energy is released",
             _production_renderer("magic", 21, 0.72), -3.0, 0.72,
             "casting rise", "arcane release"),
        _cue("fire_spell", "magic", "fire-element spell impacts",
             _production_renderer("magic", 22, 0.84), -1.5, 0.84,
             "heated rush", "flame burst"),
        _cue("ice_spell", "magic", "ice-element spell impacts",
             _production_renderer("magic", 23, 0.88), -2.0, 0.88,
             "frozen shimmer", "ice fracture"),
        _cue("lightning_spell", "magic", "lightning-element spell impacts",
             _production_renderer("magic", 24, 0.72), -1.0, 0.72,
             "electric rise", "lightning snap"),
        _cue("water_spell", "magic", "water-element spell flows and impacts",
             _production_renderer("magic", 25, 0.92), -2.5, 0.92,
             "liquid sweep", "water strike"),
        _cue("wind_spell", "magic", "wind-element spell cuts through targets",
             _production_renderer("magic", 26, 0.76), -2.0, 0.76,
             "air spiral", "wind cut"),
        _cue("earth_spell", "magic", "earth-element spell impacts",
             _production_renderer("magic", 27, 0.94), -1.0, 0.94,
             "stone rise", "ground impact"),
        _cue("light_spell", "magic", "holy or light spell resolves",
             _production_renderer("magic", 28, 0.92), -2.5, 0.92,
             "radiant rise", "light crown"),
        _cue("dark_spell", "magic", "shadow or dark spell resolves",
             _production_renderer("magic", 29, 0.98), -2.0, 0.98,
             "void fall", "dark pressure"),
        _cue("heal_spell", "magic", "restore hit points with magic",
             _production_renderer("positive", 30, 0.88), -4.5, 0.88,
             "restoration ascent", "healing motes"),
        _cue("revive_spell", "magic", "restore an incapacitated ally",
             _production_renderer("positive", 31, 1.08), -3.0, 1.08,
             "life return", "revival crown"),
        _cue("buff_spell", "magic", "beneficial status effect is applied",
             _production_renderer("positive", 32, 0.76), -4.0, 0.76,
             "status ascent", "benefit sparkle"),
        _cue("debuff_spell", "magic", "harmful status effect is applied",
             _production_renderer("negative", 33, 0.78), -3.0, 0.78,
             "status fall", "affliction rush"),
        _cue("reflect_spell", "magic", "reflective magic barrier engages",
             _production_renderer("magic", 34, 0.82), -3.5, 0.82,
             "mirror rise", "barrier ring"),
        _cue("summon_charge", "summon", "summoning ritual gathers power",
             _production_renderer("magic", 35, 1.12), -2.0, 1.12,
             "ritual pressure", "summon ascent"),
        _cue("summon_appear", "summon", "summoned ally materializes",
             _production_renderer("magic", 36, 1.22), -1.5, 1.22,
             "dimensional opening", "arrival crown"),
        _cue("summon_impact", "summon", "summoned ally completes its attack",
             _production_renderer("impact", 37, 1.06), -0.8, 1.06,
             "summon strike", "massive impact"),
        _cue("magic_learn", "summon", "new spell or summon is learned",
             _production_renderer("positive", 38, 1.12), -3.0, 1.12,
             "knowledge ascent", "learned flourish"),
        _cue("door_open", "world", "open an ordinary door",
             _production_renderer("motion", 39, 0.54), -7.0, 0.54,
             "hinge motion", "latch release"),
        _cue("switch_toggle", "world", "toggle a dungeon or field mechanism",
             _production_renderer("ui", 40, 0.32), -6.0, 0.32,
             "mechanism click", "state chime"),
        _cue("puzzle_solve", "world", "complete a field puzzle",
             _production_renderer("positive", 41, 0.94), -3.5, 0.94,
             "solution cadence", "discovery sparkle"),
        _cue("vehicle_board", "world", "board a world-travel vehicle",
             _production_renderer("vehicle", 42, 0.74), -5.0, 0.74,
             "vehicle start", "boarding signal"),
        _cue("vehicle_move", "world", "vehicle travels through the world",
             _production_renderer("vehicle", 43, 0.88), -7.0, 0.88,
             "engine pulse", "travel motion"),
        _cue("vehicle_land", "world", "airship or vehicle arrives",
             _production_renderer("vehicle", 44, 0.92), -4.5, 0.92,
             "engine settle", "arrival impact"),
        _cue("world_shift", "world", "major world state changes",
             _production_renderer("alert", 45, 1.08), -2.5, 1.08,
             "world warning", "transformation chord"),
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
