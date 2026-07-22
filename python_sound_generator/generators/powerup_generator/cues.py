"""The cue registry: every sound this project can render, by name.

A cue is a named renderer plus the level it masters to. Everything the CLI can
produce is in CUES or PACKS, so `list` is generated from the same tables the
renderer uses and cannot drift out of date.

**Pitch choices are deliberate and are the design.** These sounds communicate
through interval, not timbre:

- Rising perfect intervals (4th, 5th, octave) read as *gain*. Every pickup uses
  one.
- Major thirds stacked into an arpeggio read as *achievement*; that's the
  level-up and quest families.
- A falling minor 2nd is the most reliably "wrong"-sounding move in Western
  music; both denial cues use it.
- The negative cues are the positive gestures inverted, so a player hears
  "that's the pickup sound, backwards" without being told.

Everything is tuned around C major so the whole bank can be layered under music
without clashing. Transposing the lot is a matter of adding a constant to the
MIDI numbers below.
"""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import engine as e
from . import gestures as g
from .styles import Style

# MIDI reference: C4 = 60 (middle C), C5 = 72, C6 = 84.
C4, D4, E4, G4, A4, B4 = 60, 62, 64, 67, 69, 71
C5, D5, E5, F5, G5, A5, B5 = 72, 74, 76, 77, 79, 81, 83
C6, CS6, D6, E6, F6, G6, A6, B6 = 84, 85, 86, 88, 89, 91, 93, 95
C7, E7 = 96, 100


@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Callable[[Style, np.random.Generator], np.ndarray]
    peak_db: float
    honours: Honours | str | tuple[str, ...] = "style, seed"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


# --------------------------------------------------------------------------
# pickups — short, bright, high frequency of use
# --------------------------------------------------------------------------

def _coin(style: Style, rng: np.random.Generator) -> np.ndarray:
    """Two notes up a perfect fifth. The canonical pickup gesture."""
    out = g.arpeggio(style, [C6, G6], rng, note_dur=0.055, gap=0.048,
                     brightness=1.15)
    return e.mix(out, g.sparkle_tail(style, len(out) / e.SR, rng, 6_000.0))


def _gem(style: Style, rng: np.random.Generator) -> np.ndarray:
    """A crystalline major triad, up an octave — brighter and finer than a coin."""
    out = g.arpeggio(style, [E6, B6, E7], rng, note_dur=0.045,
                     gap=0.040, brightness=1.35, accel=0.88)
    return e.mix(out, g.sparkle_tail(style, len(out) / e.SR, rng, 7_500.0,
                                     15_000.0))


def _ammo(style: Style, rng: np.random.Generator) -> np.ndarray:
    """Low, dry, mechanical. A fourth up, no sparkle: this is a tool, not a prize."""
    out = g.arpeggio(style, [G4, C5], rng, note_dur=0.042, gap=0.038,
                     brightness=0.55)
    accent = g.accent_at(style, 0.05, rng, brightness=0.6)
    mixed = e.silence(len(out) / e.SR)
    e.add_at(mixed, 0.0, out)
    e.add_at(mixed, 0.0, accent, 0.8)
    return mixed


def _key(style: Style, rng: np.random.Generator) -> np.ndarray:
    """Metallic and distinctive — a fifth up, then the octave, with a jingle."""
    out = g.arpeggio(style, [A5, E6, A6], rng, note_dur=0.050, gap=0.044,
                     brightness=1.25, accel=0.85)
    jingle = g.sparkle_tail(style, len(out) / e.SR, rng, 5_000.0, 12_000.0)
    return e.mix(out, jingle * 1.4)


def _health(style: Style, rng: np.random.Generator) -> np.ndarray:
    """Warm and gentle: a major triad rising, slower, no hard transient."""
    return g.arpeggio(style, [E5, G5, C6], rng, note_dur=0.075, gap=0.070,
                      brightness=0.75)


# --------------------------------------------------------------------------
# powerups — state changes, longer, with the expiry pair most banks forget
# --------------------------------------------------------------------------

def _powerup_activate(style: Style, rng: np.random.Generator) -> np.ndarray:
    """Rising major arpeggio into an octave — something has switched on."""
    out = g.arpeggio(style, [C5, E5, G5, C6], rng, note_dur=0.070, gap=0.062,
                     brightness=1.1, accel=0.92)
    swell = g.glide(style, C4, C5, len(out) / e.SR * 0.6, rng, brightness=0.5)
    mixed = e.silence(len(out) / e.SR)
    e.add_at(mixed, 0.0, out)
    e.add_at(mixed, 0.0, swell, 0.30)
    return e.mix(mixed, g.sparkle_tail(style, len(mixed) / e.SR, rng))


def _powerup_warning(style: Style, rng: np.random.Generator) -> np.ndarray:
    """Three urgent pulses. Your powerup is about to run out."""
    return g.pulse_train(style, C6, 3, rng, pulse_dur=0.045, period=0.125,
                         brightness=1.0)


def _powerup_expire(style: Style, rng: np.random.Generator) -> np.ndarray:
    """The activate arpeggio, inverted and dulled. It ran out."""
    return g.arpeggio(style, [C6, G5, E5, C5], rng, note_dur=0.065, gap=0.058,
                      brightness=0.55, accel=1.08)


def _shield_up(style: Style, rng: np.random.Generator) -> np.ndarray:
    """A rising two-octave sweep with shimmer settling on top."""
    sweep = g.glide(style, C4, C6, 0.42, rng, brightness=1.0, curve=0.7,
                    with_body=True)
    out = e.silence(0.52)
    e.add_at(out, 0.0, sweep)
    e.add_at(out, 0.30, g.blip(style, C6, 0.10, rng, brightness=1.2), 0.55)
    return e.mix(out, g.sparkle_tail(style, 0.52, rng, 4_000.0))


def _shield_down(style: Style, rng: np.random.Generator) -> np.ndarray:
    """The same sweep falling, landing on a dull thud."""
    sweep = g.glide(style, C6, C4, 0.38, rng, brightness=0.6, curve=1.4,
                    with_body=True)
    out = e.silence(0.50)
    e.add_at(out, 0.0, sweep)
    thud = e.osc("sine", e.midi_to_freq(C4 - 12), 0.16) * e.exp_env(0.16, 0.05,
                                                                    0.002)
    e.add_at(out, 0.33, thud, 0.5)
    return out


# --------------------------------------------------------------------------
# progression — the fanfares; longest and most musical
# --------------------------------------------------------------------------

def _level_up(style: Style, rng: np.random.Generator) -> np.ndarray:
    """Major arpeggio over two octaves, accelerating into the top note."""
    out = g.arpeggio(style, [C5, E5, G5, C6, E6], rng, note_dur=0.085,
                     gap=0.078, brightness=1.15, accel=0.88)
    return e.mix(out, g.sparkle_tail(style, len(out) / e.SR, rng))


def _quest_complete(style: Style, rng: np.random.Generator) -> np.ndarray:
    """A real fanfare: rising run, then the tonic triad struck together."""
    run = g.arpeggio(style, [G5, C6, E6], rng, note_dur=0.090, gap=0.085,
                     brightness=1.1)
    hit = g.chord(style, [C6, E6, G6, C7], 0.55, rng, brightness=1.2)
    total = 0.085 * 3 + 0.62
    out = e.silence(total)
    e.add_at(out, 0.0, run, 0.85)
    e.add_at(out, 0.085 * 2.6, hit, 1.0)
    return e.mix(out, g.sparkle_tail(style, total, rng))


def _secret_found(style: Style, rng: np.random.Generator) -> np.ndarray:
    """The four-note discovery chime — the one everyone knows the shape of."""
    out = g.arpeggio(style, [A5, CS6, E6, A6], rng, note_dur=0.105,
                     gap=0.098, brightness=1.2, accel=0.95)
    return e.mix(out, g.sparkle_tail(style, len(out) / e.SR, rng, 5_000.0,
                                     14_000.0))


def _extra_life(style: Style, rng: np.random.Generator) -> np.ndarray:
    """A fast scalar run upward — celebratory, busy, unmistakably good news."""
    out = g.arpeggio(style, [C5, D5, E5, G5, A5, C6], rng, note_dur=0.062,
                     gap=0.056, brightness=1.2, accel=0.93)
    return e.mix(out, g.sparkle_tail(style, len(out) / e.SR, rng))


# --------------------------------------------------------------------------
# negative — the inversions
# --------------------------------------------------------------------------

def _denied(style: Style, rng: np.random.Generator) -> np.ndarray:
    """A falling minor second. Short, buzzy, and unambiguously 'no'."""
    return g.arpeggio(style, [B4, B4 - 1], rng, note_dur=0.070, gap=0.062,
                      brightness=0.45, with_body=True)


def _no_ammo(style: Style, rng: np.random.Generator) -> np.ndarray:
    """A dry mechanical click. No pitch to speak of: nothing happened."""
    out = e.silence(0.09)
    e.add_at(out, 0.0, g.accent_at(style, 0.09, rng, brightness=0.5), 1.0)
    e.add_at(out, 0.0, g.noise_crack(0.09, rng, 1_200.0, 5_000.0, decay=0.012),
             0.55)
    return out


def _item_broken(style: Style, rng: np.random.Generator) -> np.ndarray:
    """A falling third with the sound of something cracking through it."""
    fall = g.arpeggio(style, [E5, C5, A4], rng, note_dur=0.070, gap=0.060,
                      brightness=0.5, accel=1.15)
    total = len(fall) / e.SR
    out = e.silence(total)
    e.add_at(out, 0.0, fall)
    e.add_at(out, 0.0, g.noise_crack(min(0.22, total), rng, 900.0, 7_000.0), 0.5)
    return out


def _curse(style: Style, rng: np.random.Generator) -> np.ndarray:
    """Low, slow, detuned descent. The only cue that is meant to feel bad."""
    low = g.glide(style, A4 - 12, D4 - 12, 0.85, rng, brightness=0.35,
                  curve=1.5, with_body=True)
    # A second voice a semitone off, beating against the first.
    sour = g.glide(style, A4 - 12.35, D4 - 12.35, 0.85, rng, brightness=0.3,
                   curve=1.5)
    out = e.silence(0.95)
    e.add_at(out, 0.0, low, 1.0)
    e.add_at(out, 0.0, sour, 0.55)
    return out


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

CUES: dict[str, Cue] = {
    # pickups
    "coin": Cue("coin", "pickup", "two notes up a fifth; the canonical pickup",
                _coin, -9.0),
    "gem": Cue("gem", "pickup", "crystalline triad, an octave up, glittering",
               _gem, -9.5),
    "ammo": Cue("ammo", "pickup", "low dry fourth; a tool, not a prize",
                _ammo, -11.0),
    "key": Cue("key", "pickup", "metallic fifth-then-octave with a jingle",
               _key, -9.5),
    "health": Cue("health", "pickup", "warm major triad rising, no hard onset",
                  _health, -10.0),
    # powerups
    "powerup_activate": Cue("powerup_activate", "powerup",
                            "rising major arpeggio into the octave; switched on",
                            _powerup_activate, -8.5),
    "powerup_warning": Cue("powerup_warning", "powerup",
                           "three urgent pulses; about to run out",
                           _powerup_warning, -10.0),
    "powerup_expire": Cue("powerup_expire", "powerup",
                          "the activate arpeggio inverted and dulled",
                          _powerup_expire, -11.0),
    "shield_up": Cue("shield_up", "powerup",
                     "two-octave rising sweep settling into shimmer",
                     _shield_up, -9.0),
    "shield_down": Cue("shield_down", "powerup",
                       "the sweep falling onto a dull thud", _shield_down,
                       -10.0),
    # progression
    "level_up": Cue("level_up", "progression",
                    "major arpeggio over two octaves, accelerating", _level_up,
                    -8.0),
    "quest_complete": Cue("quest_complete", "progression",
                          "rising run into a struck tonic triad; a fanfare",
                          _quest_complete, -8.0),
    "secret_found": Cue("secret_found", "progression",
                        "the four-note discovery chime", _secret_found, -8.5),
    "extra_life": Cue("extra_life", "progression",
                      "fast scalar run upward; busy and celebratory",
                      _extra_life, -8.5),
    # negative
    "denied": Cue("denied", "negative", "falling minor second; buzzy and 'no'",
                  _denied, -11.0),
    "no_ammo": Cue("no_ammo", "negative", "dry mechanical click, no pitch",
                   _no_ammo, -13.0),
    "item_broken": Cue("item_broken", "negative",
                       "falling third with a crack through it", _item_broken,
                       -10.5),
    "curse": Cue("curse", "negative", "low detuned descent; meant to feel bad",
                 _curse, -10.5),
}

GROUPS = ("pickup", "powerup", "progression", "negative")


@dataclass(frozen=True)
class PackEntry:
    cue: str
    count: int = 1
    stem: str | None = None  # the game's filename, when it isn't the cue's name

    def filename(self, index: int) -> str:
        """The shared name.wav / name_v02.wav variation contract.

        offroad-racer and creature-rancher ship the same variant_filename() in
        their src/sound.c: variant 0 is the bare stem, the rest count from _v02,
        and loading stops at the first one missing. `render` numbers its own
        variations from _v01 instead; that path is for auditioning, not shipping.
        """
        base = self.stem or self.cue
        return base if index == 0 else f"{base}_v{index + 1:02d}"


PACKS: dict[str, tuple[PackEntry, ...]] = {
    # One of everything, in every style — the review bank.
    "showcase": tuple(PackEntry(name) for name in CUES),
    # The roadside cash and nitro crates. Four deep because a lap collects
    # several and one repeated file would give the rotation away.
    "offroad-racer": (PackEntry("coin", 4, "pickup"),),
    # Training a creature is a small success, not a fanfare: the activate
    # arpeggio's "switched on" reads as the animal learning it, and level_up
    # stays free for the rarer beat. Render with --style acoustic — that style's
    # sparkle tail is the chime accent this cue is written around.
    "creature-rancher": (PackEntry("powerup_activate", 4, "train"),),
}


def render(cue: Cue, style: Style, rng: np.random.Generator) -> np.ndarray:
    """Render one cue and master it to its own level."""
    signal = cue.render(style, rng)
    return e.master(signal, cue.peak_db, drive=style.drive)
