"""Original cue and pack registries for a quirky 16-bit RPG sound set."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from ..._audio import dsp
from ..._audio.models import RenderResult, VariationFilenames
from ..._audio.validation import bounded_float
from . import engine
from .engine import STYLES, Style


Renderer = Callable[[Style, random.Random, float], list[float]]


@dataclass(frozen=True, slots=True)
class Options:
    style: Style = STYLES["suburban"]
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


def _canvas(seconds: float) -> list[float]:
    return dsp.silence(seconds)


def _add(
    output: list[float],
    onset: float,
    signal: Sequence[float],
    gain: float = 1.0,
) -> None:
    dsp.add_at(output, onset, list(signal), gain)


def _noise_swell(
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    peak: float,
    low: float = 280.0,
    high: float = 8_000.0,
) -> list[float]:
    noise = dsp.bandpass(
        dsp.noise(duration, rng),
        low,
        min(engine.SR * 0.46, high * style.brightness),
        2,
    )
    envelope = dsp.bell_env(
        duration,
        peak,
        max(0.025, peak * 0.72),
        max(0.04, (duration - peak) * 0.55),
    )
    return dsp.mul(noise, envelope)


# Interface -----------------------------------------------------------------

def _cursor_move(style: Style, rng: random.Random,
                 intensity: float) -> list[float]:
    output = _canvas(0.08)
    _add(output, 0.0, engine.chip_tone(76, 0.07, style, rng,
                                       decay=0.014, echo_mix=0.0), 0.58)
    _add(output, 0.0, engine.noise_hit(0.018, style, rng,
                                       low=1_500.0, high=8_500.0),
         0.12 + 0.08 * intensity)
    return output


def _menu_confirm(style: Style, rng: random.Random,
                  intensity: float) -> list[float]:
    output = engine.arpeggio((62, 69), 0.24, style, rng,
                              gap=0.065, tone_seconds=0.17)
    _add(output, 0.0, engine.drum("clap", 0.045, style, rng), 0.10)
    return output


def _menu_cancel(style: Style, rng: random.Random,
                 intensity: float) -> list[float]:
    output = engine.arpeggio((69, 62), 0.24, style, rng,
                              gap=0.064, tone_seconds=0.16)
    _add(output, 0.10, engine.noise_hit(0.055, style, rng,
                                       low=300.0, high=2_200.0), 0.14)
    return output


def _menu_error(style: Style, rng: random.Random,
                intensity: float) -> list[float]:
    output = _canvas(0.32)
    for index, note in enumerate((66, 65, 66)):
        _add(output, index * 0.062,
             engine.chip_tone(note, 0.095, style, rng,
                              decay=0.026, vibrato=0.012, echo_mix=0.04),
             0.48 + 0.14 * intensity)
    _add(output, 0.04, engine.noise_hit(0.20, style, rng,
                                        low=240.0, high=3_400.0,
                                        decay=0.10), 0.18)
    return output


# Dialogue ------------------------------------------------------------------

def _text_blip(style: Style, rng: random.Random,
               intensity: float) -> list[float]:
    note = rng.choice((67, 69, 72, 74))
    output = _canvas(0.09)
    _add(output, 0.0, engine.chip_tone(note, 0.075, style, rng,
                                       decay=0.017, echo_mix=0.0),
         0.42 + 0.12 * intensity)
    return output


def _dialogue_open(style: Style, rng: random.Random,
                   intensity: float) -> list[float]:
    output = _canvas(0.38)
    _add(output, 0.0, engine.chord((55, 62, 65), 0.32, style, rng),
         0.38 + 0.12 * intensity)
    _add(output, 0.02, engine.noise_hit(0.07, style, rng,
                                       low=700.0, high=4_800.0), 0.14)
    return output


def _exclamation(style: Style, rng: random.Random,
                 intensity: float) -> list[float]:
    output = _canvas(0.30)
    _add(output, 0.0, engine.drum("kick", 0.18, style, rng), 0.28)
    _add(output, 0.01, engine.pitch_slide(57, 86, 0.24, style, rng,
                                          curve=0.58, decay=0.14),
         0.58 + 0.18 * intensity)
    return output


def _comedy_sting(style: Style, rng: random.Random,
                  intensity: float) -> list[float]:
    output = _canvas(0.58)
    _add(output, 0.0, engine.arpeggio((60, 66, 63, 69), 0.45, style, rng,
                                      gap=0.083, tone_seconds=0.18,
                                      crooked=0.22),
         0.55 + 0.12 * intensity)
    _add(output, 0.32, engine.drum("snare", 0.18, style, rng), 0.26)
    return output


# Items and character state -------------------------------------------------

def _item_get(style: Style, rng: random.Random,
              intensity: float) -> list[float]:
    output = _canvas(0.62)
    _add(output, 0.0, engine.arpeggio((64, 67, 71, 76), 0.50, style, rng,
                                      gap=0.078, tone_seconds=0.24,
                                      crooked=0.08),
         0.56 + 0.14 * intensity)
    _add(output, 0.22, engine.sparkle(0.34, style, rng, count=5), 0.28)
    return output


def _equip(style: Style, rng: random.Random,
           intensity: float) -> list[float]:
    output = _canvas(0.42)
    _add(output, 0.0, engine.drum("clap", 0.12, style, rng), 0.28)
    _add(output, 0.025, engine.chord((50, 57, 64), 0.34, style, rng),
         0.42 + 0.18 * intensity)
    _add(output, 0.0, engine.noise_hit(0.06, style, rng,
                                       low=1_800.0, high=10_000.0), 0.18)
    return output


def _heal(style: Style, rng: random.Random,
          intensity: float) -> list[float]:
    output = _canvas(0.78)
    _add(output, 0.0, engine.arpeggio((57, 62, 66, 69, 74), 0.64, style, rng,
                                      gap=0.082, tone_seconds=0.27,
                                      crooked=0.06),
         0.46 + 0.18 * intensity)
    _add(output, 0.15, engine.sparkle(0.54, style, rng,
                                      count=5 + round(4 * intensity)), 0.25)
    return output


def _status_inflicted(style: Style, rng: random.Random,
                      intensity: float) -> list[float]:
    output = _canvas(0.64)
    _add(output, 0.0, engine.pitch_slide(75, 48, 0.48, style, rng,
                                          curve=1.45, wobble=0.025,
                                          decay=0.30),
         0.55 + 0.18 * intensity)
    _add(output, 0.08, _noise_swell(0.43, style, rng, peak=0.18,
                                    low=180.0, high=3_600.0), 0.24)
    return output


def _level_up(style: Style, rng: random.Random,
              intensity: float) -> list[float]:
    output = _canvas(1.05)
    notes = (55, 59, 62, 67, 71, 74)
    _add(output, 0.0, engine.arpeggio(notes, 0.84, style, rng,
                                      gap=0.092, tone_seconds=0.34,
                                      crooked=0.10),
         0.56 + 0.14 * intensity)
    _add(output, 0.50, engine.chord((62, 67, 71, 74), 0.46, style, rng), 0.34)
    _add(output, 0.58, engine.sparkle(0.36, style, rng, count=7), 0.26)
    return output


# Psychic effects -----------------------------------------------------------

def _psi_charge(style: Style, rng: random.Random,
                intensity: float) -> list[float]:
    output = _canvas(0.88)
    for index in range(3):
        onset = index * 0.16
        _add(output, onset,
             engine.pitch_slide(43 + index * 4, 70 + index * 5, 0.48,
                                style, rng, curve=0.68, wobble=0.018),
             (0.20 + index * 0.10) * (0.70 + 0.30 * intensity))
    _add(output, 0.12, _noise_swell(0.68, style, rng, peak=0.45), 0.22)
    return output


def _psi_cast(style: Style, rng: random.Random,
              intensity: float) -> list[float]:
    output = _canvas(0.64)
    _add(output, 0.0, engine.pitch_slide(48, 91, 0.50, style, rng,
                                          curve=0.52, wobble=0.014,
                                          decay=0.34),
         0.60 + 0.18 * intensity)
    _add(output, 0.30, engine.chord((72, 78, 83), 0.28, style, rng), 0.26)
    _add(output, 0.0, engine.noise_hit(0.12, style, rng,
                                       low=900.0, high=11_000.0), 0.22)
    return output


def _psi_impact(style: Style, rng: random.Random,
                intensity: float) -> list[float]:
    output = _canvas(0.72)
    _add(output, 0.0, engine.drum("kick", 0.45, style, rng), 0.72)
    _add(output, 0.0, engine.noise_hit(0.40, style, rng,
                                       low=120.0, high=10_000.0,
                                       decay=0.11), 0.42 + 0.20 * intensity)
    _add(output, 0.025, engine.pitch_slide(88, 35, 0.56, style, rng,
                                           curve=1.75, wobble=0.022,
                                           decay=0.22), 0.38)
    return output


def _shield_up(style: Style, rng: random.Random,
               intensity: float) -> list[float]:
    output = _canvas(0.76)
    _add(output, 0.0, engine.pitch_slide(45, 76, 0.52, style, rng,
                                          curve=0.66, wobble=0.010), 0.42)
    _add(output, 0.18, engine.chord((60, 67, 73), 0.48, style, rng),
         0.46 + 0.16 * intensity)
    _add(output, 0.22, engine.sparkle(0.42, style, rng, count=6), 0.20)
    return output


def _teleport(style: Style, rng: random.Random,
              intensity: float) -> list[float]:
    output = _canvas(0.98)
    for index in range(5):
        onset = index * 0.12
        _add(output, onset,
             engine.pitch_slide(50 + index * 4, 75 + index * 5, 0.34,
                                style, rng, curve=0.50,
                                wobble=0.018 + index * 0.003,
                                decay=0.16),
             (0.22 + index * 0.055) * (0.72 + 0.28 * intensity))
    _add(output, 0.20, _noise_swell(0.62, style, rng, peak=0.36), 0.22)
    _add(output, 0.72, engine.noise_hit(0.14, style, rng,
                                       low=1_600.0, high=12_000.0), 0.28)
    return output


# Battle --------------------------------------------------------------------

def _battle_start(style: Style, rng: random.Random,
                  intensity: float) -> list[float]:
    output = _canvas(0.92)
    for onset, kind, gain in (
        (0.0, "kick", 0.55),
        (0.14, "snare", 0.32),
        (0.27, "kick", 0.48),
        (0.39, "clap", 0.28),
    ):
        _add(output, onset, engine.drum(kind, 0.24, style, rng), gain)
    _add(output, 0.08, engine.arpeggio((48, 54, 61, 65), 0.68, style, rng,
                                       gap=0.112, tone_seconds=0.28,
                                       crooked=0.16),
         0.48 + 0.16 * intensity)
    return output


def _player_hit(style: Style, rng: random.Random,
                intensity: float) -> list[float]:
    output = _canvas(0.34)
    _add(output, 0.0, engine.drum("kick", 0.28, style, rng), 0.58)
    _add(output, 0.0, engine.noise_hit(0.20, style, rng,
                                       low=350.0, high=6_800.0,
                                       decay=0.052), 0.38 + 0.20 * intensity)
    _add(output, 0.015, engine.pitch_slide(62, 40, 0.23, style, rng,
                                           curve=1.25, decay=0.08), 0.26)
    return output


def _enemy_hit(style: Style, rng: random.Random,
               intensity: float) -> list[float]:
    output = _canvas(0.38)
    _add(output, 0.0, engine.drum("tom", 0.31, style, rng), 0.52)
    _add(output, 0.006, engine.drum("snare", 0.24, style, rng),
         0.28 + 0.16 * intensity)
    _add(output, 0.04, engine.pitch_slide(69, 49, 0.25, style, rng,
                                          wobble=0.012, decay=0.09), 0.22)
    return output


def _critical_hit(style: Style, rng: random.Random,
                  intensity: float) -> list[float]:
    output = _canvas(0.68)
    _add(output, 0.0, engine.drum("kick", 0.42, style, rng), 0.82)
    _add(output, 0.0, engine.drum("clap", 0.30, style, rng), 0.48)
    _add(output, 0.012, engine.noise_hit(0.34, style, rng,
                                        low=180.0, high=13_000.0,
                                        decay=0.075), 0.52)
    _add(output, 0.08, engine.chord((47, 53, 60), 0.46, style, rng),
         0.38 + 0.18 * intensity)
    return output


def _victory(style: Style, rng: random.Random,
             intensity: float) -> list[float]:
    output = _canvas(1.18)
    _add(output, 0.0, engine.arpeggio((55, 62, 65, 71, 74, 79),
                                      0.92, style, rng, gap=0.098,
                                      tone_seconds=0.34, crooked=0.12),
         0.54 + 0.14 * intensity)
    _add(output, 0.56, engine.chord((62, 67, 71, 76), 0.54, style, rng), 0.36)
    _add(output, 0.10, engine.drum("snare", 0.18, style, rng), 0.18)
    _add(output, 0.72, engine.sparkle(0.36, style, rng, count=8), 0.24)
    return output


# World ---------------------------------------------------------------------

def _door_transition(style: Style, rng: random.Random,
                     intensity: float) -> list[float]:
    output = _canvas(0.48)
    _add(output, 0.0, engine.noise_hit(0.34, style, rng,
                                       low=110.0, high=3_500.0,
                                       decay=0.16), 0.30)
    _add(output, 0.015, engine.pitch_slide(58, 34, 0.36, style, rng,
                                           curve=1.30, wobble=0.010),
         0.38 + 0.15 * intensity)
    _add(output, 0.28, engine.drum("tom", 0.16, style, rng), 0.22)
    return output


def _phone_ring(style: Style, rng: random.Random,
                intensity: float) -> list[float]:
    output = _canvas(0.92)
    for base in (0.02, 0.44):
        _add(output, base,
             engine.arpeggio((74, 81, 77), 0.31, style, rng,
                             gap=0.068, tone_seconds=0.14,
                             crooked=0.12),
             0.48 + 0.14 * intensity)
    return output


def _save_complete(style: Style, rng: random.Random,
                   intensity: float) -> list[float]:
    output = _canvas(0.84)
    _add(output, 0.0, engine.arpeggio((74, 69, 65, 62), 0.64, style, rng,
                                      gap=0.092, tone_seconds=0.27,
                                      crooked=0.06),
         0.44 + 0.14 * intensity)
    _add(output, 0.38, engine.chord((50, 57, 62), 0.38, style, rng), 0.28)
    return output


def _party_join(style: Style, rng: random.Random,
                intensity: float) -> list[float]:
    output = _canvas(1.10)
    _add(output, 0.0, engine.arpeggio((60, 64, 67, 71), 0.72, style, rng,
                                      gap=0.105, tone_seconds=0.31,
                                      crooked=0.12), 0.46)
    _add(output, 0.28, engine.arpeggio((55, 62, 65, 69), 0.70, style, rng,
                                       gap=0.105, tone_seconds=0.31,
                                       crooked=-0.08),
         0.34 + 0.16 * intensity)
    _add(output, 0.67, engine.chord((60, 64, 67, 71), 0.36, style, rng), 0.30)
    return output


def _strange_event(style: Style, rng: random.Random,
                   intensity: float) -> list[float]:
    output = _canvas(0.88)
    notes = (58, 65, 61, 72)
    for index, note in enumerate(notes):
        onset = index * 0.14
        direction = 6 if index % 2 == 0 else -7
        _add(output, onset,
             engine.pitch_slide(note, note + direction, 0.34, style, rng,
                                curve=0.82 + 0.18 * index,
                                wobble=0.026, decay=0.18),
             (0.28 + 0.05 * index) * (0.74 + 0.26 * intensity))
    _add(output, 0.05, _noise_swell(0.72, style, rng, peak=0.38,
                                    low=120.0, high=6_400.0), 0.18)
    return output


def _production_renderer(kind: str, signature: int,
                         seconds: float) -> Renderer:
    """Create an original cartridge-styled cue for a gameplay semantic."""

    def render(style: Style, rng: random.Random,
               intensity: float) -> list[float]:
        output = _canvas(seconds)
        base = 44 + signature % 25
        high = 70 + signature % 17
        weight = 0.72 + 0.28 * intensity
        if kind == "ui":
            notes = (base + 12, base + 19) if signature % 2 == 0 else (
                base + 19, base + 12
            )
            _add(output, 0.0, engine.arpeggio(
                notes, min(seconds, 0.34), style, rng,
                gap=0.065, tone_seconds=min(0.18, seconds),
                crooked=(signature % 3 - 1) * 0.09,
            ), 0.58 * weight)
            _add(output, 0.0, engine.noise_hit(
                min(seconds, 0.035), style, rng,
                low=1_100.0, high=8_000.0,
            ), 0.12)
        elif kind == "impact":
            drum = ("kick", "snare", "tom", "clap")[signature % 4]
            _add(output, 0.0, engine.drum(
                drum, min(seconds, 0.38), style, rng,
            ), 0.54 * weight)
            _add(output, 0.0, engine.noise_hit(
                min(seconds, 0.30), style, rng,
                low=140.0 + signature * 23.0,
                high=4_200.0 + signature * 120.0,
                decay=0.035 + signature % 5 * 0.014,
            ), 0.34 * weight)
            _add(output, 0.01, engine.pitch_slide(
                high, base, min(seconds, 0.28), style, rng,
                curve=1.1 + signature % 4 * 0.20,
                wobble=0.006 + signature % 4 * 0.004,
            ), 0.18)
        elif kind == "motion":
            rising = signature % 2 == 0
            start = base if rising else high + 8
            end = high + 8 if rising else base
            _add(output, 0.0, engine.pitch_slide(
                start, end, min(seconds * 0.80, 0.68), style, rng,
                curve=0.62 if rising else 1.46,
                wobble=0.008 + signature % 5 * 0.003,
            ), 0.50 * weight)
            _add(output, 0.0, _noise_swell(
                min(seconds, 0.62), style, rng,
                peak=min(seconds * 0.40, 0.24),
                low=180.0, high=5_800.0 + signature * 65.0,
            ), 0.14)
        elif kind == "alert":
            notes = tuple(high if index % 2 == 0 else high - 6 - signature % 3
                          for index in range(3 + signature % 3))
            _add(output, 0.0, engine.arpeggio(
                notes, min(seconds, 0.72), style, rng,
                gap=0.095, tone_seconds=0.15, crooked=0.14,
            ), 0.52 * weight)
            _add(output, 0.0, engine.drum(
                "tom", min(seconds, 0.30), style, rng,
            ), 0.20 + 0.10 * intensity)
        elif kind == "positive":
            notes = (base, base + 4 + signature % 3, base + 7, base + 12)
            _add(output, 0.0, engine.arpeggio(
                notes, min(seconds, 0.86), style, rng,
                gap=max(0.065, min(0.12, seconds / 7.0)),
                tone_seconds=min(0.30, seconds * 0.55), crooked=0.08,
            ), 0.55 * weight)
            _add(output, min(seconds * 0.28, 0.20), engine.sparkle(
                min(seconds * 0.58, 0.48), style, rng,
                count=3 + signature % 6,
            ), 0.20)
        elif kind == "negative":
            _add(output, 0.0, engine.pitch_slide(
                high + 5, base - 5, min(seconds * 0.76, 0.62),
                style, rng, curve=1.28 + signature % 4 * 0.17,
                wobble=0.018 + signature % 5 * 0.004,
            ), 0.54 * weight)
            _add(output, min(0.06, seconds * 0.16), _noise_swell(
                min(seconds * 0.72, 0.50), style, rng,
                peak=min(seconds * 0.30, 0.19),
                low=100.0, high=3_800.0 + signature * 70.0,
            ), 0.21)
        elif kind == "role":
            notes = (base, base + 7, base + 3 + signature % 5, high)
            _add(output, 0.0, engine.arpeggio(
                notes, min(seconds, 0.78), style, rng,
                gap=0.083 + signature % 3 * 0.012,
                tone_seconds=0.25, crooked=0.18 - signature % 4 * 0.07,
            ), 0.48 * weight)
            _add(output, min(seconds * 0.30, 0.22), engine.chord(
                (base - 5, base + 2, base + 9),
                min(seconds * 0.52, 0.38), style, rng,
            ), 0.22)
        elif kind == "service":
            _add(output, 0.0, engine.drum(
                ("clap", "tom", "snare")[signature % 3],
                min(seconds, 0.22), style, rng,
            ), 0.20)
            _add(output, 0.035, engine.arpeggio(
                (base, base + 5, base + 12), min(seconds, 0.72),
                style, rng, gap=0.105, tone_seconds=0.28,
                crooked=0.10,
            ), 0.48 * weight)
        elif kind == "condition":
            _add(output, 0.0, engine.pitch_slide(
                high, base - signature % 7, min(seconds, 0.58),
                style, rng, curve=1.34, wobble=0.025 + signature % 4 * 0.008,
            ), 0.48 * weight)
            _add(output, 0.04, engine.noise_hit(
                min(seconds * 0.66, 0.42), style, rng,
                low=120.0, high=3_200.0 + signature * 95.0,
                decay=0.16,
            ), 0.20)
        else:
            raise ValueError(f"unknown production renderer kind: {kind}")
        return output

    return render


GROUPS = (
    "interface", "dialogue", "items", "psychic", "battle", "world",
    "field", "roles", "services", "conditions",
)


def _cue(
    name: str,
    group: str,
    description: str,
    renderer: Renderer,
    peak_db: float,
    seconds: float,
    *layers: str,
    foley_kinds: tuple[str, ...] = (),
) -> Cue:
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
        _cue("cursor_move", "interface", "move a cursor through a compact menu",
             _cursor_move, -10.0, 0.08, "dry square chirp", "toy switch"),
        _cue("menu_confirm", "interface", "accept a menu choice",
             _menu_confirm, -7.0, 0.24, "rising fifth", "soft clap edge"),
        _cue("menu_cancel", "interface", "back out without signaling failure",
             _menu_cancel, -8.0, 0.24, "falling fifth", "muted release"),
        _cue("menu_error", "interface", "reject an invalid menu action",
             _menu_error, -6.0, 0.32, "crooked semitone", "coarse buzz"),
        _cue("text_blip", "dialogue", "one repeatable character of dialogue text",
             _text_blip, -12.0, 0.09, "short varied speech bleep"),
        _cue("dialogue_open", "dialogue", "open a conversation or thought box",
             _dialogue_open, -8.0, 0.38, "small suspended chord", "paper edge"),
        _cue("exclamation", "dialogue", "surprise, realization, or sudden interruption",
             _exclamation, -4.5, 0.30, "toy kick", "fast octave launch"),
        _cue("comedy_sting", "dialogue", "offbeat punchline or awkward reaction",
             _comedy_sting, -5.5, 0.58, "crooked chromatic tag", "toy snare"),
        _cue("item_get", "items", "collect a useful everyday object",
             _item_get, -5.0, 0.62, "quirky rising figure", "small sparkle"),
        _cue("equip", "items", "equip clothing, gear, or a household weapon",
             _equip, -4.5, 0.42, "fastener clap", "sturdy chord", "metal edge",
             foley_kinds=("cartridge-treated cloth or gear",)),
        _cue("heal", "items", "recover health or remove fatigue",
             _heal, -5.0, 0.78, "open rising scale", "restorative motes"),
        _cue("status_inflicted", "items", "poison, confusion, homesickness, or another ailment",
             _status_inflicted, -4.0, 0.64, "warped pitch fall", "uneasy noise swell"),
        _cue("level_up", "items", "character level or attribute increases",
             _level_up, -3.5, 1.05, "asymmetric ascent", "arrival chord", "sparkle"),
        _cue("psi_charge", "psychic", "gather psychic energy before an ability",
             _psi_charge, -4.0, 0.88, "stacked bent rises", "spectral pressure"),
        _cue("psi_cast", "psychic", "release an original psychic technique",
             _psi_cast, -2.5, 0.64, "wide launch", "dissonant crown", "noise edge"),
        _cue("psi_impact", "psychic", "psychic energy strikes a target",
             _psi_impact, -1.5, 0.72, "sub impact", "broadband flash", "pitch collapse"),
        _cue("shield_up", "psychic", "protective psychic field engages",
             _shield_up, -4.0, 0.76, "field rise", "suspended shield chord", "motes"),
        _cue("teleport", "psychic", "rapid spatial departure or arrival",
             _teleport, -2.5, 0.98, "accelerating bent stages", "spectral trail", "arrival snap"),
        _cue("battle_start", "battle", "snap from exploration into an odd encounter",
             _battle_start, -2.5, 0.92, "toy drum break", "lopsided battle figure"),
        _cue("player_hit", "battle", "party member takes damage",
             _player_hit, -2.5, 0.34, "body kick", "coarse contact", "falling pain chirp",
             foley_kinds=("cartridge-treated body impact",)),
        _cue("enemy_hit", "battle", "enemy takes an ordinary hit",
             _enemy_hit, -3.0, 0.38, "toy tom", "snare contact", "bent reaction",
             foley_kinds=("cartridge-treated body impact", "wooden bat contact")),
        _cue("critical_hit", "battle", "decisive high-damage impact",
             _critical_hit, -1.0, 0.68, "kick and clap", "wide crack", "heavy chord",
             foley_kinds=("cartridge-treated heavy impact", "wooden bat contact")),
        _cue("victory", "battle", "battle resolved in the party's favor",
             _victory, -2.5, 1.18, "offbeat fanfare", "settled chord", "toy flourish"),
        _cue("door_transition", "world", "enter a building or change small areas",
             _door_transition, -6.0, 0.48, "wooden noise", "descending hinge", "latch tom",
             foley_kinds=("cartridge-treated panel and latch",)),
        _cue("phone_ring", "world", "an original household telephone call pattern",
             _phone_ring, -6.0, 0.92, "two crooked three-note rings"),
        _cue("save_complete", "world", "record progress and settle safely",
             _save_complete, -7.0, 0.84, "descending memory phrase", "home chord"),
        _cue("party_join", "world", "a new companion joins the group",
             _party_join, -4.0, 1.10, "two interlocking voices", "shared chord"),
        _cue("strange_event", "world", "surreal reveal, odd NPC beat, or reality shift",
             _strange_event, -4.0, 0.88, "alternating bent phrases", "uncertain noise"),
        _cue("attack_swing", "battle", "party member swings or throws an attack",
             _production_renderer("motion", 3, 0.48), -4.0, 0.48,
             "cartridge air cut",
             foley_kinds=("physically modelled blunt-weapon swing", "cloth movement")),
        _cue("attack_miss", "battle", "an attack misses its target",
             _production_renderer("motion", 4, 0.48), -5.0, 0.48,
             "empty swing", "miss chirp",
             foley_kinds=("physically modelled blunt-weapon swing", "cloth movement")),
        _cue("enemy_attack", "battle", "enemy commits a physical attack",
             _production_renderer("motion", 39, 0.60), -3.5, 0.60,
             "creature movement", "attack voice",
             foley_kinds=("cartridge-treated creature attack",)),
        _cue("defend", "battle", "party member braces against damage",
             _production_renderer("positive", 5, 0.48), -6.0, 0.48, "guard chord", "brace tick"),
        _cue("run_success", "battle", "party escapes the encounter",
             _production_renderer("positive", 6, 0.68), -4.5, 0.68, "escape run", "release chord"),
        _cue("run_failure", "battle", "escape attempt is blocked",
             _production_renderer("negative", 7, 0.54), -4.0, 0.54, "stumble fall", "denial buzz"),
        _cue("mortal_warning", "battle", "rolling HP approaches zero",
             _production_renderer("alert", 8, 0.76), -2.0, 0.76, "urgent counter pulse"),
        _cue("shield_break", "battle", "protective psychic field breaks",
             _production_renderer("impact", 9, 0.64), -2.5, 0.64, "field crack", "falling shards"),
        _cue("status_cured", "battle", "an unusual condition is removed",
             _production_renderer("positive", 10, 0.62), -5.0, 0.62, "cleanse ascent", "clear sparkle"),
        _cue("revive", "battle", "an unconscious party member returns",
             _production_renderer("positive", 11, 0.96), -3.0, 0.96, "heartbeat return", "rising life chord"),
        _cue("enemy_defeat", "battle", "one enemy leaves the encounter",
             _production_renderer("impact", 12, 0.72), -3.0, 0.72,
             "odd collapse", "defeat pop",
             foley_kinds=("cartridge-treated creature defeat",)),
        _cue("party_down", "battle", "the party is defeated",
             _production_renderer("negative", 13, 1.06), -2.0, 1.06,
             "rolling collapse", "lost chord",
             foley_kinds=("cartridge-treated body fall",)),
        _cue("boss_start", "battle", "major enemy encounter begins",
             _production_renderer("alert", 14, 1.02), -1.5, 1.02, "boss drum", "crooked threat call"),
        _cue("mood_shift", "battle", "enemy mood changes substantially",
             _production_renderer("role", 15, 0.56), -5.0, 0.56, "mood wobble", "reaction chord"),
        _cue("surrender", "battle", "enemy yields without being defeated",
             _production_renderer("positive", 16, 0.86), -3.5, 0.86, "tension release", "peace cadence"),
        _cue("enemy_notice", "field", "visible enemy notices the party",
             _production_renderer("alert", 17, 0.46), -4.0, 0.46, "attention chirp", "chase tick"),
        _cue("player_advantage", "field", "party enters battle with initiative",
             _production_renderer("positive", 18, 0.72), -3.5, 0.72, "advantage ascent"),
        _cue("enemy_advantage", "field", "enemy enters battle with initiative",
             _production_renderer("negative", 19, 0.72), -2.5, 0.72, "ambush fall"),
        _cue("instant_win", "field", "outmatched enemies are resolved immediately",
             _production_renderer("positive", 20, 0.82), -3.0, 0.82, "quick victory", "reward sparkle"),
        _cue("enemy_join", "field", "nearby enemy joins the encounter",
             _production_renderer("alert", 21, 0.54), -3.0, 0.54, "reinforcement call"),
        _cue("footstep", "field", "one stylized suburban footstep",
             _production_renderer("impact", 22, 0.20), -10.0, 0.20, "shoe contact", "tiny room tone"),
        _cue("mount_move", "field", "temporary fast movement or bicycle-like travel",
             _production_renderer("motion", 23, 0.62), -7.0, 0.62, "wheel pulse", "travel chirp"),
        _cue("secret_found", "field", "hidden route or object is discovered",
             _production_renderer("positive", 24, 0.92), -3.5, 0.92, "discovery figure", "surprise sparkle"),
        _cue("inspire", "roles", "unpredictable supportive party command",
             _production_renderer("role", 25, 0.88), -4.0, 0.88, "hopeful odd phrase", "chance shimmer"),
        _cue("scan_enemy", "roles", "technical character analyzes an enemy",
             _production_renderer("role", 26, 0.64), -5.0, 0.64, "scanner steps", "data answer"),
        _cue("gadget_use", "roles", "technical gadget activates",
             _production_renderer("role", 27, 0.66), -4.0, 0.66, "toy mechanism", "gadget discharge"),
        _cue("gadget_break", "roles", "technical gadget fails or breaks",
             _production_renderer("negative", 28, 0.62), -3.0, 0.62, "broken clock", "parts rattle"),
        _cue("gadget_repair", "roles", "broken gadget is repaired",
             _production_renderer("service", 29, 0.84), -5.0, 0.84, "tool ticks", "repair cadence"),
        _cue("mimic", "roles", "party member imitates another action",
             _production_renderer("role", 30, 0.72), -4.0, 0.72, "echoed phrase", "copy wobble"),
        _cue("atm", "services", "bank deposit or withdrawal completes",
             _production_renderer("service", 31, 0.58), -6.0, 0.58, "keypad ticks", "cash code"),
        _cue("shop", "services", "shop transaction completes",
             _production_renderer("service", 32, 0.62), -5.5, 0.62, "register pop", "receipt cadence"),
        _cue("hospital", "services", "medical treatment or revival service",
             _production_renderer("service", 33, 0.92), -6.0, 0.92, "monitor phrase", "recovery chord"),
        _cue("inn", "services", "lodging rest restores the party",
             _production_renderer("service", 34, 1.02), -8.0, 1.02, "sleepy cadence", "morning chime"),
        _cue("delivery", "services", "stored item arrives or is collected",
             _production_renderer("service", 35, 0.74), -5.0, 0.74, "doorstep knock", "parcel code"),
        _cue("poison_tick", "conditions", "damage-over-time condition ticks",
             _production_renderer("condition", 36, 0.46), -5.0, 0.46, "sick pulse", "toxic wobble"),
        _cue("confusion", "conditions", "control or targeting becomes confused",
             _production_renderer("condition", 37, 0.68), -4.5, 0.68, "crossed pitches", "dizzy noise"),
        _cue("homesick", "conditions", "emotional condition interrupts an action",
             _production_renderer("condition", 38, 0.86), -6.0, 0.86, "lonely fall", "home echo"),
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
    if options.style.name not in STYLES or STYLES[options.style.name] != options.style:
        raise ValueError(f"unregistered style {options.style.name!r}")
    bounded_float(options.intensity, "intensity", 0.05, 1.0)
    bounded_float(options.foley, "foley", 0.0, 1.0)

    signal = cue.renderer(options.style, rng, options.intensity)
    sources: tuple[str, ...] = ()
    if options.foley > 0.0 and cue.foley_kinds:
        from . import foley

        signal, sources = foley.apply(
            cue_name,
            signal,
            options.style,
            options.intensity,
            options.foley,
            rng,
        )
    frames = dsp.samples(cue.seconds)
    signal = (signal + [0.0] * max(0, frames - len(signal)))[:frames]
    return RenderResult(
        engine.master(signal, cue.peak_db, options.style),
        sources,
    )


def render(cue_name: str, options: Options,
           rng: random.Random) -> list[float]:
    """Render one cue while preserving the historical list return type."""

    return _render_result(cue_name, options, rng).samples


def render_with_sources(cue_name: str, options: Options,
                        rng: random.Random) -> RenderResult[list[float]]:
    return _render_result(cue_name, options, rng)


def generate(
    cue_name: str,
    *,
    style: str = "suburban",
    intensity: float = 0.85,
    foley_mix: float = 0.0,
    seed: int = 20260721,
) -> list[float]:
    """Return deterministic mono samples for one original cue."""

    options = Options(
        style=engine.style_named(style),
        intensity=intensity,
        foley=foley_mix,
    )
    return render(cue_name, options, random.Random(seed))


def generate_with_sources(
    cue_name: str,
    *,
    style: str = "suburban",
    intensity: float = 0.85,
    foley_mix: float = 0.0,
    seed: int = 20260721,
) -> RenderResult[list[float]]:
    options = Options(
        style=engine.style_named(style),
        intensity=intensity,
        foley=foley_mix,
    )
    return render_with_sources(cue_name, options, random.Random(seed))


def stem_for(cue_name: str, options: Options) -> str:
    if cue_name not in CUES:
        raise ValueError(f"unknown cue {cue_name!r}")
    return f"{cue_name}_{options.style.name}"


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
    "one organised render of every quirky 16-bit RPG cue",
    tuple(PackEntry(f"{cue.group}/{cue.name}", cue.name) for cue in CUES.values()),
)

HOMETOWN = Pack(
    "hometown",
    "variation bank for menus, conversations, items, and everyday world events",
    tuple(
        PackEntry(name, name, count)
        for name, count in (
            ("cursor_move", 5),
            ("menu_confirm", 3),
            ("menu_cancel", 3),
            ("menu_error", 3),
            ("text_blip", 6),
            ("dialogue_open", 3),
            ("exclamation", 3),
            ("comedy_sting", 3),
            ("item_get", 3),
            ("equip", 3),
            ("heal", 3),
            ("phone_ring", 3),
            ("save_complete", 2),
            ("party_join", 2),
            ("strange_event", 3),
        )
    ),
)

PSYCHIC = Pack(
    "psychic",
    "three variations of every psychic charge, cast, impact, shield, and teleport",
    tuple(PackEntry(name, name, 3) for name in (
        "psi_charge", "psi_cast", "psi_impact", "shield_up", "teleport"
    )),
)

BATTLE = Pack(
    "battle",
    "three variations of encounters, damage, critical hits, and victory",
    tuple(PackEntry(name, name, 3) for name in (
        "battle_start", "player_hit", "enemy_hit", "critical_hit", "victory"
    )),
)

PACKS = {pack.name: pack for pack in (CORE, HOMETOWN, PSYCHIC, BATTLE)}


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
