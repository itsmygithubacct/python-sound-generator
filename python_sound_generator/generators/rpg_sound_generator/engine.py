"""Procedural building blocks for role-playing game sound effects.

The RPG bank intentionally concentrates on the connective sounds around an
adventure -- inventory, loot, quests, progression, dialogue, saving, and rest.
Combat, footsteps, magic, and ambience already have dedicated providers in the
project.  This module remains a shared-standard-library synthesis engine; the
provider adds its separately audited recording layer lazily in :mod:`foley`.
"""

from __future__ import annotations

from ..._audio.dsp import write_wav

import random
from dataclasses import dataclass
from typing import Sequence

from ..._audio import dsp


SR = dsp.SR
TAU = dsp.TAU


@dataclass(frozen=True, slots=True)
class Theme:
    """A coherent timbre applied across every cue in the library."""

    name: str
    description: str
    transpose: float
    brightness: float
    detune: float
    body: float
    voice: str


THEMES: dict[str, Theme] = {
    "fantasy": Theme(
        "fantasy",
        "bright bells, warm wood, and restrained magical sparkle",
        0.0,
        1.0,
        0.003,
        1.0,
        "bell",
    ),
    "dark": Theme(
        "dark",
        "lower, weightier, detuned tones with muted high frequencies",
        -5.0,
        0.62,
        0.014,
        1.35,
        "hollow",
    ),
    "retro": Theme(
        "retro",
        "compact pulse-wave voices inspired by early game sound hardware",
        7.0,
        1.25,
        0.0,
        0.72,
        "pulse",
    ),
}


def theme_named(name: str) -> Theme:
    """Resolve a theme by name with a useful error for API callers."""

    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(
            f"unknown theme {name!r}; expected one of {sorted(THEMES)}"
        ) from None


def midi_hz(note: float, theme: Theme | None = None) -> float:
    """Convert a MIDI note number to hertz, applying a theme transpose."""

    transpose = 0.0 if theme is None else theme.transpose
    return 440.0 * (2.0 ** ((note + transpose - 69.0) / 12.0))


def _pulse_track(freqs: Sequence[float], phase: float = 0.0,
                 duty: float = 0.38) -> list[float]:
    out: list[float] = []
    for frequency in freqs:
        cycle = (phase / TAU) % 1.0
        out.append(1.0 if cycle < duty else -1.0)
        phase += TAU * frequency / SR
    return out


def _voice_track(freqs: list[float], theme: Theme,
                 rng: random.Random) -> list[float]:
    """Render one theme-aware pitched voice from a frequency track."""

    phase = rng.uniform(0.0, TAU)
    if theme.voice == "pulse":
        fundamental = _pulse_track(freqs, phase)
        octave = _pulse_track([frequency * 2.0 for frequency in freqs],
                              phase * 0.73, duty=0.25)
        return dsp.lowpass(
            dsp.mix(fundamental, dsp.scale(octave, 0.18)),
            7_000.0 * theme.brightness,
            2,
        )

    fundamental = dsp.sine_track(freqs, phase)
    if theme.voice == "hollow":
        spread = max(theme.detune, 0.006)
        lower = dsp.sine_track(
            [frequency * (1.0 - spread) for frequency in freqs],
            rng.uniform(0.0, TAU),
        )
        upper = dsp.sine_track(
            [frequency * 1.5 for frequency in freqs],
            rng.uniform(0.0, TAU),
        )
        return dsp.lowpass(
            dsp.mix(
                dsp.scale(fundamental, 0.72),
                dsp.scale(lower, 0.48),
                dsp.scale(upper, 0.14),
            ),
            2_600.0 * theme.brightness,
            2,
        )

    octave = dsp.sine_track(
        [frequency * (2.0 + theme.detune) for frequency in freqs],
        rng.uniform(0.0, TAU),
    )
    fifth = dsp.sine_track(
        [frequency * (3.0 - theme.detune) for frequency in freqs],
        rng.uniform(0.0, TAU),
    )
    return dsp.mix(
        fundamental,
        dsp.scale(octave, 0.27),
        dsp.scale(fifth, 0.11),
    )


def tone(note: float, duration: float, theme: Theme, rng: random.Random, *,
         decay: float | None = None, attack: float = 0.002) -> list[float]:
    """A short theme-aware musical tone."""

    frequency = midi_hz(note, theme)
    track = [frequency] * dsp.samples(duration)
    voice = _voice_track(track, theme, rng)
    envelope = dsp.exp_env(duration, decay or duration * 0.55, attack)
    return dsp.mul(voice, envelope)


def sweep(start_note: float, end_note: float, duration: float, theme: Theme,
          rng: random.Random, *, decay: float | None = None,
          curve: float = 1.0) -> list[float]:
    """A pitched glide whose endpoints are MIDI notes."""

    start = midi_hz(start_note, theme)
    end = midi_hz(end_note, theme)
    track = dsp.ramp(duration, start, end, curve)
    voice = _voice_track(track, theme, rng)
    envelope = dsp.exp_env(duration, decay or duration * 0.65, 0.002)
    return dsp.mul(voice, envelope)


def chime(note: float, duration: float, theme: Theme,
          rng: random.Random) -> list[float]:
    """A struck note that remains recognisable in all three themes."""

    if theme.voice == "pulse":
        return tone(note, duration, theme, rng, decay=duration * 0.30,
                    attack=0.0005)

    root = midi_hz(note, theme)
    if theme.voice == "hollow":
        ratios = (1.0, 1.41, 2.03, 2.91)
        decays = (0.62, 0.38, 0.28, 0.18)
    else:
        ratios = (1.0, 2.01, 2.76, 5.40)
        decays = (0.68, 0.48, 0.34, 0.18)
    modes = [
        (
            root * ratio * (1.0 + theme.detune * index),
            duration * decays[index],
            1.0 / (index + 1),
        )
        for index, ratio in enumerate(ratios)
        if root * ratio < SR * 0.47
    ]
    return dsp.modal_ring(duration, modes, rng)


def arpeggio(notes: Sequence[float], duration: float, theme: Theme,
             rng: random.Random, *, gap: float = 0.09,
             note_duration: float = 0.30) -> list[float]:
    """Place a sequence of chimes on one fixed-length canvas."""

    out = dsp.silence(duration)
    for index, note in enumerate(notes):
        dsp.add_at(out, index * gap, chime(note, note_duration, theme, rng))
    return out


def click(duration: float, theme: Theme, rng: random.Random, *,
          bright: bool = True) -> list[float]:
    cutoff = (3_800.0 if bright else 900.0) * theme.brightness
    noise = dsp.highpass(dsp.noise(duration, rng), cutoff, 2)
    envelope = dsp.exp_env(duration, max(0.002, duration * 0.10), 0.0003)
    return dsp.mul(noise, envelope)


def thump(duration: float, theme: Theme, rng: random.Random) -> list[float]:
    track = dsp.ramp(duration, 150.0 * theme.body, 48.0, curve=0.55)
    body = dsp.mul(
        dsp.sine_track(track, rng.uniform(0.0, TAU)),
        dsp.exp_env(duration, duration * 0.22, 0.001),
    )
    texture = dsp.mul(
        dsp.lowpass(dsp.noise(duration, rng), 480.0 * theme.body, 2),
        dsp.exp_env(duration, duration * 0.09),
    )
    return dsp.mix(body, dsp.scale(texture, 0.32))


def whoosh(duration: float, start_hz: float, end_hz: float, theme: Theme,
           rng: random.Random) -> list[float]:
    track = dsp.ramp(
        duration,
        start_hz * theme.brightness,
        end_hz * theme.brightness,
        1.15,
    )
    noise = dsp.svf_bandpass(dsp.noise(duration, rng), track, q=1.2)
    return dsp.mul(
        noise,
        dsp.bell_env(duration, duration * 0.48, duration * 0.32,
                     duration * 0.38),
    )


def sparkle(duration: float, theme: Theme, rng: random.Random, *,
            density: int = 8) -> list[float]:
    """Scatter short tuned grains; retro uses deliberate stepped bleeps."""

    out = dsp.silence(duration)
    scale = (72, 74, 76, 79, 81, 84)
    for _ in range(density):
        onset = rng.uniform(0.0, max(0.0, duration - 0.06))
        note = rng.choice(scale)
        grain = tone(note, 0.065, theme, rng, decay=0.022, attack=0.0005)
        dsp.add_at(out, onset, grain, rng.uniform(0.10, 0.28))
    return out


def creak(duration: float, theme: Theme, rng: random.Random) -> list[float]:
    slips = dsp.stick_slip(
        duration,
        rng,
        rate=24.0,
        freq=210.0 * theme.body,
        jitter=0.75,
    )
    return dsp.lowpass(slips, 1_300.0 * theme.brightness, 2)


def master(signal: list[float], peak_db: float) -> list[float]:
    return dsp.master(signal, peak_db, drive=1.15,
                      fade_in_ms=0.5, fade_out_ms=35.0)




__all__ = [
    "SR",
    "THEMES",
    "Theme",
    "arpeggio",
    "chime",
    "click",
    "creak",
    "master",
    "midi_hz",
    "sparkle",
    "sweep",
    "theme_named",
    "thump",
    "tone",
    "whoosh",
    "write_wav",
]
