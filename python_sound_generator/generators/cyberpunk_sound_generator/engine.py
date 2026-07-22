"""Procedural synthesis primitives for the cyberpunk sound library.

The vocabulary is deliberately electronic rather than merely musical: short
FM chirps, quantised oscillators, packet-like pulse trains, filtered static,
sub impacts, and controlled digital failure.  Everything in this module uses
the shared standard-library DSP core; the provider imports its two optional
hardware-contact Foley paths only for a nonzero mix.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from ..._audio import dsp
from ..._audio.chiptune import bit_reduce, oscillator_track, sample_hold
from ..._audio.dsp import write_wav as write_wav


SR = dsp.SR
TAU = dsp.TAU


@dataclass(frozen=True, slots=True)
class Style:
    """One coherent production treatment for the complete cue bank."""

    name: str
    description: str
    transpose: float
    brightness: float
    detune_cents: float
    glitch: float
    drive: float
    body: float
    waveform: str


STYLES: dict[str, Style] = {
    "electric": Style(
        "electric",
        "bright saturated synths, energetic glitches, and vivid high detail",
        3.0,
        1.22,
        4.0,
        0.78,
        1.35,
        0.82,
        "saw",
    ),
    "noir": Style(
        "noir",
        "low, unstable electronics with restrained highs and heavier machinery",
        -5.0,
        0.58,
        11.0,
        0.48,
        1.55,
        1.30,
        "triangle",
    ),
    "corporate": Style(
        "corporate",
        "clean precise interfaces with minimal distortion and controlled bandwidth",
        0.0,
        0.92,
        1.5,
        0.18,
        1.08,
        0.90,
        "pulse",
    ),
}


def style_named(name: str) -> Style:
    try:
        return STYLES[name]
    except KeyError:
        raise ValueError(
            f"unknown style {name!r}; expected one of {sorted(STYLES)}"
        ) from None


def midi_hz(note: float, style: Style | None = None) -> float:
    transpose = 0.0 if style is None else style.transpose
    return 440.0 * 2.0 ** ((note + transpose - 69.0) / 12.0)


def _frequency_track(
    note: float,
    duration: float,
    style: Style,
    rng: random.Random,
    vibrato: float,
) -> list[float]:
    cents = rng.uniform(-style.detune_cents, style.detune_cents)
    frequency = midi_hz(note + cents / 100.0, style)
    rate = rng.uniform(4.2, 8.5)
    return [
        frequency * (1.0 + vibrato * math.sin(TAU * rate * frame / SR))
        for frame in range(dsp.samples(duration))
    ]


def digital_tone(
    note: float,
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    decay: float | None = None,
    attack: float = 0.001,
    vibrato: float = 0.0,
) -> list[float]:
    """A styled oscillator with octave edge, sub body, and a short envelope."""

    track = _frequency_track(note, duration, style, rng, vibrato)
    phase = rng.uniform(0.0, TAU)
    lead = oscillator_track(track, style.waveform, phase)
    edge = oscillator_track(
        [frequency * 2.01 for frequency in track],
        "sine",
        phase * 0.61,
    )
    sub = oscillator_track(
        [frequency * 0.5 for frequency in track],
        "sine",
        phase * 1.37,
    )
    voice = dsp.mix(
        dsp.scale(lead, 0.72),
        dsp.scale(edge, 0.18 * style.brightness),
        dsp.scale(sub, 0.18 * style.body),
    )
    voice = dsp.lowpass(voice, min(16_000.0, 5_800.0 * style.brightness), 2)
    envelope = dsp.exp_env(duration, decay or duration * 0.48, attack)
    return dsp.mul(voice, envelope)


def fm_tone(
    note: float,
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    ratio: float = 2.0,
    index: float = 1.8,
    decay: float | None = None,
) -> list[float]:
    """Metallic FM-like voice used for scanners, implants, and credentials."""

    base = midi_hz(note + rng.uniform(-0.03, 0.03), style)
    phase = rng.uniform(0.0, TAU)
    mod_phase = rng.uniform(0.0, TAU)
    output = []
    for frame in range(dsp.samples(duration)):
        time = frame / SR
        modulation = math.sin(mod_phase) * index * math.exp(
            -time / max(duration * 0.38, 0.001)
        )
        output.append(math.sin(phase + modulation))
        phase += TAU * base / SR
        mod_phase += TAU * base * ratio / SR
    return dsp.mul(
        output,
        dsp.exp_env(duration, decay or duration * 0.52, 0.0008),
    )


def sweep(
    start_note: float,
    end_note: float,
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    curve: float = 1.0,
    decay: float | None = None,
) -> list[float]:
    start = midi_hz(start_note, style)
    end = midi_hz(end_note, style)
    frequencies = dsp.ramp(duration, start, end, curve)
    voice = oscillator_track(frequencies, style.waveform, rng.uniform(0.0, TAU))
    overtone = dsp.sine_track(
        [frequency * 1.503 for frequency in frequencies],
        rng.uniform(0.0, TAU),
    )
    signal = dsp.mix(voice, dsp.scale(overtone, 0.20 * style.brightness))
    return dsp.mul(
        signal,
        dsp.exp_env(duration, decay or duration * 0.72, 0.001),
    )


def static_burst(
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    low: float = 900.0,
    high: float = 9_000.0,
    decay: float | None = None,
) -> list[float]:
    high = min(SR * 0.46, high * style.brightness)
    low = min(high * 0.75, low * max(0.65, style.brightness))
    noise = dsp.bandpass(dsp.noise(duration, rng), low, high, 2)
    return dsp.mul(
        noise,
        dsp.exp_env(duration, decay or duration * 0.20, 0.0004),
    )


def sub_hit(
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    start_hz: float = 150.0,
    end_hz: float = 42.0,
) -> list[float]:
    frequencies = dsp.ramp(
        duration,
        start_hz * style.body,
        end_hz,
        0.55,
    )
    body = dsp.sine_track(frequencies, rng.uniform(0.0, TAU))
    texture = dsp.lowpass(dsp.noise(duration, rng), 520.0 * style.body, 2)
    return dsp.mul(
        dsp.mix(body, dsp.scale(texture, 0.20)),
        dsp.exp_env(duration, duration * 0.23, 0.0008),
    )


def radio_noise(duration: float, style: Style, rng: random.Random) -> list[float]:
    """A narrow communications bed with irregular amplitude flutter."""

    noise = dsp.bandpass(
        dsp.noise(duration, rng),
        420.0,
        min(8_500.0, 4_900.0 * style.brightness),
        2,
    )
    rate = rng.uniform(26.0, 42.0)
    envelope = [
        0.36 + 0.64 * max(0.0, math.sin(TAU * rate * frame / SR))
        for frame in range(len(noise))
    ]
    return dsp.mul(noise, envelope)


def pulse_train(
    note: float,
    count: int,
    period: float,
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    rising: bool = False,
) -> list[float]:
    output = dsp.silence(duration)
    for index in range(count):
        offset = index * period
        if offset >= duration:
            break
        step = index * 1.25 if rising else 0.0
        voice = digital_tone(
            note + step,
            min(0.075, duration - offset),
            style,
            rng,
            decay=0.018,
            attack=0.0004,
        )
        dsp.add_at(output, offset, voice, 0.72 + 0.05 * index)
    return output


def sequence(
    notes: Sequence[float],
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    gap: float = 0.085,
    tone_seconds: float = 0.22,
    fm: bool = False,
) -> list[float]:
    output = dsp.silence(duration)
    renderer = fm_tone if fm else digital_tone
    for index, note in enumerate(notes):
        onset = index * gap
        if onset >= duration:
            break
        voice = renderer(
            note,
            min(tone_seconds, duration - onset),
            style,
            rng,
        )
        dsp.add_at(output, onset, voice, 0.72 / (1.0 + 0.08 * index))
    return output


def glitch(
    signal: Sequence[float],
    style: Style,
    rng: random.Random,
    amount: float = 1.0,
) -> list[float]:
    """Blend in deterministic bit reduction, sample hold, and brief dropouts."""

    mix = max(0.0, min(1.0, style.glitch * amount))
    if mix <= 0.0:
        return list(signal)
    bits = max(4, 12 - round(6 * mix))
    hold = max(1, 1 + round(7 * mix))
    crushed = sample_hold(bit_reduce(signal, bits), hold)
    for _ in range(round(2 + 6 * mix)):
        if not crushed:
            break
        start = rng.randrange(len(crushed))
        width = rng.randrange(1, max(2, round(SR * 0.006 * mix)))
        gain = rng.uniform(0.0, 0.24)
        for index in range(start, min(len(crushed), start + width)):
            crushed[index] *= gain
    return [
        float(clean) * (1.0 - mix) + dirty * mix
        for clean, dirty in zip(signal, crushed)
    ]


def master(signal: Sequence[float], peak_db: float, style: Style) -> list[float]:
    return dsp.master(
        list(signal),
        peak_db,
        drive=style.drive,
        fade_in_ms=0.4,
        fade_out_ms=24.0,
    )


__all__ = [
    "SR",
    "STYLES",
    "Style",
    "digital_tone",
    "fm_tone",
    "glitch",
    "master",
    "midi_hz",
    "oscillator_track",
    "pulse_train",
    "radio_noise",
    "sequence",
    "static_burst",
    "style_named",
    "sub_hit",
    "sweep",
    "write_wav",
]
