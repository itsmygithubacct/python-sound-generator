"""Original reduced-resolution synthesis for a quirky 16-bit RPG.

This engine evokes the technical palette of early-1990s cartridge RPG audio
without copying any game's compositions or samples.  Short looping waveforms,
sample-rate reduction, toy percussion, bent pitches, and compact echoes provide
the character; cue composition remains in :mod:`cues`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from ..._audio import dsp
from ..._audio.chiptune import Waveform, echo, oscillator_track, reduce
from ..._audio.dsp import write_wav as write_wav


SR = dsp.SR
TAU = dsp.TAU


@dataclass(frozen=True, slots=True)
class Style:
    """A complete production treatment for the original cue vocabulary."""

    name: str
    description: str
    waveform: Waveform
    duty: float
    transpose: float
    bits: int
    hold: int
    detune_cents: float
    brightness: float
    body: float
    echo_ms: float
    echo_feedback: float
    drive: float


STYLES: dict[str, Style] = {
    "suburban": Style(
        "suburban",
        "bouncy square leads, toy percussion, and dry everyday charm",
        "square",
        0.38,
        0.0,
        10,
        2,
        3.5,
        0.96,
        0.90,
        58.0,
        0.20,
        1.18,
    ),
    "psychedelic": Style(
        "psychedelic",
        "detuned saw voices, smeared echoes, bent pitch, and vivid instability",
        "saw",
        0.47,
        -2.0,
        8,
        3,
        14.0,
        1.18,
        1.08,
        104.0,
        0.42,
        1.42,
    ),
    "lofi": Style(
        "lofi",
        "warm dark triangles, coarse conversion, and weighty compact transients",
        "triangle",
        0.50,
        -5.0,
        7,
        5,
        7.0,
        0.62,
        1.28,
        72.0,
        0.24,
        1.34,
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


def _degrade(signal: Sequence[float], style: Style, *, echo_mix: float) -> list[float]:
    reduced = reduce(signal, bits=style.bits, hold=style.hold)
    return echo(
        reduced,
        style.echo_ms / 1000.0,
        style.echo_feedback,
        wet=echo_mix,
    )


def chip_tone(
    note: float,
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    decay: float | None = None,
    attack: float = 0.001,
    vibrato: float = 0.0,
    waveform: Waveform | None = None,
    echo_mix: float = 0.20,
) -> list[float]:
    """A compact chip voice with humanised pitch and reduced resolution."""

    cents = rng.uniform(-style.detune_cents, style.detune_cents)
    frequency = midi_hz(note + cents / 100.0, style)
    rate = rng.uniform(4.0, 7.2)
    frequencies = [
        frequency * (1.0 + vibrato * math.sin(TAU * rate * frame / SR))
        for frame in range(dsp.samples(duration))
    ]
    selected = waveform or style.waveform
    phase = rng.uniform(0.0, TAU)
    lead = oscillator_track(
        frequencies,
        selected,
        phase,
        duty=style.duty,
    )
    octave = oscillator_track(
        [value * 2.0 for value in frequencies],
        "square",
        phase * 0.53,
        duty=0.24,
    )
    sub = dsp.sine_track(
        [value * 0.5 for value in frequencies],
        phase * 1.31,
    )
    signal = dsp.mix(
        dsp.scale(lead, 0.72),
        dsp.scale(octave, 0.13 * style.brightness),
        dsp.scale(sub, 0.16 * style.body),
    )
    signal = dsp.lowpass(
        signal,
        min(SR * 0.46, 6_200.0 * style.brightness),
        2,
    )
    signal = dsp.mul(
        signal,
        dsp.exp_env(duration, decay or duration * 0.52, attack),
    )
    return _degrade(signal, style, echo_mix=echo_mix)


def pitch_slide(
    start_note: float,
    end_note: float,
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    curve: float = 1.0,
    wobble: float = 0.0,
    decay: float | None = None,
) -> list[float]:
    start = midi_hz(start_note, style)
    end = midi_hz(end_note, style)
    base = dsp.ramp(duration, start, end, curve)
    wobble_rate = rng.uniform(5.0, 9.0)
    frequencies = [
        value * (1.0 + wobble * math.sin(TAU * wobble_rate * frame / SR))
        for frame, value in enumerate(base)
    ]
    phase = rng.uniform(0.0, TAU)
    voice = oscillator_track(
        frequencies,
        style.waveform,
        phase,
        duty=style.duty,
    )
    support = dsp.sine_track(
        [value * 1.5 for value in frequencies],
        phase * 0.67,
    )
    signal = dsp.mul(
        dsp.mix(voice, dsp.scale(support, 0.16)),
        dsp.exp_env(duration, decay or duration * 0.72, 0.001),
    )
    return _degrade(signal, style, echo_mix=0.24)


def arpeggio(
    notes: Sequence[float],
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    gap: float = 0.075,
    tone_seconds: float = 0.24,
    crooked: float = 0.0,
) -> list[float]:
    """Place notes with optional alternating timing skew for offbeat motion."""

    output = dsp.silence(duration)
    onset = 0.0
    for index, note in enumerate(notes):
        if onset >= duration:
            break
        voice = chip_tone(
            note,
            min(tone_seconds, duration - onset),
            style,
            rng,
            echo_mix=0.12,
        )
        dsp.add_at(output, onset, voice, 0.70 / (1.0 + 0.07 * index))
        skew = crooked if index % 2 == 0 else -crooked
        onset += max(0.02, gap * (1.0 + skew))
    return output


def chord(
    notes: Sequence[float],
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    roll: float = 0.008,
) -> list[float]:
    output = dsp.silence(duration)
    for index, note in enumerate(notes):
        onset = index * roll
        voice = chip_tone(
            note,
            max(0.02, duration - onset),
            style,
            rng,
            decay=duration * 0.46,
            echo_mix=0.18,
        )
        dsp.add_at(output, onset, voice, 0.64 / (1.0 + 0.28 * index))
    return output


def noise_hit(
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    low: float = 500.0,
    high: float = 8_000.0,
    decay: float | None = None,
) -> list[float]:
    high = min(SR * 0.46, high * style.brightness)
    low = min(low * max(0.65, style.brightness), high * 0.80)
    noise = dsp.bandpass(dsp.noise(duration, rng), low, high, 2)
    shaped = dsp.mul(
        noise,
        dsp.exp_env(duration, decay or duration * 0.18, 0.0004),
    )
    return _degrade(shaped, style, echo_mix=0.08)


def drum(
    kind: str,
    duration: float,
    style: Style,
    rng: random.Random,
) -> list[float]:
    """Toy-box kick, snare, tom, or clap synthesized from primitive layers."""

    if kind == "kick":
        frequencies = dsp.ramp(duration, 165.0 * style.body, 44.0, 0.48)
        body = dsp.sine_track(frequencies, rng.uniform(0.0, TAU))
        click = noise_hit(min(duration, 0.035), style, rng,
                          low=1_100.0, high=7_000.0, decay=0.008)
        output = dsp.mul(body, dsp.exp_env(duration, duration * 0.20, 0.0005))
        dsp.add_at(output, 0.0, click, 0.16)
        return output
    if kind == "snare":
        noise = noise_hit(duration, style, rng, low=620.0, high=7_800.0,
                          decay=duration * 0.20)
        body = dsp.modal_ring(
            duration,
            [(185.0 * style.body, duration * 0.18, 0.42)],
            rng,
        )
        return dsp.mix(noise, body)
    if kind == "tom":
        return dsp.modal_ring(
            duration,
            [
                (132.0 * style.body, duration * 0.30, 1.0),
                (207.0 * style.body, duration * 0.19, 0.28),
            ],
            rng,
        )
    if kind == "clap":
        output = dsp.silence(duration)
        for onset in (0.0, 0.018, 0.037):
            dsp.add_at(
                output,
                onset,
                noise_hit(max(0.02, duration - onset), style, rng,
                          low=900.0, high=9_500.0, decay=0.018),
                0.48,
            )
        return output
    raise ValueError(f"unsupported drum kind: {kind!r}")


def sparkle(
    duration: float,
    style: Style,
    rng: random.Random,
    *,
    count: int = 7,
    notes: Sequence[float] = (76, 79, 81, 84, 88),
) -> list[float]:
    output = dsp.silence(duration)
    for _ in range(count):
        onset = rng.uniform(0.0, max(0.0, duration - 0.06))
        voice = chip_tone(
            rng.choice(tuple(notes)),
            0.075,
            style,
            rng,
            decay=0.020,
            attack=0.0003,
            waveform="square",
            echo_mix=0.08,
        )
        dsp.add_at(output, onset, voice, rng.uniform(0.12, 0.28))
    return output


def master(signal: Sequence[float], peak_db: float, style: Style) -> list[float]:
    return dsp.master(
        list(signal),
        peak_db,
        drive=style.drive,
        fade_in_ms=0.35,
        fade_out_ms=28.0,
    )


__all__ = [
    "SR",
    "STYLES",
    "Style",
    "arpeggio",
    "chip_tone",
    "chord",
    "drum",
    "master",
    "midi_hz",
    "noise_hit",
    "pitch_slide",
    "sparkle",
    "style_named",
    "write_wav",
]
