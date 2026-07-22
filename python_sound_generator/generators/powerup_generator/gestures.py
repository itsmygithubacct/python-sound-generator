"""Musical gestures — the shapes a pickup sound is made of.

These are style-agnostic: a gesture says "these notes, this fast, rising", and
the style says what a note sounds like. Cues are built by composing these.

Pitch is in MIDI note numbers throughout, because these sounds are intervals
first and frequencies second — a coin is "up a fifth", not "523 Hz then 784 Hz",
and transposing the whole bank later should be a single number.
"""

from __future__ import annotations

import numpy as np

from . import engine as e
from .styles import Style


def _humanise(style: Style, note: float,
              rng: np.random.Generator) -> tuple[float, float]:
    """Per-note pitch and level wobble.

    Every note draws from the rng even when the wobble is tiny, which is what
    makes two seeds produce two genuinely different renders. Arcade's voices are
    pure oscillators and would otherwise be identical for every seed, so
    `--count 5` would write the same file five times.
    """
    detuned = note + rng.normal(0.0, style.detune_cents) / 100.0
    gain = 10.0 ** (rng.normal(0.0, style.level_jitter_db) / 20.0)
    return detuned, gain


def _voiced(style: Style, note: float, duration: float,
            rng: np.random.Generator, brightness: float,
            with_body: bool) -> np.ndarray:
    """One note in a style: lead + optional body support."""
    note, gain = _humanise(style, note, rng)
    freq = e.midi_to_freq(note)
    out = style.lead(freq, duration, rng, brightness)
    if with_body and style.body_gain > 0.0:
        body = style.body(freq, duration, rng, brightness)
        out = e.mix(out, body * style.body_gain)
    return out * gain


def blip(style: Style, note: float, duration: float, rng: np.random.Generator,
         brightness: float = 1.0, with_body: bool = True) -> np.ndarray:
    """A single note."""
    return _voiced(style, note, duration * style.note_scale, rng, brightness,
                   with_body)


def arpeggio(style: Style, notes, rng: np.random.Generator,
             note_dur: float = 0.075, gap: float | None = None,
             brightness: float = 1.0, with_body: bool = True,
             accel: float = 1.0) -> np.ndarray:
    """Notes in sequence. The last one is allowed to ring on.

    `gap` defaults to the note duration (legato-ish stepping). `accel` < 1
    speeds the sequence up as it goes, which is what makes a run feel like it is
    arriving somewhere rather than plodding.
    """
    step = note_dur if gap is None else gap
    step *= style.gap_scale
    note_dur *= style.note_scale

    onsets = []
    t = 0.0
    for i in range(len(notes)):
        onsets.append(max(0.0, t))
        # Jitter the gap, not the onset, so errors don't accumulate into a
        # noticeably early or late final note.
        wobble = 1.0 + rng.normal(0.0, style.timing_jitter)
        t += step * (accel ** i) * max(0.25, wobble)

    tail = note_dur * style.tail_scale
    total = onsets[-1] + tail
    out = e.silence(total)
    for i, (onset, note) in enumerate(zip(onsets, notes)):
        # The final note rings for the tail; the rest only need their slot.
        last = i == len(notes) - 1
        dur = tail if last else min(tail, step * 2.4)
        voice = _voiced(style, note, max(dur, 0.02), rng, brightness, with_body)
        e.add_at(out, onset, voice, 1.0)
    return out


def glide(style: Style, note_from: float, note_to: float, duration: float,
          rng: np.random.Generator, brightness: float = 1.0,
          curve: float = 1.0, with_body: bool = False) -> np.ndarray:
    """A continuous pitch sweep between two notes."""
    n = e.samples(duration)
    note_from, gain = _humanise(style, note_from, rng)
    note_to, _ = _humanise(style, note_to, rng)
    f0 = e.midi_to_freq(note_from)
    f1 = e.midi_to_freq(note_to)
    # Sweep in the log domain: a linear Hz ramp sounds like it stalls at the top.
    freqs = f0 * (f1 / f0) ** (np.linspace(0.0, 1.0, n) ** curve)
    out = style.lead(freqs, duration, rng, brightness)
    if with_body and style.body_gain > 0.0:
        out = e.mix(out, style.body(freqs, duration, rng, brightness)
                    * style.body_gain)
    return out * gain


def chord(style: Style, notes, duration: float, rng: np.random.Generator,
          brightness: float = 1.0, spread: float = 0.012) -> np.ndarray:
    """Notes together, slightly rolled so it sounds struck rather than gated."""
    total = duration + spread * len(notes)
    out = e.silence(total)
    for i, note in enumerate(notes):
        voice = _voiced(style, note, duration, rng, brightness,
                        with_body=(i == 0))
        e.add_at(out, i * spread, voice, 1.0 / (1.0 + 0.35 * i))
    return out


def pulse_train(style: Style, note: float, count: int,
                rng: np.random.Generator, pulse_dur: float = 0.055,
                period: float = 0.13, brightness: float = 1.0) -> np.ndarray:
    """Repeated identical blips — urgency, warnings, ticking."""
    total = period * (count - 1) + pulse_dur * style.tail_scale + 0.02
    out = e.silence(total)
    for i in range(count):
        voice = _voiced(style, note, pulse_dur, rng, brightness, with_body=False)
        e.add_at(out, i * period, voice, 1.0)
    return out


def sparkle_tail(style: Style, duration: float, rng: np.random.Generator,
                 low: float = 3_500.0, high: float = 13_000.0) -> np.ndarray:
    """Style-controlled glitter. Returns silence for styles that don't glitter."""
    if style.sparkle_gain <= 0.0 or style.sparkle_count <= 0:
        return e.silence(duration)
    grains = e.sparkle(duration, rng, count=style.sparkle_count, low=low,
                       high=high, decay=0.028)
    return grains * style.sparkle_gain


def accent_at(style: Style, duration: float, rng: np.random.Generator,
              brightness: float = 1.0) -> np.ndarray:
    """The onset transient, scaled by the style's taste for it."""
    return style.accent(duration, rng, brightness) * style.accent_gain


def noise_crack(duration: float, rng: np.random.Generator,
                low: float = 700.0, high: float = 6_000.0,
                decay: float = 0.045) -> np.ndarray:
    """Breaking/tearing colour — style-independent, used by the negative cues."""
    crack = e.bandpass(e.noise(duration, rng), low, high, 2)
    return crack * e.exp_env(duration, decay=decay, attack=0.0008)
