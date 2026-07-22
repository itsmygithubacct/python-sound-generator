"""The three timbre houses a cue can be rendered in.

A cue decides *what happens musically* — two notes rising, an arpeggio, a
semitone falling. A style decides *what it sounds like*. Keeping them apart is
what lets `render coin --style acoustic` and `--style arcade` be the same
gesture wearing different clothes, and it means a new style costs one table
here rather than eighteen new renderers.

A style supplies three voices:

- ``lead``   the note you actually hear the melody in
- ``body``   low support under a lead note; may be silent
- ``accent`` noise/transient colour placed at an onset

plus the tuning knobs the gestures read (note spacing, decay, sparkle).

The voices deliberately do not share an implementation. Arcade is an additive
square/triangle, acoustic is a struck bar, sci-fi is detuned saws through a
resonant filter — these are different synthesis methods, not one method with
different parameters, and pretending otherwise would flatten all three into the
same sound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import engine as e


# --------------------------------------------------------------------------
# arcade — additive square/triangle, no noise, fast and bright
# --------------------------------------------------------------------------

def _arcade_lead(freq, duration: float, rng: np.random.Generator,
                 brightness: float = 1.0) -> np.ndarray:
    """Square lead with a triangle underneath to round the edge off."""
    square = e.osc("square", freq, duration)
    triangle = e.osc("triangle", freq, duration)
    tone = 0.72 * square + 0.42 * triangle
    # Percussive but not clicky: 1.5 ms attack, then a decay that leaves the
    # note ringing for most of its slot.
    env = e.exp_env(duration, decay=duration * 0.55, attack=0.0015)
    return tone * env * (0.85 + 0.15 * brightness)


def _arcade_body(freq, duration: float, rng: np.random.Generator,
                 brightness: float = 1.0) -> np.ndarray:
    """An octave-down triangle: weight without mud."""
    sub = e.osc("triangle", np.asarray(e._freqs(freq, e.samples(duration))) * 0.5,
                duration)
    return sub * e.exp_env(duration, decay=duration * 0.40, attack=0.002) * 0.30


def _arcade_accent(duration: float, rng: np.random.Generator,
                   brightness: float = 1.0) -> np.ndarray:
    """Arcade has no noise floor by design — a tiny tick stands in."""
    tick = e.osc("square", 5_200.0 * brightness, min(duration, 0.006))
    out = e.silence(duration)
    e.add_at(out, 0.0, tick * e.exp_env(min(duration, 0.006), 0.0015), 0.25)
    return out


# --------------------------------------------------------------------------
# acoustic — struck bars and plucked strings
# --------------------------------------------------------------------------

# Glockenspiel/celesta bar partials. A struck bar is inharmonic: these ratios
# are the classic transverse-mode series, and they are why a bell reads as a
# bell rather than an organ. Deliberately not integer multiples.
_BAR_MODES = ((1.0, 1.00), (2.76, 0.36), (5.40, 0.16), (8.93, 0.07))


def _acoustic_lead(freq, duration: float, rng: np.random.Generator,
                   brightness: float = 1.0) -> np.ndarray:
    """Struck metal bar — glockenspiel-ish."""
    f0 = float(np.mean(e._freqs(freq, 8)))
    modes = []
    for ratio, gain in _BAR_MODES:
        # Higher modes die faster; that decay spread is most of the timbre.
        decay = duration * 0.85 / (1.0 + 1.6 * (ratio - 1.0))
        modes.append((f0 * ratio, max(decay, 0.008), gain * brightness))
    bar = e.modal_ring(duration, modes, rng)
    # The mallet contact itself: a few ms of bright noise, gone almost at once.
    strike = e.highpass(e.noise(duration, rng), 4_000.0, 2)
    strike *= e.exp_env(duration, decay=0.0035)
    return bar * 0.9 + strike * 0.10


def _acoustic_body(freq, duration: float, rng: np.random.Generator,
                   brightness: float = 1.0) -> np.ndarray:
    """A plucked harp string an octave down, for warmth under the bar."""
    f0 = float(np.mean(e._freqs(freq, 8))) * 0.5
    string = e.pluck(duration, f0, rng, damping=0.94, brightness=0.55)
    return string * e.exp_env(duration, decay=duration * 0.6, attack=0.001) * 0.34


def _acoustic_accent(duration: float, rng: np.random.Generator,
                     brightness: float = 1.0) -> np.ndarray:
    """Soft mallet noise."""
    hit = e.bandpass(e.noise(duration, rng), 1_800.0, 7_000.0 * brightness, 2)
    return hit * e.exp_env(duration, decay=0.010) * 0.5


# --------------------------------------------------------------------------
# sci-fi — detuned saws, resonant filter, sub
# --------------------------------------------------------------------------

def _scifi_lead(freq, duration: float, rng: np.random.Generator,
                brightness: float = 1.0) -> np.ndarray:
    """Two saws a few cents apart through a resonant lowpass that opens."""
    n = e.samples(duration)
    f = e._freqs(freq, n)
    a = e.osc("saw", f * 1.0018, duration)
    b = e.osc("saw", f * 0.9982, duration, phase0=1.1)
    tone = 0.5 * (a + b)
    # Filter opens over the first third of the note: the classic synth "pluck"
    # gesture. Tracks pitch so high notes stay proportionally bright.
    cutoff = f * (2.5 + 6.0 * brightness) * e.ramp(duration, 0.55, 1.6, curve=0.45)
    tone = e.resonant_sweep(tone, np.clip(cutoff, 60.0, e.SR * 0.45), q=2.6,
                            mode="lowpass")
    return tone * e.exp_env(duration, decay=duration * 0.5, attack=0.004)


def _scifi_body(freq, duration: float, rng: np.random.Generator,
                brightness: float = 1.0) -> np.ndarray:
    """Sine sub two octaves down — the weight sci-fi pickups lean on."""
    f0 = float(np.mean(e._freqs(freq, 8))) * 0.25
    sub = e.osc("sine", max(f0, 35.0), duration)
    return sub * e.exp_env(duration, decay=duration * 0.45, attack=0.003) * 0.45


def _scifi_accent(duration: float, rng: np.random.Generator,
                  brightness: float = 1.0) -> np.ndarray:
    """A resonant noise blip — the UI 'tick'."""
    blip = e.noise(duration, rng)
    blip = e.resonant_sweep(blip, e.ramp(duration, 2_400.0 * brightness, 900.0),
                            q=6.0)
    return blip * e.exp_env(duration, decay=0.020) * 0.7


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

Voice = Callable[..., np.ndarray]


@dataclass(frozen=True)
class Style:
    """One timbre house, plus the knobs the gestures read."""

    label: str
    description: str
    lead: Voice
    body: Voice
    accent: Voice
    # Gesture tuning
    note_scale: float       # multiplies per-note duration in arpeggios
    gap_scale: float        # multiplies the gap between arpeggio notes
    tail_scale: float       # how long the final note is allowed to ring
    sparkle_gain: float     # glitter on top; 0 disables
    sparkle_count: int
    body_gain: float        # how much low support to mix under leads
    accent_gain: float
    drive: float            # soft-limiter drive in the master chain
    # Humanisation. These are what make `--count` mean anything: without them a
    # style whose voices draw no random numbers (arcade does not — it is pure
    # oscillators) renders byte-identical files for every seed. Kept small
    # enough to read as the same sound played twice, not as a detuned mistake.
    detune_cents: float     # 1 sigma pitch wobble per note
    level_jitter_db: float  # 1 sigma level wobble per note
    timing_jitter: float    # 1 sigma of the inter-note gap


STYLES: dict[str, Style] = {
    "arcade": Style(
        label="Arcade",
        description="square/triangle blips, tight and bright, no noise floor",
        lead=_arcade_lead, body=_arcade_body, accent=_arcade_accent,
        note_scale=0.80, gap_scale=0.75, tail_scale=1.5,
        sparkle_gain=0.0, sparkle_count=0,
        body_gain=0.55, accent_gain=0.6, drive=1.5,
        # Tightest of the three: chiptune reads as precise, so barely wobble it.
        detune_cents=3.5, level_jitter_db=0.5, timing_jitter=0.020,
    ),
    "acoustic": Style(
        label="Acoustic",
        description="struck glockenspiel bars over plucked harp, long tails",
        lead=_acoustic_lead, body=_acoustic_body, accent=_acoustic_accent,
        note_scale=1.25, gap_scale=1.0, tail_scale=2.6,
        sparkle_gain=0.10, sparkle_count=14,
        body_gain=0.70, accent_gain=0.35, drive=1.2,
        # A struck bar is played by a hand: the loosest timing of the three.
        detune_cents=7.0, level_jitter_db=1.2, timing_jitter=0.045,
    ),
    "scifi": Style(
        label="Sci-fi",
        description="detuned saws through a resonant filter, sub underneath",
        lead=_scifi_lead, body=_scifi_body, accent=_scifi_accent,
        note_scale=1.0, gap_scale=0.85, tail_scale=1.8,
        sparkle_gain=0.16, sparkle_count=20,
        body_gain=0.85, accent_gain=0.75, drive=1.35,
        # Detuning is already the point of the timbre; a little more is free.
        detune_cents=9.0, level_jitter_db=0.9, timing_jitter=0.030,
    ),
}

DEFAULT_STYLE = "arcade"


def get_style(name: str) -> Style:
    try:
        return STYLES[name]
    except KeyError:
        raise ValueError(
            f"unknown style {name!r}; choose from {', '.join(STYLES)}"
        ) from None
