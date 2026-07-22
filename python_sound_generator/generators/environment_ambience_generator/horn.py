"""Barge and fog horns, ported from lib/barge_horn.py.

Ported from an earlier barge-horn prototype and its synthesis notes.

A horn is compressed air driving a diaphragm at resonance, with the trumpet
flare acting as an impedance transformer. That gives a brassy, all-harmonics,
roughly sawtooth spectrum -- hence ten harmonics at a 1/n-ish rolloff, and a
280-650 Hz "body" formant from the bell itself. A pure sine sounds thin and
electronic because it has none of that: no even *and* odd harmonics, no
mechanical inharmonicity, no turbulence, no metal in the attack.

The temporal shape is the tell, and all of it is air pressure:

- **Attack** over 200-400 ms, preceded by a *chuff* of air rush before the
  diaphragm locks into resonance, and a slight rise in pitch as pressure
  stabilises.
- **Sustain** with a 2-5 Hz tremolo from pressure fluctuation.
- **Release** with the pitch sagging ~6% and the upper harmonics dying faster
  than the fundamental, so the tail darkens as it goes.

A **diaphone** is a different machine -- a slotted piston in a slotted cylinder
-- and as the air cuts off, the piston decelerates and emits the iconic
*grunt*: a drop to well under half pitch at the very end of the blast.

Blast lengths and patterns are COLREGS: a short blast is ~1 s, a prolonged one
4-6 s, and the pattern is a vessel telling you what it is doing. Frequencies
are low because low frequencies carry -- air absorbs 0.4 dB/km at 125 Hz and
32.8 dB/km at 4 kHz, which is also why distance here takes the high end off
first.
"""

from __future__ import annotations

import math
import random

from ..._audio import dsp


SR = dsp.SR
TAU = dsp.TAU

HARMONIC_AMPS = (1.000, 0.500, 0.350, 0.220, 0.150,
                 0.100, 0.065, 0.040, 0.025, 0.015)

PROLONGED = 5.0
SHORT = 1.0
GAP = 1.5

# What a vessel is saying. COLREGS Annex III.
PATTERNS: dict[str, tuple[float, ...]] = {
    "underway": (PROLONGED,),
    "stopped": (PROLONGED, PROLONGED),
    "towing": (PROLONGED, SHORT, SHORT),
    "danger": (SHORT, SHORT, SHORT, SHORT, SHORT),
    "single": (SHORT,),
}


def blast(rng: random.Random, duration: float = 5.0, freq_a: float = 150.0,
          freq_b: float = 0.0, distance: float = 0.3,
          diaphone: bool = True) -> list[float]:
    """One blast. `freq_b` of 0 is a single-trumpet horn."""

    n = dsp.samples(duration)
    attack_time = rng.uniform(0.20, 0.40)
    release_time = rng.uniform(0.30, 0.60)
    attack = dsp.samples(attack_time)
    release = dsp.samples(release_time)
    sustain = max(0, n - attack - release)
    if sustain <= 0:
        # The original crashes here rather than clamping; a 1 s short blast is
        # only safe because attack+release usually happens to fit.
        attack = max(1, int(n * 0.3))
        release = max(1, int(n * 0.4))
        sustain = max(0, n - attack - release)

    envelope = [0.0] * n
    for i in range(attack):
        envelope[i] = 0.5 * (1.0 - math.cos(math.pi * i / max(attack, 1)))
    for i in range(attack, attack + sustain):
        envelope[i] = 1.0
    for i in range(release):
        index = attack + sustain + i
        if index >= n:
            break
        envelope[index] = math.exp(-3.0 * i / max(release, 1))

    # Pitch is referenced to a fixed 250 Hz in the original, so the shift is a
    # ratio: ~-4% climbing away on the attack, ~-6% sagging on the release.
    pitch = [1.0] * n
    for i in range(attack):
        pitch[i] = 1.0 + (-10.0 + 10.0 * i / max(attack - 1, 1)) / 250.0
    for i in range(release):
        index = attack + sustain + i
        if index >= n:
            break
        pitch[index] = 1.0 + (-15.0 * i / max(release - 1, 1)) / 250.0

    trem_rate = rng.uniform(2.5, 4.5)
    trem_depth = rng.uniform(0.05, 0.12)
    trem_phase = rng.uniform(0.0, TAU)
    trem_rate2 = rng.uniform(1.5, 3.0)
    trem_phase2 = rng.uniform(0.0, TAU)
    tremolo = [
        1.0 + trem_depth * math.sin(TAU * trem_rate * i / SR + trem_phase)
        + trem_depth * 0.4 * math.sin(TAU * trem_rate2 * i / SR + trem_phase2)
        for i in range(n)
    ]

    out = [0.0] * n
    horns = [(freq_a, 0.6)] + ([(freq_b, 0.5)] if freq_b > 0.0 else [])
    for f0, level in horns:
        voice = [0.0] * n
        for index, amp in enumerate(HARMONIC_AMPS):
            number = index + 1
            if f0 * number * 1.02 > SR * 0.5:
                continue
            detune = rng.uniform(-0.5, 0.5) * number * 0.1
            phase = rng.uniform(0.0, TAU)
            for i in range(n):
                phase += TAU * (f0 * number * pitch[i] + detune) / SR
                value = amp * math.sin(phase)
                if number > 1 and i >= attack + sustain:
                    # Upper harmonics die faster in the release: the tail
                    # darkens rather than merely fading.
                    position = (i - attack - sustain) / max(release, 1)
                    value *= math.exp(-2.0 * (number - 1) * position)
                voice[i] += value
        delay = rng.uniform(0.0, 0.03) if f0 != freq_a else 0.0
        dsp.add_at(out, delay, voice, level)

    out = dsp.mul(out, dsp.mul(envelope, tremolo))

    # Air turbulence, with the chuff of air rushing before the diaphragm locks.
    noise = dsp.bandpass(dsp.noise(duration, rng), 200.0, 2_000.0, 3)
    for i in range(n):
        shape = 0.3
        if i < attack:
            shape = 0.3 + 0.7 * math.exp(-6.0 * i / max(attack - 1, 1))
        out[i] += noise[i] * shape * envelope[i] * 0.04

    if diaphone and duration >= 3.0 and release > 100:
        start = attack + sustain
        length = min(dsp.samples(0.8), n - start)
        if length > 1:
            phase = 0.0
            for i in range(length):
                frequency = freq_a * (1.0 - 0.6 * i / max(length - 1, 1))
                phase += TAU * frequency / SR
                shape = (math.sin(math.pi * i / max(length - 1, 1))
                         * math.exp(-2.0 * i / max(length - 1, 1)))
                out[start + i] += (math.sin(phase)
                                   + 0.4 * math.sin(2.0 * phase)) * shape * 0.35

    resonance = dsp.bandpass(out, 280.0, 650.0, 1)
    out = dsp.mix(out, dsp.scale(resonance, 0.15))

    if distance > 0.05:
        out = dsp.scale(out, 1.0 / (1.0 + distance * 2.0))
        out = dsp.lowpass(out, max(300.0, 8_000.0 / (1.0 + distance * 4.0)), 2)
    return out


def signal(rng: random.Random, seconds: float, intensity: float,
           pattern: str = "underway") -> list[float]:
    """A full fog signal. `intensity` is proximity; `pattern` is the message."""

    if not 0.05 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    if pattern not in PATTERNS:
        raise ValueError(
            f"unknown pattern {pattern!r}; expected one of {sorted(PATTERNS)}"
        )
    distance = 0.15 + (1.0 - intensity) * 2.5
    blasts = PATTERNS[pattern]
    span = sum(blasts) + GAP * (len(blasts) - 1) + 1.5
    out = dsp.silence(span)
    t = 0.0
    for duration in blasts:
        dsp.add_at(out, t, blast(rng, duration, 150.0, 0.0, distance,
                                 diaphone=True))
        t += duration + GAP
    return out
