"""Church bells, ported from lib/churchbell.py.

Ported from an earlier church-bell prototype and its synthesis notes.

A bell is not a string or an air column. It vibrates flexurally in three
dimensions, so its partials are **inharmonic**, and that is the whole identity
of the sound -- there is no series of integer multiples to fall back on. The
pitch you hear is not even in the list: the nominal, twelfth and upper octave
sit near 2:3:4, and the ear infers the missing fundamental an octave below the
nominal. That inferred note is the *strike tone*, and it is why `strike_freq`
names a partial the bell mostly does not sound on its own.

Three details do the work:

- **Split modes.** No real bell is perfectly symmetric, so each mode splits
  into two close frequencies and they beat. That slow wah is the difference
  between a bell and an organ; here each partial with a detune gets a second
  copy at `freq - 2*detune`, so the beat rate is exactly twice the detune.
- **Differential decay.** The hum rings for 25 s, the prime 12, the quint 4,
  the top partials 1-2. So a bell does not merely fade, it *changes* -- bright
  and clangorous at the strike, a warm drone ten seconds later.
- **Clapper bounce.** The clapper does not strike once. There is contact, then
  a weaker bounce a few milliseconds later, and that broadband pair is the
  percussive thunk that says metal was hit by something.

Air absorption is why a distant bell sounds warm: the upper partials go first.
That is modelled as a per-partial tilt keyed on the partial's ratio rather than
as a filter, which is the original's choice and is cheaper.
"""

from __future__ import annotations

import math
import random

from ..._audio import dsp


SR = dsp.SR
TAU = dsp.TAU

# (name, freq ratio to the strike note, amplitude, -60 dB time, detune Hz)
PARTIALS: tuple[tuple[str, float, float, float, float], ...] = (
    ("hum", 0.50, 0.40, 25.0, 0.0),
    ("prime", 1.00, 1.00, 12.0, 0.0),
    ("tierce", 1.189, 0.55, 8.0, 0.3),
    ("quint", 1.506, 0.40, 4.0, 0.0),
    ("nominal", 2.00, 0.70, 6.0, 0.2),
    ("deciem", 2.514, 0.25, 3.0, 0.5),
    ("undeciem", 3.011, 0.18, 2.5, 0.7),
    ("duodeciem", 4.0, 0.12, 2.0, 0.4),
    ("superoctave", 5.33, 0.06, 1.5, 1.0),
    ("high", 6.17, 0.04, 1.0, 1.2),
)

# Westminster, as ratios from E. 0.891 = -2 semitones, 0.794 = -4, 0.595 = -9.
QUARTERS: dict[int, tuple[tuple[float, ...], ...]] = {
    1: ((1.0, 0.891, 0.794, 0.595),),
    2: ((0.794, 1.0, 0.891, 0.595),
        (1.0, 0.891, 0.794, 0.595)),
    3: ((1.0, 0.794, 0.891, 1.0),
        (0.595, 0.891, 1.0, 0.794),
        (1.0, 0.891, 0.794, 0.595)),
    4: ((1.0, 0.794, 0.891, 1.0),
        (0.595, 0.891, 1.0, 0.794),
        (1.0, 0.891, 0.794, 0.595),
        (0.595, 0.794, 0.891, 1.0)),
}

CHIME_GAP = 1.8
PHRASE_GAP = 2.5


def strike(rng: random.Random, freq: float = 392.0, duration: float = 10.0,
           distance: float = 0.0) -> list[float]:
    """One strike of a bell whose strike note is `freq`.

    The clapper's noise is seeded from the pitch rather than from `rng`, which
    is the original's choice and worth keeping: it is one lump of metal hitting
    another, and a bell whose clapper timbre changed between strikes would be
    two bells.

    What does vary is how hard it was hit. A harder strike is louder *and*
    relatively brighter -- more energy into the short upper partials, which
    decay away within a second or two and leave the same hum behind either way.
    Without this a bell renders identically on every seed, and a bank of three
    "variations" would be three copies of one file.
    """

    force = rng.uniform(0.85, 1.15)
    n = dsp.samples(duration)
    out = [0.0] * n
    nyquist = SR * 0.5

    for _, ratio, amp, decay, detune in PARTIALS:
        partial_freq = freq * ratio + detune
        if partial_freq > nyquist:
            continue
        k = 6.908 / decay
        attenuation = 1.0 / (1.0 + distance * ratio * 0.3)
        # Upper partials answer the strike force; the hum barely notices.
        tilt = 1.0 + (force - 1.0) * min(ratio, 4.0) * 0.5
        gain = amp * attenuation * tilt
        step = TAU * partial_freq / SR
        beat_step = TAU * (partial_freq - 2.0 * detune) / SR
        for i in range(n):
            envelope = math.exp(-k * i / SR)
            if envelope < 1e-5:
                break
            out[i] += gain * envelope * math.sin(step * i)
            if detune > 0.0:
                out[i] += gain * 0.3 * envelope * math.sin(beat_step * i)

    # Clapper: contact, then a weaker bounce ~5 ms later. Seeded from the pitch
    # in the original so one bell always sounds like itself; kept, because a
    # bell whose clapper noise changes per render is two bells.
    impact_rng = random.Random(int(freq * 100) & 0xFFFFFFFF)
    impact_len = min(dsp.samples(0.002), n)
    for i in range(impact_len):
        out[i] += (impact_rng.gauss(0.0, 1.0)
                   * math.exp(-8.0 * i / max(impact_len - 1, 1)) * 0.15 * force)
    bounce_rng = random.Random(int(freq * 200) & 0xFFFFFFFF)
    bounce_at = dsp.samples(0.005)
    bounce_len = dsp.samples(0.001)
    for i in range(bounce_len):
        index = bounce_at + i
        if index >= n:
            break
        out[index] += (bounce_rng.gauss(0.0, 1.0)
                       * math.exp(-10.0 * i / max(bounce_len - 1, 1)) * 0.04)

    attack = min(dsp.samples(0.001), n)
    for i in range(attack):
        out[i] *= i / max(attack - 1, 1)

    warble_rate = 1.2 + 0.3 * (freq / 400.0)
    warble_phase = rng.uniform(0.0, TAU)
    for i in range(n):
        t = i / SR
        out[i] *= ((1.0 + 0.06 * math.sin(TAU * warble_rate * t + warble_phase))
                   * (1.0 + 0.03 * math.sin(TAU * warble_rate * 0.37 * t)))
    return out


def single(rng: random.Random, seconds: float,
           intensity: float) -> list[float]:
    """One bell. `intensity` is proximity: distant and warm to close and bright."""

    if not 0.05 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    distance = (1.0 - intensity) * 3.0
    return strike(rng, 392.0, 10.0, distance)


def chime(rng: random.Random, seconds: float, intensity: float,
          quarter: int = 4, hour: int = 0) -> list[float]:
    """A Westminster quarter, optionally followed by the hour tolling.

    Quarter 1 is one phrase, quarter 4 is all four -- which is how a clock
    tells you how far into the hour you are without you having to count.
    """

    if not 0.05 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    if quarter not in QUARTERS:
        raise ValueError(f"quarter must be one of {sorted(QUARTERS)}")
    if not 0 <= hour <= 12:
        raise ValueError("hour must be between 0 and 12")

    distance = (1.0 - intensity) * 3.0
    phrases = QUARTERS[quarter]
    span = len(phrases) * (4 * CHIME_GAP + PHRASE_GAP) + 5.0
    if hour:
        span += 3.5 + hour * 3.2 + 5.0
    out = dsp.silence(span)

    t = 0.0
    for index, phrase in enumerate(phrases):
        for ratio in phrase:
            note = strike(rng, 392.0 * ratio, 4.0, distance)
            dsp.add_at(out, t, note, 0.45)
            t += CHIME_GAP
        if index < len(phrases) - 1:
            t += PHRASE_GAP
    if hour:
        # The hour bell is an octave down, and the tolls overlap: each strike
        # rings 8 s and they come 3.2 s apart, so they pile into a wash rather
        # than a countdown. That overlap is the sound of a tower.
        t += 3.5
        for _ in range(hour):
            dsp.add_at(out, t, strike(rng, 196.0, 8.0, distance), 0.65)
            t += 3.2
    return out
