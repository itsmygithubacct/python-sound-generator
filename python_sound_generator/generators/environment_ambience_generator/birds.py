"""Birdsong, ported from birdsong.py / lib/synthesis.py / lib/species/.

Ported from earlier birdsong synthesis prototypes and their species research.

Two engines, chosen per species, which is what the research actually calls for:

- :mod:`.syrinx` -- the Mindlin labial model, an honest nonlinear oscillator.
  Its harmonic structure falls out of the physics. Used where a species wants
  richness in the voice itself, and for biphonation, which nothing else can do.
- **FM** -- a swept sine with a weak second harmonic and a little shaped noise.
  Used where the species really is close to a pure tone. A hermit thrush is
  flute-like; running it through a nonlinear oscillator would only dirty it.

The rule that matters in both: **phase must be integrated, not computed as
2*pi*f*t**. A glide written the naive way steps its phase discontinuously every
sample and the note fizzes.

Two biological facts drive the rest. The syrinx sits at the tracheal
bifurcation and has *two independently controlled labia*, so a bird can sing
two voices at once -- that is why a wood thrush sounds ethereal, and it is
literally one bird in two-part harmony. And birds are fast: a cardinal sweeps
over an octave inside a 130 ms syllable, which is why "a few random glides"
never sounds like song.

The noise in the FM path is bandpassed to *track the fundamental* rather than
sprayed across the spectrum, because it is meant to sit under the harmonic
structure as breath, not beside it as hiss.
"""

from __future__ import annotations

import math
import random

from ..._audio import dsp

from ..._audio import loops as loopdsp
from . import syrinx


SR = dsp.SR
TAU = dsp.TAU


def _sweep(duration: float, f_start: float, f_end: float,
           method: str = "linear") -> list[float]:
    """A frequency contour. The shape of a glide is species vocabulary."""

    n = dsp.samples(duration)
    if n == 1:
        return [f_end]
    out = []
    for i in range(n):
        position = i / (n - 1)
        if method == "linear":
            out.append(f_start + (f_end - f_start) * position)
        elif method == "quadratic":
            out.append(f_start + (f_end - f_start) * position ** 2)
        elif method == "sinusoidal":
            out.append(f_start + (f_end - f_start) * 0.5
                       * (1.0 - math.cos(math.pi * position)))
        elif method == "logarithmic":
            ratio = max(f_end, 1.0) / max(f_start, 1.0)
            out.append(f_start * ratio ** position)
        else:
            raise ValueError(f"unknown sweep {method!r}")
    return out


def _envelope(n: int, attack: float = 0.10,
              release: float = 0.15) -> list[float]:
    """Raised cosine. Real notes have no clicks, and no pad-like fades either."""

    out = [1.0] * n
    a = max(1, int(n * attack))
    r = max(1, int(n * release))
    for i in range(min(a, n)):
        out[i] = 0.5 * (1.0 - math.cos(math.pi * i / a))
    for i in range(min(r, n)):
        out[n - 1 - i] *= 0.5 * (1.0 - math.cos(math.pi * i / r))
    return out


def _shaped_noise(rng: random.Random, duration: float, centre: float,
                  width: float) -> list[float]:
    """Breath: noise following the voice, not sitting beside it."""

    low = max(50.0, centre - width / 2.0)
    high = min(SR * 0.49, centre + width / 2.0)
    if high <= low:
        return dsp.silence(duration)
    out = dsp.bandpass(dsp.noise(duration, rng), low, high, 4)
    peak = max((abs(v) for v in out), default=0.0)
    return dsp.scale(out, 1.0 / peak) if peak > 0.0 else out


def fm_note(rng: random.Random, duration: float, f_start: float, f_end: float,
            method: str = "linear", amplitude: float = 0.8,
            vibrato_rate: float = 0.0, vibrato_depth: float = 0.0,
            noise_mix: float = 0.0, harmonic2: float = 0.0,
            attack: float = 0.10, release: float = 0.15,
            tremolo_rate: float = 0.0,
            tremolo_depth: float = 0.0) -> list[float]:
    """A sung note as a swept sine with breath on it."""

    freqs = _sweep(duration, f_start, f_end, method)
    n = len(freqs)
    if vibrato_rate > 0.0 and vibrato_depth > 0.0:
        freqs = [f + vibrato_depth * math.sin(TAU * vibrato_rate * i / SR)
                 for i, f in enumerate(freqs)]
    tone = dsp.sine_track(freqs)
    if harmonic2 > 0.0:
        second = dsp.sine_track([f * 2.0 for f in freqs])
        tone = [a + harmonic2 * b for a, b in zip(tone, second)]
    if noise_mix > 0.0:
        centre = sum(freqs) / len(freqs)
        breath = _shaped_noise(rng, duration, centre, centre * 0.8)
        tone = [a + noise_mix * b for a, b in zip(tone, breath)]
    env = _envelope(n, attack, release)
    if tremolo_rate > 0.0 and tremolo_depth > 0.0:
        env = [e * (1.0 - tremolo_depth * (0.5 + 0.5
                    * math.sin(TAU * tremolo_rate * i / SR)))
               for i, e in enumerate(env)]
    return [t * e * amplitude for t, e in zip(tone, env)]


def syrinx_note(rng: random.Random, duration: float, f_start: float,
                f_end: float, method: str = "sinusoidal",
                amplitude: float = 0.8, alpha: float = 0.20,
                beta: float = 0.10, vibrato_rate: float = 0.0,
                vibrato_depth: float = 0.0, attack: float = 0.10,
                release: float = 0.15) -> list[float]:
    """A sung note from the physical labial oscillator.

    (alpha, beta) is the voice -- air-sac pressure and muscle tension. The
    pitch rides on gamma, which only rescales time, so the timbre does not
    drift as the bird glides.
    """

    freqs = _sweep(duration, f_start, f_end, method)
    if vibrato_rate > 0.0 and vibrato_depth > 0.0:
        freqs = [f + vibrato_depth * math.sin(TAU * vibrato_rate * i / SR)
                 for i, f in enumerate(freqs)]
    tone = syrinx.note(freqs, syrinx.voice(alpha, beta))
    env = _envelope(len(freqs), attack, release)
    return [t * e * amplitude for t, e in zip(tone, env)]


def _concat(parts: list[list[float]], gap: float) -> list[float]:
    out: list[float] = []
    filler = [0.0] * dsp.samples(gap)
    for index, part in enumerate(parts):
        out.extend(part)
        if index < len(parts) - 1:
            out.extend(filler)
    return out


# --------------------------------------------------------------------------
# species
# --------------------------------------------------------------------------

def blackbird(rng: random.Random) -> list[float]:
    """Melodic motif, then a twitter. The alternation is the species.

    The motif runs on the physical syrinx -- this is the phrase the original
    wanted nonlinear richness for -- and the twitter stays on FM with 40%
    noise, because a blackbird's twitter genuinely is noisy.
    """

    notes = []
    base = rng.uniform(1_300.0, 3_500.0)
    for _ in range(rng.randint(2, 4)):
        duration = rng.uniform(0.20, 0.45)
        start = base + rng.uniform(-200.0, 200.0)
        end = max(1_300.0, min(6_000.0, start + rng.uniform(-500.0, 800.0)))
        notes.append(syrinx_note(
            rng, duration, start, end, "sinusoidal", 0.8,
            vibrato_rate=6.0, vibrato_depth=20.0, attack=0.10, release=0.18,
        ))
    motif = _concat(notes, 0.08)

    twitters = []
    twitter_base = rng.uniform(4_000.0, 7_000.0)
    for _ in range(rng.randint(4, 10)):
        duration = rng.uniform(0.03, 0.07)
        end = twitter_base + rng.uniform(500.0, 1_500.0)
        tone = fm_note(rng, duration, twitter_base, end, "linear", 1.0,
                       attack=0.05, release=0.08)
        noise = dsp.mul(_shaped_noise(rng, duration, twitter_base, 2_000.0),
                        _envelope(len(tone), 0.05, 0.08))
        twitters.append([0.6 * a + 0.4 * b for a, b in zip(tone, noise)])
    twitter = dsp.scale(_concat(twitters, 0.025), 0.7)
    return _concat([motif, twitter], 0.4)


def robin(rng: random.Random) -> list[float]:
    """A harmonic stack -- fundamental plus 2nd and 4th. Bright and clear."""

    notes = []
    base = rng.uniform(3_500.0, 4_500.0)
    for _ in range(rng.randint(4, 8)):
        if rng.random() < 0.6:
            duration = rng.uniform(0.15, 0.35)
            start = base + rng.uniform(-500.0, 500.0)
            end = max(2_000.0, min(6_000.0, start + rng.uniform(-400.0, 600.0)))
            freqs = _sweep(duration, start, end, "linear")
            rate = rng.uniform(6.0, 12.0)
            depth = rng.uniform(15.0, 40.0)
            freqs = [f + depth * math.sin(TAU * rate * i / SR)
                     for i, f in enumerate(freqs)]
            tone = dsp.sine_track(freqs)
            for harmonic, gain in ((2.0, 0.35), (4.0, 0.15)):
                # Skip a harmonic that would alias rather than fold it back in.
                if max(freqs) * harmonic >= SR * 0.475:
                    continue
                part = dsp.sine_track([f * harmonic for f in freqs])
                tone = [a + gain * b for a, b in zip(tone, part)]
            env = _envelope(len(tone), 0.08, 0.15)
            env = [e * (1.0 - 0.12 * (0.5 + 0.5 * math.sin(TAU * 14.0 * i / SR)))
                   for i, e in enumerate(env)]
            notes.append([t * e * 0.75 for t, e in zip(tone, env)])
        else:
            duration = rng.uniform(0.05, 0.12)
            notes.append(fm_note(rng, duration, base * 1.5, base * 1.5 + 400.0,
                                 "quadratic", 0.6, noise_mix=0.12,
                                 attack=0.05, release=0.08))
    return _concat(notes, 0.06)


def nightingale(rng: random.Random) -> list[float]:
    """Long, varied, and nocturnal: whistles, trills, and a hard rattle."""

    notes = []
    for _ in range(rng.randint(3, 7)):
        pick = rng.random()
        if pick < 0.4:
            duration = rng.uniform(0.25, 0.6)
            start = rng.uniform(1_800.0, 2_600.0)
            notes.append(fm_note(rng, duration, start, start * rng.uniform(1.4, 2.2),
                                 "sinusoidal", 0.85, vibrato_rate=5.0,
                                 vibrato_depth=25.0, harmonic2=0.10,
                                 noise_mix=0.05, attack=0.12, release=0.2))
        elif pick < 0.75:
            base = rng.uniform(3_000.0, 4_800.0)
            for _ in range(rng.randint(5, 12)):
                notes.append(fm_note(rng, rng.uniform(0.02, 0.05), base,
                                     base * 1.15, "linear", 0.7,
                                     attack=0.1, release=0.15))
        else:
            base = rng.uniform(1_200.0, 1_900.0)
            notes.append(syrinx_note(rng, rng.uniform(0.15, 0.3), base,
                                     base * 1.1, "linear", 0.8,
                                     alpha=0.40, beta=0.30))
    return _concat(notes, 0.07)


def house_wren(rng: random.Random) -> list[float]:
    """A fast tumbling trill.

    Wrens trade trill speed against sweep width -- roughly
    bandwidth = -168.5 * rate + 6019 Hz. Honouring that tradeoff is part of
    why a fast trill reads as a bird and not as an arpeggiator.
    """

    notes = []
    rate = rng.uniform(12.0, 22.0)
    bandwidth = max(400.0, -168.5 * rate + 6_019.0)
    base = rng.uniform(2_500.0, 3_800.0)
    for _ in range(rng.randint(14, 26)):
        duration = 0.6 / rate
        start = base + rng.uniform(-200.0, 200.0)
        notes.append(fm_note(rng, duration, start, start + bandwidth * 0.25,
                             "quadratic", 0.7, noise_mix=0.08,
                             attack=0.08, release=0.12))
    return _concat(notes, 0.4 / rate)


def wood_thrush(rng: random.Random) -> list[float]:
    """Biphonation: two labia, two voices, one bird.

    This is what the syrinx is *for*. The two sides run as independent limit
    cycles at a real interval, not one voice with a chorus on it -- so the
    beating between them is physical rather than an effect.
    """

    notes = []
    for _ in range(rng.randint(2, 4)):
        duration = rng.uniform(0.3, 0.55)
        low = rng.uniform(1_800.0, 2_600.0)
        # A just interval: the two voices are separate oscillators, so this is
        # harmony rather than detuning.
        high = low * rng.choice((1.5, 1.6, 1.25))
        left = syrinx_note(rng, duration, low, low * rng.uniform(0.95, 1.1),
                           "sinusoidal", 0.55, alpha=0.20, beta=0.10)
        right = syrinx_note(rng, duration, high, high * rng.uniform(0.95, 1.1),
                            "sinusoidal", 0.45, alpha=0.05, beta=0.30)
        offset = dsp.samples(0.002)
        both = list(left)
        for i, value in enumerate(right):
            if i + offset < len(both):
                both[i + offset] += value
        notes.append(both)
    return _concat(notes, 0.25)


def hermit_thrush(rng: random.Random) -> list[float]:
    """Flute-like: harmonics 3-12 over a fundamental it never sounds.

    Pure integer ratios at 1/h amplitude, no FM. The species sits at the exact
    opposite end of the same engine from the blackbird's twitter.
    """

    notes = []
    for _ in range(rng.randint(2, 4)):
        duration = rng.uniform(0.2, 0.4)
        fundamental = rng.uniform(350.0, 520.0)
        tone = [0.0] * dsp.samples(duration)
        for harmonic in range(3, 13):
            freq = fundamental * harmonic
            if freq >= SR * 0.475:
                break
            part = dsp.sine(freq, duration, rng.uniform(0.0, TAU))
            tone = [a + b / harmonic for a, b in zip(tone, part)]
        env = _envelope(len(tone), 0.12, 0.25)
        notes.append([t * e * 0.6 for t, e in zip(tone, env)])
    return _concat(notes, 0.35)


SPECIES = {
    "blackbird": blackbird,
    "robin": robin,
    "nightingale": nightingale,
    "house_wren": house_wren,
    "wood_thrush": wood_thrush,
    "hermit_thrush": hermit_thrush,
}

# bout size, gap within a bout, gap between bouts
BOUTS: dict[str, tuple[tuple[int, int], tuple[float, float], tuple[float, float]]] = {
    "blackbird": ((2, 4), (0.5, 2.0), (30.0, 90.0)),
    "nightingale": ((3, 6), (0.3, 1.5), (20.0, 60.0)),
    "wood_thrush": ((2, 3), (1.0, 3.0), (40.0, 120.0)),
    "robin": ((2, 4), (0.5, 1.5), (25.0, 70.0)),
    "house_wren": ((3, 6), (0.2, 0.8), (15.0, 45.0)),
    "hermit_thrush": ((2, 3), (1.0, 3.0), (30.0, 90.0)),
}


def _distant(signal: list[float], distance: float) -> list[float]:
    """Range: quieter, and duller first.

    Air absorbs the high end with distance, so a far bird loses its top before
    it loses its level. Skipping the filter and only turning it down is what
    makes a distant bird sound like a close bird played quietly.
    """

    out = dsp.scale(signal, 0.6 / (1.0 + distance * 3.0))
    if distance > 0.3:
        cutoff = max(600.0, 8_000.0 * (1.0 - distance * 0.8))
        out = dsp.lowpass(out, cutoff, 4 if distance > 0.6 else 2)
    return out


def chorus(rng: random.Random, seconds: float, intensity: float,
           species: tuple[str, ...] | None = None) -> list[float]:
    """Birds placed across a loop, at range, in bouts.

    Density is a divisor on silence, as in the original: it does not make each
    bird sing more, it makes the gaps between bouts shorter. Birds sing in
    bouts and then shut up, and the gaps are most of what a wood sounds like.
    """

    if not 0.05 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    names = species or ("blackbird", "robin", "nightingale", "house_wren",
                        "wood_thrush")
    span = loopdsp.period(seconds)
    out = dsp.silence(seconds)
    density = max(0.05, intensity)

    for name in names:
        if name not in SPECIES:
            raise ValueError(
                f"unknown species {name!r}; expected one of {sorted(SPECIES)}"
            )
        size, intra, inter = BOUTS[name]
        distance = rng.uniform(0.0, 1.0)
        t = rng.uniform(0.0, span)
        guard = 0
        while guard < 64:
            guard += 1
            for _ in range(rng.randint(*size)):
                song = _distant(SPECIES[name](rng), distance)
                loopdsp.add_wrapped(out, t % span, song)
                t += len(song) / SR + rng.uniform(*intra)
            t += rng.uniform(*inter) / density
            if t > span:
                break
    return out
