"""Ocean and surf, ported from ocean_noise.py.

Ported from an earlier NumPy/SciPy ocean prototype. That version is stereo, 48 kHz,
and runs as a stream with fades on both ends. This is stdlib, mono, 44.1 kHz,
and loops. The model is otherwise the same one, and the parts that make it
convincing are kept intact:

- **Waves come in sets.** Three to eight of them, heights drawn from a Rayleigh
  distribution and sorted to peak in the middle, then a gap of quieter water.
  Evenly spaced waves of equal size read as a machine immediately.
- **Each wave rises fast and falls slow.** A ~30% half-cosine rise into a ~70%
  fall; symmetric waves sound like a synthesiser breathing.
- **Louder is brighter.** Cutoff tracks wave height, because a wave that is
  loud is loud *by breaking*, and breaking is what makes the high end.
- **The hiss arrives late.** Surf follows the crest by 0.3-0.8 s: the wave
  breaks, and *then* you hear the spray. Getting this causality wrong is what
  makes synthetic surf sound like noise with an envelope on it.
- **It never goes silent.** A background floor under everything, always.

The one structural change loops force: the original walks time forward laying
waves down until it runs out of buffer, and the last wave is simply cut off.
A loop cannot do that -- the cut would be the seam. So the set is built first
and then rescaled to tile the period exactly (see `_wave_envelope`), which
moves each wave's length by a percent or two and is inaudible against periods
that were random to begin with.
"""

from __future__ import annotations

import math
import random

from ..._audio import dsp

from ..._audio import loops as loopdsp


SR = dsp.SR


def _rayleigh(rng: random.Random, scale: float) -> float:
    """Rayleigh-distributed wave height.

    Real sea-surface elevation is Gaussian, which makes wave *height* -- the
    magnitude of two quadrature components -- Rayleigh. It is the standard
    distribution in ocean engineering and it is why the occasional wave is
    much bigger than its neighbours rather than the set being uniform.
    numpy has this built in; the stdlib has Weibull, and Rayleigh(s) is
    exactly Weibull(shape 2, scale s*sqrt(2)).
    """

    return rng.weibullvariate(scale * math.sqrt(2.0), 2.0)


def _one_wave(strength: float, samples: int, rise_fraction: float,
              background: float) -> list[float]:
    """One wave: fast half-cosine rise, slow exponential/cosine fall."""

    rise = max(1, int(samples * rise_fraction))
    fall = max(1, samples - rise)
    wave = [0.0] * samples
    for i in range(min(rise, samples)):
        wave[i] = strength * 0.5 * (1.0 - math.cos(math.pi * i / rise))
    for i in range(fall):
        index = rise + i
        if index >= samples:
            break
        position = i / fall
        wave[index] = (strength * 0.6 * math.exp(-3.0 * position)
                       + strength * 0.4 * 0.5 * (1.0 + math.cos(math.pi * position)))
    return [value + background for value in wave]


def _set_heights(rng: random.Random, count: int) -> list[float]:
    """Rayleigh heights arranged to build to the middle of the set and fall."""

    raw = sorted(_rayleigh(rng, 0.7) for _ in range(count))
    heights = [0.0] * count
    middle = count // 2
    for j in range(count):
        # Deal the tallest to the centre and work outwards, alternating sides.
        target = middle + j // 2 if j % 2 == 0 else middle - (j + 1) // 2
        if 0 <= target < count:
            heights[target] = raw[count - 1 - j]
    peak = max(heights) or 1.0
    return [value / peak for value in heights]


def _wave_envelope(seconds: float, rng: random.Random,
                   intensity: float) -> list[float]:
    """Wave height over one loop, tiling the period exactly."""

    span = loopdsp.period(seconds)
    minimum = 5.0 / (0.4 + 0.6 * intensity)
    maximum = 14.0 / (0.4 + 0.6 * intensity)
    background = 0.06 + 0.04 * intensity

    # Lay out the sets as (duration, builder) first, with no reference to the
    # buffer, then stretch the whole plan onto it.
    plan: list[tuple[float, float, float]] = []   # duration, strength, rise
    total = 0.0
    while total < span:
        heights = _set_heights(rng, rng.randint(3, 8))
        for height in heights:
            duration = rng.uniform(minimum, maximum)
            strength = height * intensity * rng.uniform(0.7, 1.0)
            if rng.random() < 0.10:
                # The odd wave that is simply bigger than the rest.
                strength = min(1.0, strength * rng.uniform(1.3, 1.7))
            plan.append((duration, strength, rng.uniform(0.20, 0.35)))
            total += duration
            if total >= span:
                break
        if total < span:
            gap = rng.uniform(5.0, 20.0) / (0.5 + 0.5 * intensity)
            plan.append((gap, 0.0, 0.5))
            total += gap

    stretch = span / total
    env: list[float] = []
    for duration, strength, rise in plan:
        count = max(1, int(round(duration * stretch * SR)))
        if strength <= 0.0:
            env.extend([background * rng.uniform(0.6, 1.0)] * count)
        else:
            env.extend(_one_wave(strength, count, rise, background))
    target = dsp.samples(seconds)
    if len(env) > target:
        del env[target:]
    while len(env) < target:
        env.append(background)

    # Surf beat: the slow swell of sets arriving and easing off. Quantised, so
    # a beat too slow for the loop flattens rather than clicking; a 60-180 s
    # beat needs a loop long enough to hold it.
    beat = loopdsp.lfo(1.0 / rng.uniform(60.0, 120.0), seconds, 0.4)
    beat2 = loopdsp.lfo(1.0 / rng.uniform(100.0, 180.0), seconds, 0.2)
    return [e * (0.7 + 0.3 * b1) * (0.85 + 0.15 * b2)
            for e, b1, b2 in zip(env, beat, beat2)]


def surf(rng: random.Random, seconds: float, intensity: float) -> list[float]:
    """Ocean surf. `intensity` is sea state: calm lapping to rough surf."""

    if not 0.05 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    wave = _wave_envelope(seconds, rng, intensity)

    # Layer 1: deep rumble, distant swell. Present whatever the surface does.
    rumble = dsp.mul(
        loopdsp.lowpass(loopdsp.lowpass(dsp.noise(seconds, rng), 150.0, 3),
                        80.0, 2),
        loopdsp.lfo(0.04, seconds, 0.3),
    )

    # Layer 2: wave body, cutoff riding the wave height.
    cutoff = [200.0 + 100.0 * intensity
              + (800.0 + 600.0 * intensity) * value for value in wave]
    body = dsp.mul(
        loopdsp.lowpass(
            loopdsp.svf_lowpass(dsp.noise(seconds, rng), cutoff, q=0.7),
            2_000.0, 2,
        ),
        wave,
    )

    # Layer 3: surf hiss, delayed behind the crest -- break first, spray after.
    delayed = loopdsp.rotate(wave, rng.uniform(0.3, 0.8))
    sharp = [value ** 1.5 for value in delayed]
    peak = max(sharp) or 1.0
    hiss = dsp.mul(
        loopdsp.bandpass(dsp.noise(seconds, rng), 500.0, 6_000.0, 3),
        [value / peak for value in sharp],
    )

    # Layer 4: foam fizz, only on the crests that actually broke.
    fizz_env = [max(0.0, min(1.0, value - 0.3)) ** 2 for value in wave]
    fizz = dsp.mul(
        loopdsp.bandpass(dsp.noise(seconds, rng), 2_000.0, 12_000.0, 3),
        fizz_env,
    )

    # Layer 5: background wash -- distant surf and wind on water, so that the
    # gaps between sets are quiet water rather than digital silence.
    wash = dsp.mul(
        loopdsp.bandpass(dsp.noise(seconds, rng), 200.0, 3_000.0, 2),
        loopdsp.lfo(0.015, seconds, 0.2),
    )

    return dsp.mix(
        loopdsp.level(rumble, 0.30 * (0.4 + 0.6 * intensity)),
        loopdsp.level(body, 0.34),
        loopdsp.level(hiss, 0.10 + 0.22 * intensity),
        loopdsp.level(fizz, 0.02 + 0.10 * intensity),
        loopdsp.level(wash, 0.06 + 0.04 * intensity),
    )
