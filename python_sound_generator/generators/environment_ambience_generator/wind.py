"""Wind, ported from lib/wind.py.

Ported from an earlier NumPy/SciPy wind prototype: stereo, 48 kHz, streamed with
fades. This is stdlib, mono, 44.1 kHz, and loops. Every cutoff in the original
is an absolute frequency and none of its Nyquist guards ever bind, so the
sample-rate move is spectrally a no-op.

Wind itself is silent. Everything you hear is air *interacting with something*,
which is why each layer here is a different mechanism with its own noise and
its own modulation rather than an EQ band of one source:

- **rumble** and **body** are brown noise, because real wind noise falls at
  -5 to -6 dB/octave and brown is -6 (pink is only -3). Kolmogorov puts the
  velocity spectrum at k^-5/3, and the pressure spectrum -- what you actually
  hear -- steeper still at -7/3.
- **hiss**, **foliage** and the **aeolian** tone are white, because there the
  filter is the mechanism and the source spectrum should not also colour it.

Two ideas carry most of the realism.

*Louder must mean brighter.* Body is one noise realisation sent through a dark
and a bright lowpass and crossfaded by the gust envelope, so cutoff and level
rise together. Decoupling the two is the classic tell of fake wind. The
crossfade also sidesteps time-varying coefficients entirely, so there are no
block boundaries to print.

*The foliage bands are fixed.* 300-3000 Hz for broadleaf flap, 1500-6000 Hz for
conifer needle hiss, regardless of wind speed. The research doc is emphatic:
the spectral colour of a tree species does not change with wind speed, only its
volume -- an oak always sounds like an oak. Make those bands track intensity
and the wood appears to change species as the wind gets up.

Modulators are filtered noise, never sines: a sine is periodic and the ear
finds it within seconds, which is the difference between wind and a wobble
effect. Van der Hoven's spectral gap is why they sit at separated timescales --
swell, gusts, turbulence -- with deliberately little in between.
"""

from __future__ import annotations

import math
import random

from ..._audio import dsp

from ..._audio import loops as loopdsp


SR = dsp.SR


def brown(seconds: float, rng: random.Random) -> list[float]:
    """Brown noise that loops.

    Brown noise is the running sum of white noise, and a running sum is a
    random walk -- it ends wherever it wandered to, which is not where it
    started, so the naive version cannot loop at any length.

    The fix is exact rather than cosmetic. Force the white noise to sum to zero
    first; then the walk necessarily returns to zero at the end, and the sample
    that would follow the last one is `walk[-1] + white[0]` = `white[0]` =
    `walk[0]`. The walk's periodic extension is then continuous by
    construction, and the DC-trimming highpass on top of it is primed
    circularly like everything else here.
    """

    white = dsp.noise(seconds, rng)
    mean = sum(white) / len(white)
    walk = []
    total = 0.0
    for value in white:
        total += value - mean
        walk.append(total)
    out = loopdsp.highpass(walk, 10.0, 2)
    peak = max((abs(value) for value in out), default=0.0)
    return dsp.scale(out, 1.0 / peak) if peak > 0.0 else out


def smooth_random(seconds: float, rate: float,
                  rng: random.Random) -> list[float]:
    """A modulator: noise lowpassed to `rate`, mapped to 0..1.

    Filtered noise rather than a sine, and the research doc is emphatic about
    it: a sine is periodic and the ear locks onto it within seconds, which is
    the difference between wind and a wobble effect.

    It goes through :func:`loopdsp.smooth` rather than ``loopdsp.lowpass``
    because the latter clamps its cutoff at 20 Hz -- correct for audio, three
    decades wrong for a modulator, and audible as chop.

    Min-max normalisation is what makes the very low rates work: it throws the
    gain away and keeps the shape, so a swell too slow to fit the loop stretches
    to fill it rather than vanishing.
    """

    curve = loopdsp.smooth(dsp.noise(seconds, rng), rate, poles=2)
    low, high = min(curve), max(curve)
    if high - low < 1e-12:
        return [0.5] * len(curve)
    span = high - low
    return [(value - low) / span for value in curve]


def _asymmetric(gusts: list[float], seconds: float) -> list[float]:
    """One-pole follower: gusts rise faster than they fall.

    Runs at a 10 ms block rate rather than per sample -- the original's trick,
    and a good one: the envelope carries nothing above 100 Hz, so tracking it
    at 44.1 kHz would be 441x the work for no audible difference.

    The follower is an IIR, so it carries state and has to be primed like any
    other filter or the loop opens mid-decay. It is primed by running it over
    the wrapped tail first.
    """

    block = dsp.samples(0.01)
    count = (len(gusts) + block - 1) // block
    coarse = []
    for i in range(count):
        chunk = gusts[i * block:min((i + 1) * block, len(gusts))]
        coarse.append(sum(chunk) / len(chunk))

    attack = 1.0 - math.exp(-block / (SR * 0.15))
    release = 1.0 - math.exp(-block / (SR * 0.6))

    # Prime over one wrapped pass, then keep the second.
    state = coarse[0]
    for _ in range(2):
        followed = []
        for value in coarse:
            coeff = attack if value > state else release
            state += coeff * (value - state)
            followed.append(state)

    out = [0.0] * len(gusts)
    for i in range(len(gusts)):
        position = i / block
        index = int(position)
        frac = position - index
        first = followed[min(index, count - 1)]
        second = followed[min(index + 1, count - 1)]
        out[i] = first * (1.0 - frac) + second * frac
    low, high = min(out), max(out)
    if high - low < 1e-12:
        return [0.5] * len(out)
    return [(value - low) / (high - low) for value in out]


def gust_envelope(seconds: float, rng: random.Random,
                  intensity: float) -> list[float]:
    """Wind speed over time, on the three timescales weather actually has.

    Swell (20-66 s), gusts (3-14 s) and turbulence (0.3-1.25 s), weighted
    0.45/0.40/0.15. The rates are drawn from continuous uniforms so no two are
    in an integer ratio and the pattern never realigns.

    Note this never reaches 1.0 even at full intensity -- the three components
    do not peak together. That is not a bug to normalise away: it is why the
    default reads as moderate wind rather than a gale.
    """

    base = 0.15 + 0.15 * intensity
    swell = smooth_random(seconds, rng.uniform(0.015, 0.05), rng)
    gusts = smooth_random(seconds, rng.uniform(0.07, 0.3), rng)
    asym = _asymmetric(gusts, seconds)
    turbulence = smooth_random(seconds, rng.uniform(0.8, 3.0), rng)
    return [
        base + (1.0 - base) * (0.45 * s + 0.40 * a + 0.15 * t) * intensity
        for s, a, t in zip(swell, asym, turbulence)
    ]


def _gate(env: list[float], threshold: float, knee: float) -> list[float]:
    """Only speak above `threshold`, then renormalise to peak 1."""

    gated = [max(0.0, min(1.0, value - threshold)) ** knee for value in env]
    peak = max(gated, default=0.0)
    if peak <= 0.0:
        return gated
    return [value / peak for value in gated]


def foliage_bed(seconds: float, rng: random.Random, env: list[float],
                amount: float) -> list[float]:
    """Leaves. Two mechanisms, deliberately not one.

    Broad flat leaves flag-flap and rustle low; conifer needles and thin twigs
    shed vortices and hiss high. Separate noise, separate modulation rates:
    they must not move together, because in a real wood they do not.
    """

    if amount <= 0.05:
        return dsp.silence(seconds)
    broadleaf = dsp.mul(
        dsp.mul(loopdsp.bandpass(dsp.noise(seconds, rng), 300.0, 3_000.0, 2),
                env),
        smooth_random(seconds, rng.uniform(0.2, 0.8), rng),
    )
    conifer = dsp.mul(
        dsp.mul(loopdsp.bandpass(dsp.noise(seconds, rng), 1_500.0, 6_000.0, 3),
                env),
        smooth_random(seconds, rng.uniform(0.1, 0.4), rng),
    )
    return dsp.scale(
        dsp.mix(dsp.scale(broadleaf, 0.6 * 0.6), dsp.scale(conifer, 0.35 * 0.4)),
        amount,
    )


def _aeolian(seconds: float, rng: random.Random, env: list[float],
             intensity: float) -> list[float]:
    """A wire singing. Strouhal: f = 0.2 * V / d.

    Gated to strong wind because the real thing is: acoustic power near a
    surface goes as roughly the sixth power of wind speed, so a tone that is
    inaudible in a breeze arrives suddenly. Narrow-band noise rather than a
    sine, at +-8%, so it stays airy instead of turning into a test tone.
    """

    if intensity < 0.55:
        return dsp.silence(seconds)
    diameter = rng.uniform(0.002, 0.005)
    speed = 5.0 + 12.0 * intensity
    freq = 0.2 * speed / diameter
    width = max(20.0, freq * 0.08)
    tone = loopdsp.bandpass(dsp.noise(seconds, rng),
                            max(50.0, freq - width), freq + width, 4)
    strength = ((intensity - 0.55) / 0.45) ** 2
    return dsp.scale(dsp.mul(tone, _gate(env, 0.4, 2.0)), strength)


def blow(rng: random.Random, seconds: float, intensity: float,
         foliage: float = 0.3) -> list[float]:
    """Wind. `intensity` is Beaufort: 0.1 breeze, 0.5 fresh, 1.0 near gale."""

    if not 0.05 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    if not 0.0 <= foliage <= 1.0:
        raise ValueError("foliage must be between 0 and 1")
    env = gust_envelope(seconds, rng, intensity)

    rumble = dsp.mul(
        loopdsp.lowpass(brown(seconds, rng), 80.0 + 120.0 * intensity, 2), env
    )

    body_noise = brown(seconds, rng)
    dark_cut = 150.0 + 150.0 * intensity
    bright_cut = dark_cut + 500.0 + 900.0 * intensity
    dark = loopdsp.lowpass(body_noise, dark_cut, 3)
    bright = loopdsp.lowpass(body_noise, bright_cut, 3)
    body = dsp.mul(
        [d + (b - d) * e for d, b, e in zip(dark, bright, env)], env
    )

    hiss = dsp.mul(
        dsp.mul(
            loopdsp.bandpass(dsp.noise(seconds, rng), 500.0 + 300.0 * intensity,
                             3_000.0 + 5_000.0 * intensity, 3),
            _gate(env, 0.3, 1.5),
        ),
        smooth_random(seconds, rng.uniform(0.15, 0.4), rng),
    )

    leaves = foliage_bed(seconds, rng, env, foliage)
    wire = _aeolian(seconds, rng, env, intensity)

    parts = [
        loopdsp.level(rumble, 0.30),
        loopdsp.level(body, 0.34),
    ]
    # Gated layers keep plain gains: their level is the statement. Normalising
    # a hiss that only speaks in the gusts would make a breeze hiss constantly.
    if max((abs(v) for v in hiss), default=0.0) > 1e-9:
        parts.append(dsp.scale(hiss, 0.20 * intensity * 3.0))
    if max((abs(v) for v in leaves), default=0.0) > 1e-9:
        parts.append(dsp.scale(leaves, 0.25 * 3.0))
    if max((abs(v) for v in wire), default=0.0) > 1e-9:
        parts.append(dsp.scale(wire, 0.08 * 3.0))
    return dsp.mix(*parts)
