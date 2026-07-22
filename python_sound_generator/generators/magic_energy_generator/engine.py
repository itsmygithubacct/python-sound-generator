"""Fantasy energy gestures built from noise, resonances, and sparse grains.

The existing combat ``energy_release`` was auditioned as a good phaser or
laser. These gestures deliberately avoid its low boom and single smooth sweep.
Fantasy identity comes from a struck-steel attack, inharmonic bell modes,
irregular sparkle, and air: acoustic metaphors for an enchanted blade rather
than the vocabulary of a gun.
"""

from __future__ import annotations

from ..._audio.dsp import write_wav as write_wav

import math
import random

from ..._audio import dsp

SR = dsp.SR


def _envelope(duration: float, attack: float, decay: float) -> list[float]:
    return [
        min(1.0, t / max(attack, 1e-6)) * math.exp(-t / decay)
        for t in dsp.timebase(duration)
    ]


def _impulses(
    duration: float,
    rng: random.Random,
    *,
    count: int,
    start: float,
    end: float,
) -> list[float]:
    signal = dsp.silence(duration)
    first = max(0, round(start * SR))
    last = min(len(signal), max(first + 1, round(end * SR)))
    for _ in range(count):
        index = rng.randrange(first, last)
        signal[index] += rng.uniform(0.45, 1.0) * rng.choice((-1.0, 1.0))
    return signal


def beam(rng: random.Random, duration: float = 0.48,
         intensity: float = 0.9) -> list[float]:
    """Enchanted sword projectile: steel snap, crescent, and airy streak."""

    duration = float(duration)
    t = dsp.timebase(duration)

    # A physical point of origin: steel modes struck by a very short noise
    # impulse. Inharmonic ratios keep it metallic rather than musical.
    strike_noise = dsp.mul(
        dsp.highpass(dsp.noise(duration, rng), 900.0, 4),
        [math.exp(-x / 0.010) for x in t],
    )
    steel = dsp.silence(duration)
    base = rng.uniform(1180.0, 1510.0)
    for ratio, gain, decay in (
        (1.0, 0.65, 0.075),
        (1.73, 0.42, 0.060),
        (2.91, 0.24, 0.045),
    ):
        mode = dsp.resonator(strike_noise, base * ratio, q=18.0, gain=gain)
        mode = dsp.mul(mode, [math.exp(-x / decay) for x in t])
        steel = dsp.mix(steel, mode)

    # The crescent is a small inharmonic cluster that blooms after the snap.
    # It bends only slightly; a single strong sweep was exactly what made the
    # older energy cue read as a laser.
    crescent_env = [
        min(1.0, x / 0.018) * math.exp(-max(0.0, x - 0.018) / 0.17)
        for x in t
    ]
    crescent = dsp.silence(duration)
    for start, end, gain in (
        (760.0, 980.0, 0.34),
        (1320.0, 1160.0, 0.24),
        (2070.0, 1740.0, 0.16),
    ):
        wobble = rng.uniform(8.0, 17.0)
        freqs = [
            start + (end - start) * min(1.0, x / duration)
            + 18.0 * math.sin(dsp.TAU * wobble * x + rng.random() * dsp.TAU)
            for x in t
        ]
        voice = dsp.sine_track(freqs, phase=rng.random() * dsp.TAU)
        crescent = dsp.mix(crescent, dsp.scale(dsp.mul(voice, crescent_env), gain))

    # A short, broad, receding air trail. Its centre falls, but the random
    # grains prevent the clean electronic zzzap contour of a phaser.
    air_noise = dsp.noise(duration, rng)
    centres = [5100.0 * math.exp(-x / 0.31) + 1050.0 for x in t]
    air = dsp.svf_bandpass(air_noise, centres, q=1.15)
    air_env = [
        (1.0 - math.exp(-x / 0.012)) * math.exp(-x / 0.19)
        for x in t
    ]
    air = dsp.mul(air, air_env)

    sparkle_impulses = _impulses(
        duration, rng, count=11, start=0.025, end=min(duration, 0.34)
    )
    sparkle = dsp.mix(
        dsp.resonator(sparkle_impulses, 4100.0, q=22.0, gain=0.8),
        dsp.resonator(sparkle_impulses, 6850.0, q=26.0, gain=0.5),
    )
    sparkle = dsp.mul(sparkle, [math.exp(-x / 0.22) for x in t])

    return dsp.scale(
        dsp.mix(
            dsp.scale(steel, 0.82),
            crescent,
            dsp.scale(air, 0.62),
            dsp.scale(sparkle, 0.30),
        ),
        intensity,
    )


def dissolve(rng: random.Random, duration: float = 1.08,
             intensity: float = 0.8) -> list[float]:
    """Life energy unbinding into descending motes and a fading breath."""

    duration = float(duration)
    t = dsp.timebase(duration)
    frames = len(t)

    # An airy body thins as it falls. It starts after a short gap so this layer
    # can sit under a human gasp without masking its onset.
    noise = dsp.noise(duration, rng)
    centres = [3600.0 * math.exp(-max(0.0, x - 0.10) / 0.58) + 260.0 for x in t]
    air = dsp.svf_bandpass(noise, centres, q=1.0)
    air_env = [
        0.0 if x < 0.08 else (1.0 - math.exp(-(x - 0.08) / 0.06))
        * math.exp(-(x - 0.08) / 0.52)
        for x in t
    ]
    air = dsp.mul(air, air_env)

    # Motes are separate decaying resonances whose event rate slows over time.
    # Their downward distribution says dissolve; a continuous oscillator glide
    # would say power-down or laser.
    motes = dsp.silence(duration)
    for index in range(16):
        progress = index / 15.0
        when = 0.10 + (progress ** 0.72) * max(0.05, duration - 0.22)
        when += rng.uniform(-0.012, 0.012)
        frequency = 4700.0 * math.exp(-progress * 2.0) + rng.uniform(120.0, 420.0)
        impulse = dsp.silence(duration)
        frame = min(frames - 1, max(0, round(when * SR)))
        impulse[frame] = rng.uniform(0.45, 1.0)
        mote = dsp.resonator(impulse, frequency, q=24.0, gain=1.0)
        motes = dsp.mix(motes, mote)

    # A quiet falling three-part aura supplies continuity between the motes.
    aura = dsp.silence(duration)
    aura_env = [
        (0.0 if x < 0.06 else 1.0 - math.exp(-(x - 0.06) / 0.08))
        * math.exp(-max(0.0, x - 0.06) / 0.56)
        for x in t
    ]
    for ratio, gain in ((1.0, 0.24), (1.41, 0.15), (2.27, 0.09)):
        freqs = [
            ratio * (920.0 * math.exp(-x / 0.72) + 125.0)
            for x in t
        ]
        aura = dsp.mix(
            aura,
            dsp.scale(
                dsp.mul(dsp.sine_track(freqs, rng.random() * dsp.TAU), aura_env),
                gain,
            ),
        )

    return dsp.scale(
        dsp.mix(dsp.scale(air, 0.52), dsp.scale(motes, 0.85), aura), intensity
    )


def master(signal: list[float], peak_db: float) -> list[float]:
    return dsp.master(signal, peak_db, drive=1.12, fade_in_ms=0.5, fade_out_ms=35.0)
