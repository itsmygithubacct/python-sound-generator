"""Procedural footstep synthesis with NumPy and SciPy.

The model separates a step into a shoe exciter, a floor impulse response,
plantar pressure/friction, and (for loose surfaces) stochastic microscopic
collisions.  No recorded footstep samples are used.
"""

from __future__ import annotations

from ..._audio.pcm import AudioBuffer, write_pcm16

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt


DEFAULT_SAMPLE_RATE = 44_100


@dataclass(frozen=True)
class ParticleBand:
    """One size class in a stochastic aggregate-material model."""

    low: float
    high: float
    events: int
    decay: float
    gain: float
    tonal_mix: float = 0.0


@dataclass(frozen=True)
class SurfaceProfile:
    """Perceptually important properties of a walked-on material."""

    label: str
    kind: str
    duration: float
    impact_decay: float
    spectrum: tuple[tuple[float, float], ...]
    impact_gain: float
    body_gain: float
    contact_gain: float
    modes: tuple[tuple[float, float, float], ...] = ()
    particles: tuple[ParticleBand, ...] = ()
    scuff_gain: float = 0.025


@dataclass(frozen=True)
class FootwearProfile:
    """Parameters for the shoe-side exciter."""

    label: str
    hardness: float
    mass: float
    brightness: float
    heel_gain: float
    roll_gain: float
    toe_gain: float
    toe_delay: float




SURFACES: dict[str, SurfaceProfile] = {
    "concrete": SurfaceProfile(
        label="Concrete",
        kind="solid",
        duration=0.29,
        impact_decay=0.021,
        spectrum=((25, -28), (60, -15), (120, -6), (240, -1),
                  (520, 0), (1_100, -2), (2_500, -8),
                  (5_500, -18), (10_000, -30), (16_000, -40)),
        impact_gain=0.76,
        body_gain=0.085,
        contact_gain=0.045,
        modes=((104, 0.030, 0.055), (173, 0.024, 0.050),
               (287, 0.019, 0.042), (463, 0.014, 0.032),
               (790, 0.010, 0.022)),
        scuff_gain=0.024,
    ),
    "wood": SurfaceProfile(
        label="Wood",
        kind="solid",
        duration=0.40,
        impact_decay=0.046,
        spectrum=((25, -15), (55, -5), (88, 0), (145, 2),
                  (240, -3), (480, -10), (1_100, -17),
                  (3_000, -24), (12_000, -34)),
        impact_gain=0.64,
        body_gain=0.24,
        contact_gain=0.055,
        modes=((82, 0.095, 0.14), (111, 0.078, 0.13),
               (147, 0.063, 0.11), (193, 0.050, 0.075),
               (258, 0.037, 0.050), (405, 0.024, 0.030)),
        scuff_gain=0.020,
    ),
    "gravel": SurfaceProfile(
        label="Gravel",
        kind="aggregate",
        duration=0.54,
        impact_decay=0.019,
        spectrum=((25, -42), (70, -30), (150, -18), (300, -5),
                  (650, 0), (1_400, -1), (3_200, -4),
                  (7_000, -8), (16_000, -18)),
        impact_gain=0.23,
        body_gain=0.025,
        contact_gain=0.27,
        modes=((150, 0.020, 0.025), (310, 0.014, 0.018)),
        particles=(
            ParticleBand(180, 700, 48, 0.018, 0.25, 0.20),
            ParticleBand(550, 2_300, 85, 0.010, 0.23, 0.12),
            ParticleBand(1_800, 8_500, 120, 0.005, 0.18, 0.05),
        ),
        scuff_gain=0.050,
    ),
    "grass": SurfaceProfile(
        label="Grass",
        kind="aggregate",
        duration=0.49,
        impact_decay=0.026,
        spectrum=((25, -25), (70, -11), (140, -3), (320, 0),
                  (850, -3), (2_300, -9), (6_000, -19),
                  (14_000, -32)),
        impact_gain=0.18,
        body_gain=0.085,
        contact_gain=0.30,
        particles=(
            ParticleBand(220, 1_100, 42, 0.012, 0.16, 0.05),
            ParticleBand(1_200, 5_800, 24, 0.005, 0.10, 0.03),
        ),
        scuff_gain=0.045,
    ),
    "snow": SurfaceProfile(
        label="Snow",
        kind="aggregate",
        duration=0.57,
        impact_decay=0.024,
        spectrum=((25, -40), (90, -27), (220, -14), (500, -5),
                  (1_100, 0), (2_500, -1), (5_500, -5),
                  (11_000, -13), (18_000, -24)),
        impact_gain=0.16,
        body_gain=0.045,
        contact_gain=0.31,
        particles=(
            ParticleBand(450, 1_800, 85, 0.007, 0.18, 0.03),
            ParticleBand(1_400, 5_500, 145, 0.004, 0.20, 0.01),
            ParticleBand(4_000, 11_000, 95, 0.002, 0.13, 0.0),
        ),
        scuff_gain=0.035,
    ),
    "metal": SurfaceProfile(
        label="Metal",
        kind="solid",
        duration=0.68,
        impact_decay=0.035,
        spectrum=((25, -23), (70, -9), (140, -2), (320, -3),
                  (800, -5), (2_000, -8), (6_000, -12),
                  (15_000, -24)),
        impact_gain=0.60,
        body_gain=0.12,
        contact_gain=0.055,
        modes=((137, 0.210, 0.11), (223, 0.175, 0.095),
               (367, 0.140, 0.075), (593, 0.105, 0.052),
               (967, 0.075, 0.035), (1_563, 0.050, 0.022)),
        scuff_gain=0.024,
    ),
    "dirt": SurfaceProfile(
        label="Dirt",
        kind="aggregate",
        duration=0.50,
        impact_decay=0.025,
        spectrum=((25, -25), (60, -12), (120, -4), (280, 0),
                  (700, -2), (1_800, -7), (5_000, -15),
                  (14_000, -29)),
        impact_gain=0.23,
        body_gain=0.080,
        contact_gain=0.29,
        modes=((93, 0.024, 0.030),),
        particles=(
            ParticleBand(160, 850, 55, 0.011, 0.18, 0.06),
            ParticleBand(700, 3_500, 65, 0.006, 0.15, 0.025),
        ),
        scuff_gain=0.046,
    ),
    "carpet": SurfaceProfile(
        label="Carpet",
        kind="soft",
        duration=0.31,
        impact_decay=0.030,
        spectrum=((25, -10), (55, -2), (95, 0), (180, -4),
                  (380, -11), (850, -21), (2_000, -34),
                  (8_000, -48)),
        impact_gain=0.40,
        body_gain=0.26,
        contact_gain=0.035,
        modes=((71, 0.030, 0.060), (119, 0.020, 0.035)),
        scuff_gain=0.008,
    ),
}


FOOTWEAR: dict[str, FootwearProfile] = {
    "sneaker": FootwearProfile(
        "Sneaker", 0.24, 0.90, 0.72, 0.90, 0.72, 0.26, 0.092
    ),
    "boot": FootwearProfile(
        "Boot", 0.62, 1.18, 0.76, 1.10, 0.62, 0.30, 0.104
    ),
    "barefoot": FootwearProfile(
        "Barefoot", 0.08, 0.62, 0.58, 0.64, 0.85, 0.52, 0.072
    ),
    "hard_shoe": FootwearProfile(
        "Hard shoe", 0.96, 0.94, 1.16, 0.95, 0.34, 0.58, 0.086
    ),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _validate_settings(
    surface: str,
    footwear: str,
    steps: int,
    tempo: float,
    intensity: float,
    variation: float,
    stereo_width: float,
    sample_rate: int,
) -> None:
    if surface not in SURFACES:
        raise ValueError(
            f"unknown surface {surface!r}; choose from {', '.join(SURFACES)}"
        )
    if footwear not in FOOTWEAR:
        raise ValueError(
            f"unknown footwear {footwear!r}; choose from {', '.join(FOOTWEAR)}"
        )
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not 20.0 <= tempo <= 400.0:
        raise ValueError("tempo must be between 20 and 400 steps per minute")
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0 and 1")
    if not 0.0 <= variation <= 1.0:
        raise ValueError("variation must be between 0 and 1")
    if not 0.0 <= stereo_width <= 1.0:
        raise ValueError("stereo_width must be between 0 and 1")
    if not 8_000 <= sample_rate <= 192_000:
        raise ValueError("sample_rate must be between 8000 and 192000 Hz")


def _unit_peak(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    return signal / peak if peak > 1e-12 else signal


def _colored_noise(
    count: int,
    sample_rate: int,
    anchors: Sequence[tuple[float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate noise with a smooth spectrum interpolated in log-frequency."""

    count = max(2, int(count))
    source = rng.standard_normal(count)
    spectrum = np.fft.rfft(source)
    frequencies = np.fft.rfftfreq(count, 1.0 / sample_rate)
    anchor_frequencies = np.asarray([point[0] for point in anchors], dtype=float)
    anchor_db = np.asarray([point[1] for point in anchors], dtype=float)
    safe_frequencies = np.maximum(frequencies, max(1.0, anchor_frequencies[0]))
    response_db = np.interp(
        np.log(safe_frequencies), np.log(anchor_frequencies), anchor_db
    )
    response = np.power(10.0, response_db / 20.0)
    response[0] = 0.0
    shaped = np.fft.irfft(spectrum * response, n=count)
    shaped -= np.mean(shaped)
    rms = float(np.sqrt(np.mean(shaped * shaped)))
    return shaped / rms if rms > 1e-12 else shaped


def _band_noise(
    count: int,
    low: float,
    high: float,
    sample_rate: int,
    rng: np.random.Generator,
    order: int = 3,
) -> np.ndarray:
    nyquist = sample_rate * 0.5
    low = _clamp(low, 10.0, nyquist * 0.88)
    high = _clamp(high, low + 10.0, nyquist * 0.95)
    source = rng.standard_normal(max(2, count))
    sos = butter(order, (low, high), btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfilt(sos, source)
    rms = float(np.sqrt(np.mean(filtered * filtered)))
    return filtered / rms if rms > 1e-12 else filtered


def _pulse(
    time: np.ndarray, onset: float, attack: float, decay: float
) -> np.ndarray:
    local = time - onset
    result = np.zeros_like(time)
    rising = (local > 0.0) & (local < attack)
    result[rising] = np.sin(local[rising] / attack * math.pi * 0.5) ** 2
    falling = local >= attack
    result[falling] = np.exp(-(local[falling] - attack) / decay)
    return result


def _contact_curves(
    frame_count: int,
    sample_rate: int,
    footwear: FootwearProfile,
    variation: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    time = np.arange(frame_count, dtype=float) / sample_rate
    hardness = footwear.hardness
    heel_attack = (0.0065 - 0.0053 * hardness) * rng.uniform(0.92, 1.08)
    heel_decay = (0.030 - 0.013 * hardness) * rng.uniform(0.90, 1.10)
    toe_attack = (0.0055 - 0.0043 * hardness) * rng.uniform(0.90, 1.10)
    toe_decay = (0.026 - 0.010 * hardness) * rng.uniform(0.90, 1.10)
    toe_onset = footwear.toe_delay * (1.0 + rng.uniform(-0.10, 0.10) * variation)

    heel = _pulse(time, 0.0, heel_attack, heel_decay)
    roll = _pulse(time, 0.016, 0.040, 0.070)
    toe = _pulse(time, toe_onset, toe_attack, toe_decay)
    pressure = np.clip(
        heel * 0.78 + roll * footwear.roll_gain * 0.42
        + toe * footwear.toe_gain * 0.50,
        0.0,
        1.0,
    )
    return heel, roll, toe, pressure, toe_onset


def _floor_ir(
    surface: SurfaceProfile,
    footwear: FootwearProfile,
    sample_rate: int,
    variation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    longest_mode = max((decay for _, decay, _ in surface.modes), default=0.0)
    duration = min(0.58, max(0.090, surface.impact_decay * 7.0, longest_mode * 4.3))
    count = max(32, round(duration * sample_rate))
    time = np.arange(count, dtype=float) / sample_rate

    broadband = _colored_noise(count, sample_rate, surface.spectrum, rng)
    broadband *= np.exp(-time / surface.impact_decay)
    impulse_response = broadband * 0.88

    for frequency, decay, gain in surface.modes:
        detuned = frequency * (1.0 + rng.uniform(-0.022, 0.022) * variation)
        detuned *= 0.97 + 0.06 * footwear.mass
        phase = rng.uniform(-0.12, 0.12)
        mode = np.sin(2.0 * math.pi * detuned * time + phase)
        mode *= np.exp(-time / decay) * gain
        impulse_response += mode

    fade = min(len(impulse_response), max(4, round(0.012 * sample_rate)))
    impulse_response[-fade:] *= np.linspace(1.0, 0.0, fade)
    energy = float(np.sqrt(np.sum(impulse_response * impulse_response)))
    return impulse_response / energy if energy > 1e-12 else impulse_response


def _aggregate_activity(
    pressure: np.ndarray,
    toe_onset: float,
    surface: SurfaceProfile,
    sample_rate: int,
) -> np.ndarray:
    time = np.arange(len(pressure), dtype=float) / sample_rate
    if surface.kind != "aggregate":
        return pressure
    yielding = _pulse(
        time,
        toe_onset * 0.58,
        0.042,
        max(0.055, surface.duration * 0.14),
    )
    return np.maximum(pressure * 0.72, yielding * 0.92)


def _particle_ir(
    band: ParticleBand,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = max(16, round(band.decay * 6.0 * sample_rate))
    time = np.arange(count, dtype=float) / sample_rate
    center = math.sqrt(band.low * band.high)
    anchors = (
        (max(15.0, band.low * 0.45), -32.0),
        (band.low, -3.0),
        (center, 1.0),
        (band.high, -3.0),
        (band.high * 1.8, -34.0),
    )
    noise = _colored_noise(count, sample_rate, anchors, rng)
    response = noise * np.exp(-time / band.decay)
    if band.tonal_mix > 0.0:
        frequency = center * rng.uniform(0.78, 1.24)
        ring = np.sin(2.0 * math.pi * frequency * time + rng.uniform(0, math.pi))
        ring *= np.exp(-time / (band.decay * 1.15))
        response = response * (1.0 - band.tonal_mix) + ring * band.tonal_mix
    energy = float(np.sqrt(np.sum(response * response)))
    return response / energy if energy > 1e-12 else response


def _particle_field(
    surface: SurfaceProfile,
    activity: np.ndarray,
    intensity: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.zeros_like(activity)
    if not surface.particles or not np.any(activity > 0.0):
        return result
    weights = np.power(np.maximum(activity, 0.0), 1.35) + 0.0005
    weights /= np.sum(weights)

    for band in surface.particles:
        event_count = max(1, round(band.events * (0.60 + 0.58 * intensity)))
        indices = rng.choice(len(activity), size=event_count, replace=True, p=weights)
        amplitudes = np.clip(rng.lognormal(-0.35, 0.62, event_count), 0.12, 2.6)
        amplitudes *= rng.choice(np.asarray((-1.0, 1.0)), size=event_count)
        amplitudes *= 0.50 + 0.65 * np.sqrt(activity[indices])
        impulse_train = np.zeros_like(activity)
        np.add.at(impulse_train, indices, amplitudes)
        collision = fftconvolve(
            impulse_train, _particle_ir(band, sample_rate, rng), mode="full"
        )[:len(activity)]
        collision = _unit_peak(collision)
        result += collision * band.gain
    return result


def _render_step(
    surface: SurfaceProfile,
    footwear: FootwearProfile,
    intensity: float,
    variation: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    frame_count = max(1, round(surface.duration * sample_rate))
    if intensity <= 0.0:
        return np.zeros(frame_count, dtype=float)
    time = np.arange(frame_count, dtype=float) / sample_rate
    heel, roll, toe, pressure, toe_onset = _contact_curves(
        frame_count, sample_rate, footwear, variation, rng
    )
    activity = _aggregate_activity(pressure, toe_onset, surface, sample_rate)
    force = intensity ** 0.72 * rng.uniform(1.0 - 0.11 * variation, 1.0 + 0.11 * variation)

    # Shoe excitation is a brief stochastic shock; its spectrum depends on sole hardness.
    exciter_high = min(sample_rate * 0.46, 2_400.0 + 10_500.0 * footwear.brightness)
    exciter = _band_noise(
        frame_count, 55.0, exciter_high, sample_rate, rng, order=2
    )
    impact_envelope = (
        heel ** (2.5 + footwear.hardness * 0.7) * footwear.heel_gain
        + toe ** (2.4 + footwear.hardness * 0.6) * footwear.toe_gain
    )
    exciter *= impact_envelope
    floor_response = fftconvolve(
        exciter, _floor_ir(surface, footwear, sample_rate, variation, rng), mode="full"
    )[:frame_count]
    result = _unit_peak(floor_response) * surface.impact_gain

    # A quiet, non-tonal low body conveys weight without turning into a kick drum.
    body_high = min(sample_rate * 0.43, 170.0 + 80.0 * footwear.mass)
    body = _band_noise(frame_count, 34.0, body_high, sample_rate, rng, order=2)
    body *= heel * 0.82 + roll * 0.22 + toe * footwear.toe_gain * 0.24
    result += _unit_peak(body) * surface.body_gain * footwear.mass

    # Continuous force/friction carries material identity between microscopic events.
    texture = _colored_noise(frame_count, sample_rate, surface.spectrum, rng)
    modulation_source = np.abs(rng.standard_normal(frame_count))
    modulation_sos = butter(2, 28.0, btype="lowpass", fs=sample_rate, output="sos")
    modulation = sosfilt(modulation_sos, modulation_source)
    modulation -= np.min(modulation)
    modulation /= max(1e-12, float(np.max(modulation)))
    modulation = 0.48 + modulation * 0.52
    texture *= np.power(activity, 0.88 if surface.kind == "aggregate" else 1.40)
    texture *= modulation
    result += _unit_peak(texture) * surface.contact_gain * footwear.roll_gain

    particles = _particle_field(surface, activity, intensity, sample_rate, rng)
    result += particles

    # Direct sole detail is short and strongest for hard shoes.
    sole_detail = _band_noise(
        frame_count,
        500.0,
        min(sample_rate * 0.46, 5_000.0 + 7_000.0 * footwear.brightness),
        sample_rate,
        rng,
        order=2,
    )
    sole_envelope = heel ** 3.5 + toe ** 3.2 * footwear.toe_gain
    sole_gain = 0.030 + 0.090 * footwear.hardness
    if surface.kind == "aggregate":
        sole_gain *= 0.55
    if surface.kind == "soft":
        sole_gain *= 0.25
    result += _unit_peak(sole_detail * sole_envelope) * sole_gain

    # Not every step brushes; intermittent toe release avoids a mechanical sequence.
    scuff_probability = 0.62 if surface.kind == "aggregate" else 0.30
    if rng.random() < scuff_probability and surface.scuff_gain > 0.0:
        start = toe_onset + rng.uniform(0.018, 0.040)
        length = rng.uniform(0.030, 0.072)
        scuff_env = _pulse(time, start, 0.010, length * 0.42)
        scuff = _colored_noise(frame_count, sample_rate, surface.spectrum, rng)
        result += _unit_peak(scuff * scuff_env) * surface.scuff_gain

    result *= force
    fade = min(frame_count, max(4, round(0.018 * sample_rate)))
    result[-fade:] *= np.linspace(1.0, 0.0, fade)
    return np.tanh(result * 1.08) / math.tanh(1.08)


def _master(channels: Sequence[np.ndarray], sample_rate: int) -> tuple[np.ndarray, ...]:
    if not channels:
        return ()
    processed: list[np.ndarray] = []
    highpass = butter(2, 27.0, btype="highpass", fs=sample_rate, output="sos")
    lowpass = butter(
        2, min(16_000.0, sample_rate * 0.43), btype="lowpass",
        fs=sample_rate, output="sos"
    )
    for channel in channels:
        filtered = sosfilt(highpass, channel)
        filtered = sosfilt(lowpass, filtered)
        dry = filtered.copy()
        # Sparse early reflections place the dry synthesis in a small real space.
        for delay_seconds, gain in ((0.011, 0.026), (0.023, -0.018), (0.037, 0.012)):
            delay = round(delay_seconds * sample_rate)
            if delay < len(filtered):
                filtered[delay:] += dry[:-delay] * gain
        processed.append(filtered)

    frame_count = len(processed[0])
    fade_in = min(frame_count, max(2, round(0.004 * sample_rate)))
    fade_out = min(frame_count, max(2, round(0.025 * sample_rate)))
    for channel in processed:
        channel[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
        channel[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)

    peak = max(float(np.max(np.abs(channel))) for channel in processed)
    scale = 1.18
    if peak * scale > 0.95:
        scale = 0.95 / peak
    return tuple(np.clip(channel * scale, -0.95, 0.95) for channel in processed)


def generate_footsteps(
    surface: str = "concrete",
    footwear: str = "sneaker",
    steps: int = 8,
    tempo: float = 112.0,
    intensity: float = 0.78,
    variation: float = 0.65,
    stereo_width: float = 0.38,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
    mono: bool = False,
) -> AudioBuffer:
    """Generate a randomized walking sequence.

    ``tempo`` is individual footfalls per minute. A numeric ``seed`` makes the
    result sample-for-sample repeatable.
    """

    _validate_settings(
        surface, footwear, steps, tempo, intensity, variation,
        stereo_width, sample_rate
    )
    surface_profile = SURFACES[surface]
    footwear_profile = FOOTWEAR[footwear]
    np_rng = np.random.default_rng(seed)
    timing_rng = random.Random(seed)
    interval = 60.0 / tempo
    start_padding = 0.035

    rendered: list[tuple[float, np.ndarray, float]] = []
    latest_end = 0.0
    for index in range(steps):
        jitter = timing_rng.uniform(-0.030, 0.030) * interval * variation if index else 0.0
        onset = start_padding + index * interval + jitter
        local_intensity = _clamp(
            intensity * (1.0 + timing_rng.uniform(-0.10, 0.10) * variation),
            0.0,
            1.0,
        )
        step = _render_step(
            surface_profile, footwear_profile, local_intensity,
            variation, sample_rate, np_rng
        )
        side = -1.0 if index % 2 == 0 else 1.0
        pan = _clamp(
            side * stereo_width
            * (0.82 + timing_rng.uniform(-0.12, 0.12) * variation),
            -1.0,
            1.0,
        )
        rendered.append((onset, step, pan))
        latest_end = max(latest_end, onset + len(step) / sample_rate)

    frame_count = max(1, round((latest_end + 0.055) * sample_rate))
    left = np.zeros(frame_count, dtype=float)
    right = np.zeros(frame_count, dtype=float)
    for onset, step, pan in rendered:
        start = round(onset * sample_rate)
        end = min(frame_count, start + len(step))
        usable = step[:end - start]
        angle = (pan + 1.0) * math.pi * 0.25
        left[start:end] += usable * math.cos(angle)
        right[start:end] += usable * math.sin(angle)

    if mono:
        mixed = (left + right) / math.sqrt(2.0)
        mastered = _master((mixed,), sample_rate)
    else:
        mastered = _master((left, right), sample_rate)
    return AudioBuffer(sample_rate, tuple(channel.tolist() for channel in mastered))


def write_wav(path: str | Path, audio: AudioBuffer) -> Path:
    return write_pcm16(path, audio)


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen footsteps",
        description="Synthesize a procedural footstep sequence as a 16-bit WAV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output", nargs="?", default="footsteps.wav", help="output WAV path"
    )
    parser.add_argument("--surface", choices=SURFACES, default="concrete")
    parser.add_argument("--footwear", choices=FOOTWEAR, default="sneaker")
    parser.add_argument("--steps", type=_positive_integer, default=8)
    parser.add_argument(
        "--tempo", type=float, default=112.0, help="individual steps per minute"
    )
    parser.add_argument("--intensity", type=_unit_interval, default=0.78)
    parser.add_argument(
        "--variation", type=_unit_interval, default=0.65,
        help="human timing/tone variation"
    )
    parser.add_argument(
        "--stereo-width", type=_unit_interval, default=0.38,
        help="left/right spread"
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--seed", type=int, help="repeatable random seed")
    parser.add_argument("--mono", action="store_true", help="write one channel")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        audio = generate_footsteps(
            surface=arguments.surface,
            footwear=arguments.footwear,
            steps=arguments.steps,
            tempo=arguments.tempo,
            intensity=arguments.intensity,
            variation=arguments.variation,
            stereo_width=arguments.stereo_width,
            sample_rate=arguments.sample_rate,
            seed=arguments.seed,
            mono=arguments.mono,
        )
        output = write_wav(arguments.output, audio)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    channel_word = "mono" if audio.channel_count == 1 else "stereo"
    print(
        f"Wrote {arguments.steps} {arguments.surface} footsteps "
        f"({audio.duration:.2f}s, {audio.sample_rate} Hz, {channel_word}) to {output}"
    )
    return 0


__all__ = [
    "AudioBuffer", "FOOTWEAR", "SURFACES", "build_parser",
    "generate_footsteps", "main", "write_wav",
]
