#!/usr/bin/env python3
"""Procedurally synthesize car engine sounds with the Python standard library.

The model is deliberately lightweight, but follows the useful physical cues:
a four-stroke firing rate, crankshaft harmonics, combustion pulses, resonant
exhaust plumbing, intake rush, valve-train ticks, load-dependent brightness,
gear-change dips, and exhaust pops on the overrun. No recordings or third-party
DSP packages are required.
"""

from __future__ import annotations

from ..._audio.pcm import AudioBuffer, write_pcm16

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


TAU = 2.0 * math.pi
DEFAULT_SAMPLE_RATE = 44_100
MODES = ("idle", "cruise", "accelerate", "decelerate", "rev")


@dataclass(frozen=True)
class EngineProfile:
    """Acoustic character and operating range for one engine layout."""

    label: str
    cylinders: int
    idle_rpm: float
    redline_rpm: float
    harmonics: tuple[float, ...]
    resonances: tuple[tuple[float, float, float], ...]
    unevenness: float
    exhaust_gain: float
    intake_gain: float
    mechanical_gain: float
    bass_gain: float
    brightness: float
    roughness: float




ENGINE_PROFILES: dict[str, EngineProfile] = {
    "inline3": EngineProfile(
        "Turbo inline-three",
        3,
        850.0,
        6_800.0,
        (1.00, 0.62, 0.34, 0.20, 0.13, 0.08),
        ((105.0, 1.1, 0.34), (315.0, 1.7, 0.23), (790.0, 2.2, 0.10)),
        0.18,
        1.00,
        0.33,
        0.19,
        0.36,
        0.85,
        0.25,
    ),
    "inline4": EngineProfile(
        "Inline-four",
        4,
        780.0,
        7_200.0,
        (1.00, 0.58, 0.36, 0.22, 0.14, 0.09),
        ((92.0, 1.1, 0.28), (280.0, 1.8, 0.22), (720.0, 2.3, 0.11)),
        0.09,
        0.94,
        0.30,
        0.22,
        0.30,
        0.92,
        0.18,
    ),
    "v6": EngineProfile(
        "V6",
        6,
        720.0,
        7_000.0,
        (1.00, 0.70, 0.42, 0.26, 0.17, 0.11, 0.07),
        ((86.0, 1.0, 0.32), (245.0, 1.6, 0.25), (610.0, 2.1, 0.13)),
        0.12,
        1.00,
        0.29,
        0.18,
        0.36,
        0.88,
        0.16,
    ),
    "inline6": EngineProfile(
        "Inline-six",
        6,
        680.0,
        7_200.0,
        (1.00, 0.76, 0.48, 0.29, 0.18, 0.11, 0.07),
        ((82.0, 1.1, 0.29), (225.0, 1.7, 0.22), (565.0, 2.4, 0.14)),
        0.035,
        1.00,
        0.27,
        0.16,
        0.32,
        0.94,
        0.08,
    ),
    "v8": EngineProfile(
        "Cross-plane V8",
        8,
        650.0,
        6_500.0,
        (1.00, 0.82, 0.54, 0.34, 0.22, 0.14, 0.09, 0.06),
        ((72.0, 0.9, 0.43), (188.0, 1.5, 0.31), (470.0, 2.0, 0.14)),
        0.24,
        1.10,
        0.23,
        0.16,
        0.48,
        0.75,
        0.24,
    ),
    "flat6": EngineProfile(
        "Flat-six",
        6,
        760.0,
        8_200.0,
        (1.00, 0.72, 0.45, 0.30, 0.21, 0.14, 0.09, 0.06),
        ((98.0, 1.2, 0.24), (335.0, 1.9, 0.21), (880.0, 2.6, 0.15)),
        0.045,
        0.96,
        0.34,
        0.18,
        0.24,
        1.08,
        0.08,
    ),
    "v10": EngineProfile(
        "High-rev V10",
        10,
        900.0,
        9_000.0,
        (1.00, 0.80, 0.55, 0.38, 0.27, 0.19, 0.13, 0.09, 0.06),
        ((112.0, 1.2, 0.20), (410.0, 2.0, 0.20), (1_080.0, 2.8, 0.17)),
        0.055,
        0.96,
        0.38,
        0.20,
        0.22,
        1.18,
        0.10,
    ),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _smoothstep(value: float) -> float:
    value = _clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _resolve_rpms(
    profile: EngineProfile,
    mode: str,
    start_rpm: float | None,
    end_rpm: float | None,
) -> tuple[float, float]:
    if start_rpm is None:
        if mode == "cruise":
            start_rpm = min(2_500.0, profile.redline_rpm * 0.40)
        elif mode == "decelerate":
            start_rpm = profile.redline_rpm * 0.82
        else:
            start_rpm = profile.idle_rpm

    if end_rpm is None:
        if mode in ("idle", "cruise"):
            end_rpm = start_rpm
        elif mode in ("accelerate", "rev"):
            end_rpm = profile.redline_rpm * 0.90
        else:
            end_rpm = profile.idle_rpm
    return float(start_rpm), float(end_rpm)


def _validate_settings(
    engine: str,
    mode: str,
    duration: float,
    start_rpm: float,
    end_rpm: float,
    throttle: float,
    variation: float,
    gear_shifts: int,
    stereo_width: float,
    sample_rate: int,
    peak_db: float,
) -> None:
    if engine not in ENGINE_PROFILES:
        raise ValueError(
            f"unknown engine {engine!r}; choose from {', '.join(ENGINE_PROFILES)}"
        )
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    if not 0.20 <= duration <= 120.0:
        raise ValueError("duration must be between 0.2 and 120 seconds")
    profile = ENGINE_PROFILES[engine]
    rpm_limit = profile.redline_rpm * 1.12
    if not 300.0 <= start_rpm <= rpm_limit:
        raise ValueError(f"start_rpm must be between 300 and {rpm_limit:.0f}")
    if not 300.0 <= end_rpm <= rpm_limit:
        raise ValueError(f"end_rpm must be between 300 and {rpm_limit:.0f}")
    if mode == "accelerate" and end_rpm < start_rpm:
        raise ValueError("accelerate mode requires end_rpm >= start_rpm")
    if mode == "decelerate" and end_rpm > start_rpm:
        raise ValueError("decelerate mode requires end_rpm <= start_rpm")
    if mode == "rev" and end_rpm < start_rpm:
        raise ValueError("rev mode requires peak end_rpm >= start_rpm")
    if not 0.0 <= throttle <= 1.0:
        raise ValueError("throttle must be between 0 and 1")
    if not 0.0 <= variation <= 1.0:
        raise ValueError("variation must be between 0 and 1")
    if not 0 <= gear_shifts <= 8:
        raise ValueError("gear_shifts must be between 0 and 8")
    if not 0.0 <= stereo_width <= 1.0:
        raise ValueError("stereo_width must be between 0 and 1")
    if not 8_000 <= sample_rate <= 192_000:
        raise ValueError("sample_rate must be between 8000 and 192000 Hz")
    if not -18.0 <= peak_db <= 0.0:
        raise ValueError("peak_db must be between -18 and 0 dBFS")


def _rpm_and_load_curves(
    profile: EngineProfile,
    mode: str,
    duration: float,
    start_rpm: float,
    end_rpm: float,
    throttle: float,
    variation: float,
    gear_shifts: int,
    sample_rate: int,
    rng: random.Random,
) -> tuple[list[float], list[float]]:
    """Build continuous RPM and engine-load control signals."""

    count = max(1, round(duration * sample_rate))
    rpms = [0.0] * count
    loads = [0.0] * count
    shift_positions = (
        [(index + 1) / (gear_shifts + 1) for index in range(gear_shifts)]
        if mode == "accelerate"
        else []
    )

    noise_target = 0.0
    noise_state = 0.0
    target_period = max(1, sample_rate // 12)
    smoothing = 1.0 - math.exp(-TAU * 2.2 / sample_rate)

    for index in range(count):
        progress = index / max(1, count - 1)
        if mode == "idle":
            base = start_rpm + (end_rpm - start_rpm) * _smoothstep(progress)
            load = 0.13 + 0.18 * throttle
            hunt = math.sin(TAU * 1.15 * index / sample_rate + 0.4)
            base += profile.idle_rpm * 0.018 * variation * hunt
        elif mode == "cruise":
            base = start_rpm + (end_rpm - start_rpm) * _smoothstep(progress)
            load = 0.28 + 0.42 * throttle
            base *= 1.0 + 0.006 * variation * math.sin(
                TAU * 0.42 * index / sample_rate
            )
        elif mode == "accelerate":
            base = start_rpm + (end_rpm - start_rpm) * progress ** 0.82
            load = 0.42 + 0.58 * throttle
        elif mode == "decelerate":
            base = start_rpm + (end_rpm - start_rpm) * _smoothstep(progress)
            load = 0.08 + 0.12 * throttle
        else:  # rev: climb, briefly hold, then return to the starting RPM
            if progress < 0.52:
                base = start_rpm + (end_rpm - start_rpm) * _smoothstep(
                    progress / 0.52
                )
                load = 0.16 + 0.84 * throttle * _smoothstep(progress / 0.10)
            elif progress < 0.62:
                base = end_rpm
                load = 0.32 + 0.68 * throttle
            else:
                release = _smoothstep((progress - 0.62) / 0.38)
                base = end_rpm + (start_rpm - end_rpm) * release
                load = 0.08 + 0.18 * throttle * (1.0 - release)

        for shift_at in shift_positions:
            elapsed = progress - shift_at
            if 0.0 <= elapsed <= 0.16:
                shift_shape = (
                    (1.0 - math.exp(-elapsed / 0.006))
                    * math.exp(-elapsed / 0.060)
                )
                base *= 1.0 - (0.105 + 0.035 * throttle) * shift_shape
                load *= 1.0 - 0.88 * shift_shape

        if index % target_period == 0:
            noise_target = rng.uniform(-1.0, 1.0)
        noise_state += smoothing * (noise_target - noise_state)
        rough_scale = (0.004 + 0.014 * profile.roughness) * variation
        base *= 1.0 + noise_state * rough_scale
        rpms[index] = _clamp(base, 300.0, profile.redline_rpm * 1.12)
        loads[index] = _clamp(load, 0.03, 1.0)
    return rpms, loads


def _fixed_lowpass(
    samples: Sequence[float], cutoff: float, sample_rate: int
) -> list[float]:
    cutoff = _clamp(cutoff, 10.0, sample_rate * 0.475)
    coefficient = 1.0 - math.exp(-TAU * cutoff / sample_rate)
    state = 0.0
    result: list[float] = []
    append = result.append
    for sample in samples:
        state += coefficient * (sample - state)
        append(state)
    return result


def _fixed_highpass(
    samples: Sequence[float], cutoff: float, sample_rate: int
) -> list[float]:
    low = _fixed_lowpass(samples, cutoff, sample_rate)
    return [sample - bass for sample, bass in zip(samples, low)]


def _dynamic_lowpass(
    samples: Sequence[float], cutoffs: Sequence[float], sample_rate: int
) -> list[float]:
    state = 0.0
    result = [0.0] * len(samples)
    limit = sample_rate * 0.475
    for index, (sample, cutoff) in enumerate(zip(samples, cutoffs)):
        cutoff = _clamp(cutoff, 10.0, limit)
        coefficient = 1.0 - math.exp(-TAU * cutoff / sample_rate)
        state += coefficient * (sample - state)
        result[index] = state
    return result


def _resonant_bandpass(
    samples: Sequence[float], frequency: float, q: float, sample_rate: int
) -> list[float]:
    """Apply a constant-peak-gain RBJ band-pass biquad."""

    frequency = _clamp(frequency, 20.0, sample_rate * 0.45)
    q = max(0.2, q)
    omega = TAU * frequency / sample_rate
    alpha = math.sin(omega) / (2.0 * q)
    a0 = 1.0 + alpha
    b0 = alpha / a0
    b2 = -alpha / a0
    a1 = -2.0 * math.cos(omega) / a0
    a2 = (1.0 - alpha) / a0
    x1 = x2 = y1 = y2 = 0.0
    result = [0.0] * len(samples)
    for index, sample in enumerate(samples):
        output = b0 * sample + b2 * x2 - a1 * y1 - a2 * y2
        result[index] = output
        x2, x1 = x1, sample
        y2, y1 = y1, output
    return result


def _dc_block(samples: Sequence[float], sample_rate: int) -> list[float]:
    pole = math.exp(-TAU * 14.0 / sample_rate)
    previous_input = 0.0
    previous_output = 0.0
    result: list[float] = []
    append = result.append
    for sample in samples:
        output = sample - previous_input + pole * previous_output
        append(output)
        previous_input = sample
        previous_output = output
    return result


def _synthesize_layers(
    profile: EngineProfile,
    mode: str,
    rpms: Sequence[float],
    loads: Sequence[float],
    sample_rate: int,
    rng: random.Random,
) -> tuple[list[float], list[float], list[float], list[float]]:
    count = len(rpms)
    combustion = [0.0] * count
    intake_raw = [0.0] * count
    mechanical_raw = [0.0] * count
    bass_raw = [0.0] * count
    exhaust_cutoffs = [0.0] * count
    intake_low_cutoffs = [0.0] * count
    intake_high_cutoffs = [0.0] * count

    crank_phase = rng.uniform(0.0, TAU)
    firing_phase = rng.uniform(0.0, TAU)
    harmonic_phases = [rng.uniform(-0.18, 0.18) for _ in profile.harmonics]
    event_gain = 1.0
    tick_envelope = 0.0
    tick_decay = math.exp(-1.0 / (sample_rate * 0.0032))
    pop_envelope = 0.0
    pop_phase = 0.0
    pop_frequency = 75.0
    pop_decay = math.exp(-1.0 / (sample_rate * 0.035))

    for index, (rpm, load) in enumerate(zip(rpms, loads)):
        rpm_ratio = _clamp(rpm / profile.redline_rpm, 0.0, 1.2)
        crank_step = TAU * (rpm / 60.0) / sample_rate
        firing_step = crank_step * profile.cylinders / 2.0
        crank_phase = (crank_phase + crank_step) % TAU
        previous_firing = firing_phase
        firing_phase = (firing_phase + firing_step) % TAU
        fired = firing_phase < previous_firing
        if fired:
            event_gain = 1.0 + rng.uniform(-0.20, 0.20) * profile.roughness
            tick_envelope = 1.0

        bank_warp = profile.unevenness * (
            0.70 * math.sin(crank_phase) + 0.30 * math.sin(2.0 * crank_phase + 0.6)
        )
        pulse_phase = firing_phase + bank_warp
        pulse = max(0.0, math.cos(pulse_phase)) ** 10 * event_gain
        harmonic_voice = 0.0
        for harmonic, gain in enumerate(profile.harmonics, start=1):
            harmonic_voice += gain * math.sin(
                harmonic * pulse_phase + harmonic_phases[harmonic - 1]
            )
        harmonic_voice /= max(1.0, sum(profile.harmonics))

        combustion_noise = rng.uniform(-1.0, 1.0)
        drive = 0.34 + 0.82 * load
        raw = (
            1.34 * pulse
            + 0.58 * harmonic_voice
            + 0.10 * combustion_noise * (0.25 + pulse)
        ) * drive * profile.exhaust_gain

        released = mode in ("decelerate", "rev") and load < 0.25 and rpm_ratio > 0.22
        pop_rate = 1.5 + 7.0 * profile.roughness
        if released and rng.random() < pop_rate / sample_rate:
            pop_envelope = rng.uniform(0.55, 1.0)
            pop_frequency = rng.uniform(58.0, 105.0)
            pop_phase = rng.uniform(0.0, TAU)
        if pop_envelope > 1e-5:
            pop_phase += TAU * pop_frequency / sample_rate
            pop = (
                math.sin(pop_phase) + 0.42 * rng.uniform(-1.0, 1.0)
            ) * pop_envelope
            raw += pop * (0.42 + 0.45 * profile.bass_gain)
            pop_envelope *= pop_decay

        combustion[index] = math.tanh(raw * 1.18)
        bass_raw[index] = (
            0.72 * pulse + 0.28 * math.sin(crank_phase) + 0.18 * harmonic_voice
        ) * profile.bass_gain * (0.42 + 0.58 * load)

        intake_gate = 0.24 + 0.76 * min(1.0, pulse * 1.7)
        intake_raw[index] = (
            rng.uniform(-1.0, 1.0)
            * intake_gate
            * (0.18 + 0.82 * load)
            * profile.intake_gain
        )

        tick_envelope *= tick_decay
        valve_tone = (
            0.40 * math.sin(2.0 * firing_phase + 0.2)
            + 0.23 * math.sin(3.0 * firing_phase + 1.1)
            + 0.12 * math.sin(5.0 * firing_phase + 0.5)
        )
        tick = tick_envelope * rng.uniform(-1.0, 1.0)
        mechanical_raw[index] = (
            valve_tone * (0.18 + 0.82 * rpm_ratio)
            + 0.72 * tick
        ) * profile.mechanical_gain

        exhaust_cutoffs[index] = (
            420.0
            + profile.brightness
            * (720.0 + 3_900.0 * load + 2_200.0 * rpm_ratio)
        )
        intake_low_cutoffs[index] = 170.0 + 480.0 * rpm_ratio
        intake_high_cutoffs[index] = (
            1_050.0 + profile.brightness * (2_400.0 + 5_000.0 * rpm_ratio)
        )

    exhaust = _dynamic_lowpass(combustion, exhaust_cutoffs, sample_rate)
    for frequency, q, gain in profile.resonances:
        resonated = _resonant_bandpass(combustion, frequency, q, sample_rate)
        for index, value in enumerate(resonated):
            exhaust[index] += value * gain

    intake_bass = _dynamic_lowpass(intake_raw, intake_low_cutoffs, sample_rate)
    intake_high = [
        sample - bass for sample, bass in zip(intake_raw, intake_bass)
    ]
    intake = _dynamic_lowpass(intake_high, intake_high_cutoffs, sample_rate)
    mechanical = _fixed_highpass(mechanical_raw, 145.0, sample_rate)
    bass = _fixed_lowpass(bass_raw, 210.0, sample_rate)
    return exhaust, intake, mechanical, bass


def _master(
    channels: Sequence[Sequence[float]], sample_rate: int, peak_db: float
) -> tuple[list[float], ...]:
    processed = [_dc_block(channel, sample_rate) for channel in channels]
    if not processed:
        return ()

    drive = 1.22
    drive_scale = math.tanh(drive)
    for channel in processed:
        for index, sample in enumerate(channel):
            channel[index] = math.tanh(sample * drive) / drive_scale

    # Asymmetric combustion pulses can pick up a small offset in the saturator,
    # so block DC once more after the nonlinear stage.
    processed = [_dc_block(channel, sample_rate) for channel in processed]

    fade_in = min(len(processed[0]), max(2, round(0.012 * sample_rate)))
    fade_out = min(len(processed[0]), max(2, round(0.055 * sample_rate)))
    for channel in processed:
        for index in range(fade_in):
            channel[index] *= index / (fade_in - 1)
        for index in range(fade_out):
            channel[-fade_out + index] *= 1.0 - index / (fade_out - 1)

    peak = max((abs(value) for channel in processed for value in channel), default=0.0)
    if peak > 1e-12:
        target = 10.0 ** (peak_db / 20.0)
        scale = target / peak
        for channel in processed:
            for index, sample in enumerate(channel):
                channel[index] = _clamp(sample * scale, -1.0, 1.0)
    return tuple(processed)


def generate_engine_sound(
    engine: str = "v8",
    mode: str = "accelerate",
    duration: float = 6.0,
    start_rpm: float | None = None,
    end_rpm: float | None = None,
    throttle: float = 0.82,
    variation: float = 0.45,
    gear_shifts: int = 2,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
    stereo: bool = False,
    stereo_width: float = 0.55,
    peak_db: float = -1.0,
) -> AudioBuffer:
    """Generate an engine performance as floating-point PCM audio.

    A four-stroke engine produces cylinders / 2 combustion events per
    crankshaft revolution. The rev mode treats end_rpm as the peak and returns
    to start_rpm. Gear shifts only affect accelerate mode. Mono is the default
    because it spatializes cleanly as a game sound source.
    """

    if engine not in ENGINE_PROFILES:
        raise ValueError(
            f"unknown engine {engine!r}; choose from {', '.join(ENGINE_PROFILES)}"
        )
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    profile = ENGINE_PROFILES[engine]
    resolved_start, resolved_end = _resolve_rpms(
        profile, mode, start_rpm, end_rpm
    )
    _validate_settings(
        engine,
        mode,
        duration,
        resolved_start,
        resolved_end,
        throttle,
        variation,
        gear_shifts,
        stereo_width,
        sample_rate,
        peak_db,
    )

    rng = random.Random(seed)
    rpms, loads = _rpm_and_load_curves(
        profile,
        mode,
        duration,
        resolved_start,
        resolved_end,
        throttle,
        variation,
        gear_shifts,
        sample_rate,
        rng,
    )
    exhaust, intake, mechanical, bass = _synthesize_layers(
        profile, mode, rpms, loads, sample_rate, rng
    )
    center = [
        0.72 * ex + 0.24 * air + 0.23 * mech + 0.34 * low
        for ex, air, mech, low in zip(exhaust, intake, mechanical, bass)
    ]

    if stereo:
        delay = max(1, round(0.0007 * sample_rate))
        left = [0.0] * len(center)
        right = [0.0] * len(center)
        for index, value in enumerate(center):
            delayed = exhaust[index - delay] if index >= delay else 0.0
            side = (
                0.20 * (intake[index] - mechanical[index])
                + 0.085 * (exhaust[index] - delayed)
            ) * stereo_width
            left[index] = value + side
            right[index] = value - side
        channels = _master((left, right), sample_rate, peak_db)
    else:
        channels = _master((center,), sample_rate, peak_db)
    return AudioBuffer(sample_rate, channels)


def write_wav(path: str | Path, audio: AudioBuffer) -> Path:
    return write_pcm16(path, audio)


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen car-engine procedural",
        description=(
            "Synthesize a car engine performance as 16-bit PCM WAV using only "
            "the Python standard library."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output", nargs="?", default="car_engine.wav", help="output WAV path"
    )
    parser.add_argument("--engine", choices=ENGINE_PROFILES, default="v8")
    parser.add_argument("--mode", choices=MODES, default="accelerate")
    parser.add_argument("--duration", type=float, default=6.0, help="seconds")
    parser.add_argument("--start-rpm", type=float)
    parser.add_argument(
        "--end-rpm",
        type=float,
        help="ending RPM, or peak RPM in rev mode",
    )
    parser.add_argument("--throttle", type=_unit_interval, default=0.82)
    parser.add_argument(
        "--variation",
        type=_unit_interval,
        default=0.45,
        help="combustion and RPM irregularity",
    )
    parser.add_argument(
        "--gear-shifts",
        type=_nonnegative_integer,
        default=2,
        help="RPM/load dips in accelerate mode",
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--seed", type=int, help="repeatable random seed")
    parser.add_argument(
        "--stereo", action="store_true", help="write a subtle stereo image"
    )
    parser.add_argument("--stereo-width", type=_unit_interval, default=0.55)
    parser.add_argument(
        "--peak-db", type=float, default=-1.0, help="normalized peak in dBFS"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        audio = generate_engine_sound(
            engine=arguments.engine,
            mode=arguments.mode,
            duration=arguments.duration,
            start_rpm=arguments.start_rpm,
            end_rpm=arguments.end_rpm,
            throttle=arguments.throttle,
            variation=arguments.variation,
            gear_shifts=arguments.gear_shifts,
            sample_rate=arguments.sample_rate,
            seed=arguments.seed,
            stereo=arguments.stereo,
            stereo_width=arguments.stereo_width,
            peak_db=arguments.peak_db,
        )
        output = write_wav(arguments.output, audio)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    channels = "stereo" if audio.channel_count == 2 else "mono"
    print(
        f"Wrote {ENGINE_PROFILES[arguments.engine].label} {arguments.mode} "
        f"({audio.duration:.2f}s, {audio.sample_rate} Hz, {channels}) to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
