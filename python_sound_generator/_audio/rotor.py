"""Generic loop-safe propeller and fan blade-pass synthesis."""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import dsp, loops


SR = dsp.SR
TAU = dsp.TAU
MIN_DURATION = 0.5
MAX_DURATION = 10.0
CEILING = SR * 0.45


@dataclass(frozen=True, slots=True)
class RotorProfile:
    """Only the physical fields required by the shared blade-pass model."""

    name: str
    shaft_rate: float
    blades: int
    open_rotor: bool
    tone_gain: float
    noise_gain: float
    roughness: float
    idle: float


def _check_unit(value: float, name: str) -> float:
    if not 0.05 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.05 and 1")
    return value


def _check_duration(duration: float) -> float:
    if not MIN_DURATION <= duration <= MAX_DURATION:
        raise ValueError(
            f"duration must be between {MIN_DURATION} and {MAX_DURATION}"
        )
    return duration


def _vibrato(
    duration: float,
    rate: float,
    depth: float,
    rng: random.Random,
) -> list[float] | None:
    if depth <= 0.0:
        return None
    wave = loops.sine(rate, duration, rng.uniform(0.0, TAU), minimum=0)
    return [1.0 + depth * value for value in wave]


def _harmonic_stack(
    frequency: float,
    duration: float,
    rng: random.Random,
    count: int,
    rolloff: float = 1.0,
    vibrato: list[float] | None = None,
) -> list[float]:
    base = loops.quantize(frequency, duration, 1)
    output = dsp.silence(duration)
    for index in range(1, count + 1):
        partial = base * index
        if partial >= CEILING:
            break
        gain = 1.0 / (index ** rolloff)
        if vibrato is None:
            voice = dsp.sine(partial, duration, rng.uniform(0.0, TAU))
        else:
            voice = dsp.sine_track(
                [partial * value for value in vibrato],
                rng.uniform(0.0, TAU),
            )
        output = dsp.mix(output, dsp.scale(voice, gain))
    return output


def blade_pass(
    profile: RotorProfile,
    rng: random.Random,
    power: float = 0.6,
    duration: float = 2.5,
    intensity: float = 0.85,
) -> list[float]:
    """Render a seamless open-propeller or ducted-fan loop."""

    power = _check_unit(power, "power")
    intensity = _check_unit(intensity, "intensity")
    duration = _check_duration(duration)
    if profile.blades <= 0 or profile.shaft_rate <= 0.0:
        raise ValueError(f"{profile.name} has no propeller or fan")

    shaft = profile.shaft_rate * (
        profile.idle + (1.0 - profile.idle) * power
    )
    frequency = shaft * profile.blades
    hunt = _vibrato(
        duration,
        2.6 + 2.0 * rng.random(),
        0.004 + 0.006 * profile.roughness,
        rng,
    )
    partials = 9 if profile.open_rotor else 6
    tone = _harmonic_stack(frequency, duration, rng, partials, 1.25, hunt)

    low = max(40.0, frequency * 0.6)
    high = min(CEILING, frequency * (14.0 + 22.0 * power))
    swish = loops.bandpass(dsp.noise(duration, rng), low, high, 2)
    swish = dsp.mul(swish, loops.lfo(frequency, duration, 0.55))

    if profile.open_rotor:
        output = dsp.mix(
            dsp.scale(tone, 0.75 * profile.tone_gain),
            dsp.scale(swish, 0.90 * profile.noise_gain),
        )
    else:
        duct = loops.resonator(swish, frequency * 0.75, q=4.0, gain=0.6)
        output = dsp.mix(
            dsp.scale(tone, 1.00 * profile.tone_gain),
            dsp.scale(swish, 0.45 * profile.noise_gain),
            dsp.scale(duct, 0.40),
        )

    level = (0.22 + 0.78 * power ** 1.2) * (0.55 + 0.52 * intensity)
    return dsp.scale(output, level)


__all__ = ["RotorProfile", "blade_pass"]
