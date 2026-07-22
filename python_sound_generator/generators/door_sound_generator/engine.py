"""Seeded door Foley variations built from real CC0/public-domain recordings.

NumPy and SciPy handle source selection, silence trimming, material-dependent
playback rate and EQ, action layering, early reflections, and mastering.  The
bundled recordings are kept as unmodified mono PCM conversions under
``sources/`` with their provenance documented beside them.
"""

from __future__ import annotations

from ..._audio.assets import resolve_asset

from ..._audio.foley_numpy import read_recording

from ..._audio.pcm import AudioBuffer, write_pcm16

from ..._audio.foley_numpy import trim_activity as trim_recording_activity

from ..._audio.foley_numpy import mix_at_frames


import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, resample, resample_poly, sosfilt


DEFAULT_SAMPLE_RATE = 44_100
SOURCE_DIRECTORY = Path(__file__).resolve().parent / "sources"


@dataclass(frozen=True)
class MaterialProfile:
    """Playback and equalization character applied to recorded Foley."""

    label: str
    speed: float
    warmth: float
    brightness: float
    creak_bias: float


@dataclass(frozen=True)
class ActionProfile:
    """Default render length and mastering level for one gesture."""

    label: str
    duration: float
    target_peak: float




MATERIALS: dict[str, MaterialProfile] = {
    "light_wood": MaterialProfile(
        "Light wooden interior door", 1.055, -0.04, 0.13, 0.03
    ),
    "heavy_wood": MaterialProfile(
        "Heavy wooden door", 0.900, 0.25, -0.05, 0.11
    ),
    "hollow_wood": MaterialProfile(
        "Hollow wooden door", 0.985, 0.10, 0.15, 0.19
    ),
    "metal": MaterialProfile(
        "Metal utility door", 1.020, -0.10, 0.28, 0.26
    ),
}


ACTIONS: dict[str, ActionProfile] = {
    "open": ActionProfile("Open", 1.65, 0.74),
    "close": ActionProfile("Close", 1.25, 0.80),
    "slam": ActionProfile("Slam", 0.85, 0.91),
    "creak": ActionProfile("Hinge creak", 1.90, 0.70),
    "latch": ActionProfile("Handle and latch", 0.65, 0.66),
}


CLEAN_OPEN = (
    "laleksic_door_open.wav",
    "rubberduck_door_open.wav",
    "iwan_door-01.wav",
    "iwan_door-04.wav",
    "iwan_door-06.wav",
    "iwan_door-08.wav",
)
CLEAN_CLOSE = (
    "laleksic_door_close.wav",
    "rubberduck_door_close_01.wav",
    "rubberduck_door_close_02.wav",
    "rubberduck_door_close_03.wav",
    "rubberduck_door_close_04.wav",
    "iwan_door-02.wav",
    "iwan_door-03.wav",
    "iwan_door-05.wav",
    "iwan_door-07.wav",
    "iwan_door-09.wav",
)
CREAKY_OPEN = ("laleksic_door_creak_open.wav",)
CREAKY_CLOSE = ("laleksic_door_creak_close.wav",)
HINGES = (
    "mkfi_squeaky_hinge.wav",
    "leonmire_squeaky_door.wav",
    "rubberduck_door_01.wav",
    "rubberduck_door_02.wav",
)
LATCHES = (
    "laleksic_lock.wav",
    "laleksic_cant_open.wav",
    "iwan_door-03.wav",
    "iwan_door-05.wav",
    "iwan_door-09.wav",
)
SLAMS = tuple(f"rubberduck_slam_{index:02d}.wav" for index in range(1, 8))




def _trim_activity(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    return trim_recording_activity(
            signal, sample_rate=sample_rate, window_seconds=0.012,
            threshold_ratio=0.018, threshold_floor_db=-58.0,
            pre_seconds=0.018, post_seconds=0.018,
        )


@lru_cache(maxsize=64)
def _load_source(filename: str) -> tuple[int, np.ndarray]:
    path = resolve_asset(SOURCE_DIRECTORY / filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing bundled door source {path}; see sources/provenance.json"
        )
    return read_recording(
        path, expected_rate=None,
        trim=lambda data, rate: _trim_activity(data, rate),
        label="door source",
    )


def _speed_and_resample(
    signal: np.ndarray,
    source_rate: int,
    output_rate: int,
    speed: float,
) -> np.ndarray:
    ratio = Fraction(output_rate / (source_rate * speed)).limit_denominator(2_000)
    if ratio.numerator == ratio.denominator:
        return signal.copy()
    return resample_poly(signal, ratio.numerator, ratio.denominator)


def _material_eq(
    signal: np.ndarray,
    profile: MaterialProfile,
    sample_rate: int,
) -> np.ndarray:
    low_cutoff = min(310.0, sample_rate * 0.18)
    high_cutoff = min(2_650.0, sample_rate * 0.36)
    low = sosfilt(
        butter(2, low_cutoff, btype="lowpass", fs=sample_rate, output="sos"),
        signal,
    )
    high_reference = sosfilt(
        butter(2, high_cutoff, btype="lowpass", fs=sample_rate, output="sos"),
        signal,
    )
    high = signal - high_reference
    return signal + profile.warmth * low + profile.brightness * high


def _prepare_source(
    filename: str,
    profile: MaterialProfile,
    sample_rate: int,
    rng: np.random.Generator,
    speed_multiplier: float = 1.0,
) -> np.ndarray:
    source_rate, cached = _load_source(filename)
    speed = profile.speed * speed_multiplier * rng.uniform(0.975, 1.026)
    signal = _speed_and_resample(cached, source_rate, sample_rate, speed)
    signal = _material_eq(signal, profile, sample_rate)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal = signal / peak
    fade = min(len(signal) // 2, max(1, round(0.004 * sample_rate)))
    if fade:
        signal[:fade] *= np.linspace(0.0, 1.0, fade)
        signal[-fade:] *= np.linspace(1.0, 0.0, fade)
    return signal


def _choose(rng: np.random.Generator, choices: Sequence[str]) -> str:
    return choices[int(rng.integers(0, len(choices)))]


def _activity_crop(
    signal: np.ndarray,
    frame_count: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(signal) <= frame_count:
        return signal.copy()
    envelope = np.sqrt(
        uniform_filter1d(signal * signal, max(3, round(0.045 * sample_rate)))
    )
    candidates = np.flatnonzero(envelope >= np.percentile(envelope, 58.0))
    center = int(rng.choice(candidates)) if len(candidates) else len(signal) // 2
    start = center - round(frame_count * rng.uniform(0.28, 0.70))
    start = min(max(0, start), len(signal) - frame_count)
    result = signal[start : start + frame_count].copy()
    fade = min(len(result) // 2, max(1, round(0.020 * sample_rate)))
    result[:fade] *= np.linspace(0.0, 1.0, fade)
    result[-fade:] *= np.linspace(1.0, 0.0, fade)
    return result


def _fit_primary(signal: np.ndarray, maximum_frames: int) -> np.ndarray:
    if len(signal) <= maximum_frames:
        return signal.copy()
    return resample(signal, maximum_frames)


def _mix_at(destination: np.ndarray, source: np.ndarray, start_frame: int, gain: float) -> None:
    mix_at_frames(destination, source, start_frame, gain)


def _real_movement_body(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract low/mid movement from the recording itself, not generated noise."""

    highpass = butter(2, 38.0, btype="highpass", fs=sample_rate, output="sos")
    lowpass = butter(
        3, min(2_400.0, sample_rate * 0.34), btype="lowpass", fs=sample_rate, output="sos"
    )
    result = sosfilt(highpass, signal)
    result = sosfilt(lowpass, result)
    return result


def _render_open_or_close(
    action: str,
    profile: MaterialProfile,
    frame_count: int,
    intensity: float,
    age: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    output = np.zeros(frame_count, dtype=float)
    probability = min(0.97, 0.08 + 0.78 * age + profile.creak_bias)
    use_creaky = bool(rng.random() < probability)
    if action == "open":
        pool = CREAKY_OPEN if use_creaky else CLEAN_OPEN
    else:
        pool = CREAKY_CLOSE if use_creaky else CLEAN_CLOSE
    primary_name = _choose(rng, pool)
    primary = _prepare_source(primary_name, profile, sample_rate, rng)
    primary = _fit_primary(primary, max(1, frame_count - round(0.05 * sample_rate)))
    start = round(0.025 * sample_rate)
    _mix_at(output, primary, start, 0.84)

    # Reinforce the real recorded panel movement without synthesizing an air bed.
    movement = _real_movement_body(primary, sample_rate)
    _mix_at(output, movement, start + round(0.006 * sample_rate), 0.20 + 0.08 * intensity)
    used = [primary_name]

    if not use_creaky and age > 0.18:
        hinge_name = _choose(rng, HINGES)
        hinge = _prepare_source(hinge_name, profile, sample_rate, rng)
        desired = min(
            len(hinge),
            max(1, frame_count - round(0.18 * sample_rate)),
        )
        hinge = _activity_crop(hinge, desired, sample_rate, rng)
        _mix_at(
            output,
            hinge,
            round(0.13 * sample_rate),
            (0.10 + 0.26 * age) * (0.9 if action == "open" else 0.75),
        )
        used.append(hinge_name)

    # A real lock/handle recording adds mechanical definition at the correct end.
    latch_name = _choose(rng, LATCHES)
    latch = _prepare_source(latch_name, profile, sample_rate, rng, 1.04)
    latch = _activity_crop(
        latch, min(len(latch), round(0.32 * sample_rate)), sample_rate, rng
    )
    if action == "open":
        latch_start = round(0.015 * sample_rate)
    else:
        peak = int(np.argmax(np.abs(primary)))
        latch_start = start + peak - int(np.argmax(np.abs(latch)))
    _mix_at(output, latch, latch_start, 0.085 + 0.045 * intensity)
    used.append(latch_name)
    return output, used


def _render_slam(
    profile: MaterialProfile,
    frame_count: int,
    intensity: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    output = np.zeros(frame_count, dtype=float)
    body_name = _choose(rng, CLEAN_CLOSE)
    body = _prepare_source(body_name, profile, sample_rate, rng, 1.10)
    body = _fit_primary(body, max(1, frame_count - round(0.04 * sample_rate)))
    _mix_at(output, body, round(0.018 * sample_rate), 0.56)
    movement = _real_movement_body(body, sample_rate)
    _mix_at(output, movement, round(0.022 * sample_rate), 0.24)

    impact_name = _choose(rng, SLAMS)
    impact = _prepare_source(impact_name, profile, sample_rate, rng, 0.96)
    impact = _fit_primary(impact, max(1, round(0.58 * sample_rate)))
    impact_peak = int(np.argmax(np.abs(impact)))
    target_peak = round(frame_count * rng.uniform(0.40, 0.53))
    _mix_at(output, impact, target_peak - impact_peak, 0.80 + 0.20 * intensity)
    return output, [body_name, impact_name]


def _render_creak(
    profile: MaterialProfile,
    frame_count: int,
    intensity: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    output = np.zeros(frame_count, dtype=float)
    primary_name = _choose(rng, HINGES)
    primary = _prepare_source(primary_name, profile, sample_rate, rng)
    primary = _activity_crop(
        primary, max(1, frame_count - round(0.08 * sample_rate)), sample_rate, rng
    )
    _mix_at(output, primary, round(0.04 * sample_rate), 0.88)
    movement = _real_movement_body(primary, sample_rate)
    _mix_at(output, movement, round(0.045 * sample_rate), 0.18 + 0.06 * intensity)
    return output, [primary_name]


def _render_latch(
    profile: MaterialProfile,
    frame_count: int,
    intensity: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    output = np.zeros(frame_count, dtype=float)
    primary_name = _choose(rng, LATCHES)
    primary = _prepare_source(primary_name, profile, sample_rate, rng, 1.03)
    primary = _activity_crop(
        primary, min(len(primary), max(1, frame_count - round(0.06 * sample_rate))), sample_rate, rng
    )
    _mix_at(output, primary, round(0.03 * sample_rate), 0.92)
    return output, [primary_name]


def _early_reflections(
    signal: np.ndarray,
    room: float,
    sample_rate: int,
    rng: np.random.Generator,
    side: float = 0.0,
) -> np.ndarray:
    result = signal.copy()
    if room <= 0.0:
        return result
    for index in range(4):
        delay = round(
            rng.uniform(0.007, 0.034) * sample_rate * (1.0 + 0.08 * side)
        )
        gain = room * rng.uniform(0.030, 0.080) * math.exp(-0.32 * index)
        if 0 < delay < len(signal):
            result[delay:] += signal[:-delay] * gain
    return result


def _master_channels(
    channels: Sequence[np.ndarray],
    action: str,
    intensity: float,
    sample_rate: int,
) -> tuple[np.ndarray, ...]:
    highpass = butter(2, 24.0, btype="highpass", fs=sample_rate, output="sos")
    lowpass = butter(
        2, min(17_000.0, sample_rate * 0.45), btype="lowpass", fs=sample_rate, output="sos"
    )
    processed: list[np.ndarray] = []
    for source in channels:
        signal = sosfilt(highpass, source)
        signal = sosfilt(lowpass, signal)
        fade_in = min(len(signal), max(1, round(0.002 * sample_rate)))
        fade_out = min(len(signal), max(1, round(0.025 * sample_rate)))
        signal[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
        signal[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
        processed.append(signal)

    peak = max(float(np.max(np.abs(channel))) for channel in processed)
    target = ACTIONS[action].target_peak * (0.80 + 0.20 * min(1.0, intensity))
    if peak > 1e-12:
        gain = target / peak
        processed = [channel * gain for channel in processed]
    return tuple(np.clip(channel, -0.98, 0.98) for channel in processed)


def generate_door_sound(
    action: str = "close",
    material: str = "heavy_wood",
    *,
    intensity: float = 0.85,
    age: float = 0.65,
    duration: float | None = None,
    room: float = 0.08,
    seed: int | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    stereo: bool = False,
) -> AudioBuffer:
    """Generate one seeded variation from the bundled real Foley bank."""

    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}; choose from {', '.join(ACTIONS)}")
    if material not in MATERIALS:
        raise ValueError(
            f"unknown material {material!r}; choose from {', '.join(MATERIALS)}"
        )
    if not 0.05 <= intensity <= 1.5:
        raise ValueError("intensity must be between 0.05 and 1.5")
    if not 0.0 <= age <= 1.0:
        raise ValueError("age must be between 0 and 1")
    if not 0.0 <= room <= 1.0:
        raise ValueError("room must be between 0 and 1")
    if sample_rate < 16_000:
        raise ValueError("sample_rate must be at least 16000 Hz")
    if duration is None:
        duration = ACTIONS[action].duration
    minimum_duration = 0.08 if action == "latch" else 0.25
    if not minimum_duration <= duration <= 8.0:
        raise ValueError(
            f"duration must be between {minimum_duration:g} and 8 seconds "
            f"for {action}"
        )

    rng = np.random.default_rng(seed)
    frame_count = round(duration * sample_rate)
    profile = MATERIALS[material]
    if action in ("open", "close"):
        raw, used = _render_open_or_close(
            action, profile, frame_count, intensity, age, sample_rate, rng
        )
    elif action == "slam":
        raw, used = _render_slam(
            profile, frame_count, intensity, sample_rate, rng
        )
    elif action == "creak":
        raw, used = _render_creak(
            profile, frame_count, intensity, sample_rate, rng
        )
    else:
        raw, used = _render_latch(
            profile, frame_count, intensity, sample_rate, rng
        )

    raw *= intensity
    if stereo:
        left = _early_reflections(raw * 0.99, room, sample_rate, rng, -1.0)
        right = _early_reflections(raw * 0.94, room, sample_rate, rng, 1.0)
        delay = max(1, round(0.0006 * sample_rate))
        right = np.concatenate((np.zeros(delay), right[:-delay]))
        rendered = _master_channels((left, right), action, intensity, sample_rate)
    else:
        mono = _early_reflections(raw, room, sample_rate, rng)
        rendered = _master_channels((mono,), action, intensity, sample_rate)

    return AudioBuffer(
        sample_rate,
        tuple(channel.astype(float).tolist() for channel in rendered),
        tuple(used),
    )


def write_wav(path: str | Path, audio: AudioBuffer) -> Path:
    return write_pcm16(path, audio)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="door.wav")
    parser.add_argument("--action", choices=tuple(ACTIONS), default="close")
    parser.add_argument("--material", choices=tuple(MATERIALS), default="heavy_wood")
    parser.add_argument("--intensity", type=float, default=0.85)
    parser.add_argument("--age", type=float, default=0.65)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--room", type=float, default=0.08)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--stereo", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    audio = generate_door_sound(
        action=arguments.action,
        material=arguments.material,
        intensity=arguments.intensity,
        age=arguments.age,
        duration=arguments.duration,
        room=arguments.room,
        seed=arguments.seed,
        sample_rate=arguments.sample_rate,
        stereo=arguments.stereo,
    )
    path = write_wav(arguments.output, audio)
    print(
        f"Generated {ACTIONS[arguments.action].label.lower()} / "
        f"{MATERIALS[arguments.material].label.lower()}: {path} "
        f"({audio.duration:.3f}s, {audio.sample_rate} Hz, "
        f"{audio.channel_count} channel{'s' if audio.channel_count != 1 else ''})"
    )
    print("Sources: " + ", ".join(audio.sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
