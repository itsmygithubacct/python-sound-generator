"""Hybrid procedural and CC0-Foley synthesis for ninja game sounds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import butter, iirpeak, lfilter, sosfilt

from ..._audio.foley_numpy import SR, master as shared_master
from ..._audio.foley_numpy import write_wav as write_wav
from . import foley


TAU = math.tau


@dataclass(frozen=True, slots=True)
class Style:
    """One production treatment spanning procedural and recorded layers."""

    name: str
    description: str
    speed: float
    brightness: float
    body: float
    air: float
    foley_gain: float
    room: float
    drive: float


STYLES: dict[str, Style] = {
    "shadow": Style(
        "shadow",
        "restrained, close, dry, and optimized for readable stealth gameplay",
        1.06,
        0.78,
        0.72,
        0.70,
        0.84,
        0.04,
        1.10,
    ),
    "arcade": Style(
        "arcade",
        "fast, bright, exaggerated transients with compact tonal punctuation",
        1.17,
        1.24,
        0.86,
        1.28,
        0.66,
        0.10,
        1.42,
    ),
    "cinematic": Style(
        "cinematic",
        "weighty Foley, darker impacts, broad air, and a longer reflected tail",
        0.92,
        0.88,
        1.30,
        1.08,
        1.00,
        0.22,
        1.28,
    ),
}


def style_named(name: str) -> Style:
    try:
        return STYLES[name]
    except KeyError:
        raise ValueError(
            f"unknown style {name!r}; expected one of {sorted(STYLES)}"
        ) from None


def frames(seconds: float) -> int:
    return max(1, round(seconds * SR))


def canvas(seconds: float) -> np.ndarray:
    return np.zeros(frames(seconds), dtype=np.float64)


def add_at(
    destination: np.ndarray,
    onset: float,
    source: Sequence[float],
    gain: float = 1.0,
) -> None:
    start = round(onset * SR)
    data = np.asarray(source, dtype=np.float64)
    if start < 0:
        data = data[-start:]
        start = 0
    if start >= len(destination) or len(data) == 0:
        return
    span = min(len(data), len(destination) - start)
    destination[start:start + span] += data[:span] * gain


def _band(signal: np.ndarray, low: float, high: float) -> np.ndarray:
    low = max(20.0, min(low, SR * 0.44))
    high = max(low + 40.0, min(high, SR * 0.48))
    return sosfilt(
        butter(3, (low, high), btype="bandpass", fs=SR, output="sos"),
        signal,
    )


def _fade(signal: np.ndarray, attack: float, release: float) -> np.ndarray:
    output = np.asarray(signal, dtype=np.float64).copy()
    attack_frames = min(len(output), max(1, round(attack * SR)))
    release_frames = min(len(output), max(1, round(release * SR)))
    if attack_frames > 1:
        output[:attack_frames] *= np.linspace(0.0, 1.0, attack_frames)
    if release_frames > 1:
        output[-release_frames:] *= np.linspace(1.0, 0.0, release_frames)
    return output


def whoosh(
    seconds: float,
    style: Style,
    rng: np.random.Generator,
    *,
    low: float = 260.0,
    high: float = 9_000.0,
    reverse: bool = False,
) -> np.ndarray:
    """Broadband movement with a shaped center and a faint blade whistle."""

    count = frames(seconds)
    noise = _band(
        rng.standard_normal(count),
        low,
        min(SR * 0.47, high * style.brightness),
    )
    position = np.linspace(0.0, 1.0, count, endpoint=False)
    center = np.sin(math.pi * position) ** 1.7
    if reverse:
        center = center[::-1]
    rate = np.linspace(420.0, 1_600.0 * style.brightness, count)
    if reverse:
        rate = rate[::-1]
    phase = np.cumsum(rate / SR) * TAU
    whistle = np.sin(phase) * center
    return _fade(
        noise * center * 0.72 + whistle * 0.12 * style.air,
        0.004,
        min(0.11, seconds * 0.32),
    )


def noise_hit(
    seconds: float,
    style: Style,
    rng: np.random.Generator,
    *,
    low: float = 100.0,
    high: float = 9_000.0,
    decay_seconds: float = 0.09,
) -> np.ndarray:
    count = frames(seconds)
    signal = _band(
        rng.standard_normal(count),
        low,
        min(SR * 0.47, high * style.brightness),
    )
    envelope = np.exp(-np.arange(count) / SR / max(0.004, decay_seconds))
    return _fade(signal * envelope, 0.001, min(0.035, seconds * 0.25))


def sub_hit(
    seconds: float,
    style: Style,
    rng: np.random.Generator,
    *,
    frequency: float = 72.0,
    decay_seconds: float = 0.14,
) -> np.ndarray:
    count = frames(seconds)
    noise = rng.standard_normal(count)
    low = sosfilt(
        butter(3, 340.0, btype="lowpass", fs=SR, output="sos"), noise
    )
    b, a = iirpeak(max(24.0, frequency), 3.8, fs=SR)
    body = lfilter(b, a, low)
    envelope = np.exp(-np.arange(count) / SR / max(0.01, decay_seconds))
    return _fade(body * envelope * style.body, 0.001, 0.03)


def pitch_gesture(
    start_hz: float,
    end_hz: float,
    seconds: float,
    style: Style,
    *,
    decay_seconds: float | None = None,
    harmonics: tuple[tuple[float, float], ...] = ((1.0, 1.0),),
) -> np.ndarray:
    count = frames(seconds)
    position = np.linspace(0.0, 1.0, count, endpoint=False)
    frequencies = start_hz * np.power(end_hz / start_hz, position)
    phase = np.cumsum(frequencies / SR) * TAU
    output = np.zeros(count, dtype=np.float64)
    for ratio, gain in harmonics:
        output += np.sin(phase * ratio) * gain
    if decay_seconds is None:
        envelope = np.sin(math.pi * position) ** 1.2
    else:
        envelope = np.exp(-position * seconds / max(0.005, decay_seconds))
    return _fade(output * envelope, 0.001, min(0.08, seconds * 0.25))


def blade_ring(
    seconds: float,
    style: Style,
    rng: np.random.Generator,
    *,
    base_hz: float = 920.0,
) -> np.ndarray:
    count = frames(seconds)
    time = np.arange(count) / SR
    output = np.zeros(count, dtype=np.float64)
    detune = float(rng.uniform(0.985, 1.018))
    for ratio, gain, decay_time in (
        (1.0, 0.56, 0.23),
        (1.43, 0.29, 0.16),
        (2.17, 0.18, 0.11),
        (3.08, 0.10, 0.07),
    ):
        frequency = base_hz * ratio * detune * style.brightness
        output += np.sin(TAU * frequency * time) * gain * np.exp(
            -time / (decay_time * (0.80 + 0.45 * style.room))
        )
    return _fade(output, 0.0005, min(0.08, seconds * 0.25))


def alert_tone(
    notes: Sequence[float],
    seconds: float,
    style: Style,
) -> np.ndarray:
    output = canvas(seconds)
    gap = seconds / max(1, len(notes))
    for index, frequency in enumerate(notes):
        tone = pitch_gesture(
            frequency,
            frequency * 1.015,
            min(0.19, gap * 0.88),
            style,
            decay_seconds=0.10,
            harmonics=((1.0, 1.0), (2.0, 0.18)),
        )
        add_at(output, index * gap, tone, 0.52)
    return output


def recorded(
    pool: Sequence[foley.FoleySource],
    style: Style,
    rng: np.random.Generator,
    *,
    speed: float = 1.0,
    low_hz: float = 35.0,
    high_hz: float = 18_000.0,
    decay_seconds: float | None = None,
    maximum_seconds: float | None = None,
) -> tuple[np.ndarray, str]:
    """Render one recorded layer under the selected style treatment."""

    effective_high = min(SR * 0.49, high_hz * style.brightness)
    effective_low = min(low_hz, max(20.0, effective_high - 80.0))
    signal, source = foley.take(
        pool,
        rng,
        speed=speed * style.speed,
        low_hz=effective_low,
        high_hz=effective_high,
        decay_seconds=decay_seconds,
        maximum_seconds=maximum_seconds,
    )
    return signal * style.foley_gain, source


def master(signal: Sequence[float], peak_db: float, style: Style) -> list[float]:
    data = np.asarray(signal, dtype=np.float64).copy()
    if style.room > 0.0 and len(data):
        delay = max(1, round((0.038 + style.room * 0.10) * SR))
        if delay < len(data):
            data[delay:] += data[:-delay] * style.room
    return shared_master(data, peak_db, drive=style.drive)


__all__ = [
    "SR",
    "STYLES",
    "Style",
    "add_at",
    "alert_tone",
    "blade_ring",
    "canvas",
    "frames",
    "master",
    "noise_hit",
    "pitch_gesture",
    "recorded",
    "style_named",
    "sub_hit",
    "whoosh",
    "write_wav",
]
