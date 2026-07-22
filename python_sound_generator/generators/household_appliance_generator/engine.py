"""Recording-driven household appliance sounds.

The motor, fan, drum, water, and mechanism textures in this module are real
CC0 field recordings.  Code only selects a steady window, changes playback
speed, applies conservative appliance-specific filtering, and closes loop
seams.  The shared :mod:`foley_core` owns WAV loading, varispeed, mastering,
and output so this generator follows the same transform path as the other
recording-backed packages in this repository.
"""

from __future__ import annotations

from ..._audio.assets import asset_names

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from ..._audio.foley_numpy import SR, load_wav, varispeed, write_wav


SOURCES = Path(__file__).resolve().parent / "sources"
CROSSFADE_SECONDS = 0.45


@dataclass(frozen=True)
class Appliance:
    """One recorded appliance and the small transform range it tolerates."""

    name: str
    label: str
    description: str
    sources: tuple[str, ...]
    loop: bool
    seconds: float
    low_hz: float
    high_hz: float
    peak_db: float


APPLIANCES: dict[str, Appliance] = {
    "air_conditioner": Appliance(
        "air_conditioner", "Air conditioner",
        "compressor and circulating fan running steadily",
        ("bret_air_conditioner.wav",), True, 6.0, 35.0, 12_000.0, -14.0,
    ),
    "air_fryer": Appliance(
        "air_fryer", "Air fryer",
        "compact convection fan and heated chamber",
        ("bret_air_fryer.wav",), True, 5.0, 55.0, 13_500.0, -13.0,
    ),
    "can_opener": Appliance(
        "can_opener", "Electric can opener",
        "one complete motor-and-gear cutting pass",
        ("bret_can_opener.wav",), False, 3.2, 70.0, 11_000.0, -8.0,
    ),
    "microwave": Appliance(
        "microwave", "Microwave oven",
        "magnetron transformer, turntable, and cooling fan",
        ("bret_microwave_01.wav", "bret_microwave_02.wav"),
        True, 5.0, 45.0, 11_500.0, -14.0,
    ),
    "range_hood": Appliance(
        "range_hood", "Kitchen range hood",
        "stove extraction fan running steadily",
        ("bret_range_hood.wav",), True, 5.0, 45.0, 13_000.0, -15.0,
    ),
    "vacuum": Appliance(
        "vacuum", "Vacuum cleaner",
        "suction turbine, airflow, and floor contact",
        ("bret_vacuum.wav",), True, 5.0, 50.0, 14_500.0, -11.5,
    ),
    "washer": Appliance(
        "washer", "Washing machine",
        "drum, water, motor, and cabinet movement during a wash cycle",
        ("bret_washer.wav",), True, 7.0, 30.0, 11_000.0, -14.0,
    ),
}


def appliance_named(name: str) -> Appliance:
    try:
        return APPLIANCES[name]
    except KeyError:
        raise ValueError(
            f"unknown appliance {name!r}; expected one of {sorted(APPLIANCES)}"
        ) from None


def _shape(signal: np.ndarray, appliance: Appliance) -> np.ndarray:
    """Remove only energy outside the useful band of the real recording."""

    data = np.asarray(signal, dtype=np.float64)
    if data.size == 0:
        return data
    low = max(15.0, appliance.low_hz)
    high = min(SR * 0.47, appliance.high_hz)
    if low >= high:
        raise ValueError(f"invalid Foley band for {appliance.name}")
    sos = butter(3, (low, high), btype="bandpass", fs=SR, output="sos")
    return sosfilt(sos, data)


def _tile(signal: np.ndarray, frames: int) -> np.ndarray:
    """Extend recorded material with equal-power folds, not hard repeats."""

    if signal.size == 0:
        raise ValueError("Foley source contains no audio")
    if len(signal) >= frames:
        return signal[:frames].copy()
    source = np.asarray(signal, dtype=np.float64)
    fade = min(round(CROSSFADE_SECONDS * SR), len(source) // 3)
    if fade < 2:
        return np.tile(source, math.ceil(frames / len(source)))[:frames]
    built = source.copy()
    phase = np.linspace(0.0, math.pi * 0.5, fade)
    while len(built) < frames:
        join = (built[-fade:] * np.cos(phase)
                + source[:fade] * np.sin(phase))
        built = np.concatenate((built[:-fade], join, source[fade:]))
    return built[:frames]


def _candidate_window(signal: np.ndarray, frames: int,
                      rng: random.Random, tries: int = 7) -> np.ndarray:
    """Choose a seeded active window instead of landing in room tone."""

    if len(signal) <= frames:
        return _tile(signal, frames)
    best: tuple[float, int] | None = None
    for _ in range(tries):
        start = rng.randrange(0, len(signal) - frames + 1)
        chunk = signal[start:start + frames]
        # RMS rewards a working appliance; peak/RMS rejects an isolated bump.
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        peak = float(np.max(np.abs(chunk)))
        score = rms / max(1.0, peak / max(rms, 1e-9) / 7.0)
        if best is None or score > best[0]:
            best = (score, start)
    assert best is not None
    return signal[best[1]:best[1] + frames].copy()


def _crossfade_loop(signal: np.ndarray, frames: int) -> np.ndarray:
    """Fold a recording's continuation over its head with equal power."""

    fade = min(round(CROSSFADE_SECONDS * SR), frames // 3)
    if fade < 2:
        raise ValueError("loop is too short to crossfade")
    source = _tile(signal, frames + fade)
    body = source[:frames].copy()
    tail = source[frames:frames + fade]
    phase = np.linspace(0.0, math.pi * 0.5, fade)
    body[:fade] = body[:fade] * np.sin(phase) + tail * np.cos(phase)
    return body


def _master(signal: np.ndarray, peak_db: float, intensity: float,
            loop: bool) -> list[float]:
    """Memoryless loop-safe mastering; one-shots additionally fade at edges."""

    data = np.asarray(signal, dtype=np.float64).copy()
    if data.size == 0:
        return []
    drive = 1.2
    data = np.tanh(data * drive) / math.tanh(drive)
    if not loop:
        fade_in = min(len(data) // 2, round(0.008 * SR))
        fade_out = min(len(data) // 2, round(0.060 * SR))
        if fade_in > 1:
            data[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
        if fade_out > 1:
            data[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    peak = float(np.max(np.abs(data)))
    if peak > 1e-12:
        target = 10.0 ** (peak_db / 20.0)
        # Intensity changes level without making a quiet setting inaudible.
        data *= target * (0.35 + 0.65 * intensity) / peak
    return np.clip(data, -0.98, 0.98).tolist()


def render_appliance(appliance: Appliance, seconds: float, speed: float,
                     intensity: float, rng: random.Random
                     ) -> tuple[list[float], tuple[str, ...]]:
    """Render one appliance entirely from its bundled Foley recording."""

    source_name = appliance.sources[rng.randrange(len(appliance.sources))]
    data = load_wav(SOURCES / source_name, trim_threshold=0.004)
    data = varispeed(data, speed)
    data = _shape(data, appliance)
    frames = max(1, round(seconds * SR))
    if appliance.loop:
        fade = min(round(CROSSFADE_SECONDS * SR), frames // 3)
        window = _candidate_window(data, frames + max(2, fade), rng)
        signal = _crossfade_loop(window, frames)
    else:
        signal = _candidate_window(data, frames, rng)
    return _master(signal, appliance.peak_db, intensity, appliance.loop), (
        source_name,
    )


def bundled_sources() -> tuple[str, ...]:
    return asset_names(SOURCES)


__all__ = [
    "APPLIANCES", "CROSSFADE_SECONDS", "SR", "SOURCES", "Appliance",
    "appliance_named", "bundled_sources", "render_appliance", "write_wav",
]
