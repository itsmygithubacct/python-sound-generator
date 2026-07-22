"""Shared NumPy/SciPy primitives for recording-driven one-shot Foley.

The recording-backed generators used to carry private copies of the same four
operations: varispeed, low-pass darkening, tail control, and sample placement.
Keeping those copies in sync was item 10's most concrete maintenance problem.
This module is deliberately small: it owns transformations, never material or
event policy.  Material generators still decide what a recording means and
event generators still decide when its layers arrive.
"""

from __future__ import annotations

import math
import random
import wave
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, iirpeak, lfilter, resample_poly, sosfilt

from .assets import resolve_asset


SR = 44_100


def pcm_to_float(
    pcm: np.ndarray,
    *,
    mixdown: bool = True,
) -> np.ndarray:
    """Convert integer or floating WAV PCM to float64 in ``[-1, 1]``.

    Stereo input is averaged to mono by default.  Set ``mixdown=False`` for
    renderers that intentionally preserve the source channel layout.
    """

    data = np.asarray(pcm)
    if data.dtype.kind == "f":
        result = data.astype(np.float64)
    elif data.dtype == np.uint8:
        result = (data.astype(np.float64) - 128.0) / 128.0
    elif data.dtype.kind in "iu":
        info = np.iinfo(data.dtype)
        result = data.astype(np.float64) / max(abs(info.min), info.max)
    else:
        raise ValueError(f"unsupported WAV dtype: {data.dtype}")
    if result.ndim == 2 and mixdown:
        result = result.mean(axis=1)
    expected_dimensions = (1,) if mixdown else (1, 2)
    if result.ndim not in expected_dimensions:
        raise ValueError("source WAV must contain mono or interleaved PCM")
    return result


def read_recording(
    path: str | Path,
    *,
    expected_rate: int | None = SR,
    center: bool = True,
    normalize: bool = True,
    trim: Callable[[np.ndarray, int], np.ndarray] | None = None,
    immutable: bool = True,
    label: str = "source",
) -> tuple[int, np.ndarray]:
    """Read mono PCM with explicit rate, trim, normalization, and cache policy."""

    source = resolve_asset(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing bundled {label} {source}")
    rate, pcm = wavfile.read(source)
    rate = int(rate)
    if expected_rate is not None and rate != expected_rate:
        raise ValueError(
            f"source {source.name} is {rate} Hz; expected {expected_rate}"
        )
    signal = pcm_to_float(np.asarray(pcm))
    if center and len(signal):
        signal = signal - float(np.mean(signal))
    if trim is not None:
        signal = trim(signal, rate)
    if normalize:
        peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
        if peak < 1e-8:
            raise ValueError(f"{label} {source} contains no audible signal")
        signal = signal / peak
    signal = np.asarray(signal, dtype=np.float64)
    if immutable:
        signal.setflags(write=False)
    return rate, signal


@lru_cache(maxsize=256)
def load_wav(path: str | Path, *, trim_threshold: float = 0.02) -> np.ndarray:
    """Load, resample, peak-normalise, and trim a recording to its onset.

    The returned array is cached and must be treated as immutable.  All shared
    shaping helpers return a new array when they alter it.
    """

    if not math.isfinite(trim_threshold) or not 0.0 <= trim_threshold <= 1.0:
        raise ValueError("trim threshold must be between 0 and 1")

    source = resolve_asset(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing bundled Foley source {source}")
    rate, pcm = wavfile.read(source)
    data = pcm_to_float(np.asarray(pcm))
    if int(rate) != SR:
        data = resample_poly(data, SR, int(rate))
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Foley source {source} contains non-finite samples")
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 1e-9:
        raise ValueError(f"Foley source {source} contains no audible signal")
    data = data / peak
    loud = np.flatnonzero(np.abs(data) > trim_threshold)
    if loud.size:
        data = data[max(0, int(loud[0]) - 8):]
    result = np.asarray(data, dtype=np.float64)
    result.setflags(write=False)
    return result


def varispeed(data: np.ndarray, factor: float) -> np.ndarray:
    """Move pitch and duration together, which is how recorded mass scales."""

    signal = np.asarray(data, dtype=np.float64)
    if signal.size == 0 or abs(factor - 1.0) < 1e-3:
        return signal
    ratio = Fraction(1.0 / max(0.05, factor)).limit_denominator(64)
    return resample_poly(signal, ratio.numerator, ratio.denominator)


def varispeed_configured(
    data: np.ndarray,
    factor: float,
    *,
    minimum: float = 0.25,
    maximum: float = 4.0,
    denominator: int = 512,
    copy_identity: bool = True,
) -> np.ndarray:
    """Varispeed with explicit domain policy for clamps and approximation."""

    if not math.isfinite(factor):
        raise ValueError("varispeed factor must be finite")
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum varispeed must be positive and finite")
    if not math.isfinite(maximum) or maximum < minimum:
        raise ValueError("maximum varispeed must be finite and at least minimum")
    if denominator < 1:
        raise ValueError("varispeed denominator must be at least 1")

    signal = np.asarray(data, dtype=np.float64)
    speed = float(min(max(factor, minimum), maximum))
    ratio = Fraction(1.0 / speed).limit_denominator(denominator)
    if ratio.numerator == ratio.denominator:
        return signal.copy() if copy_identity else signal
    return resample_poly(signal, ratio.numerator, ratio.denominator)


def fade_edges(
    data: np.ndarray,
    seconds: float = 0.004,
    *,
    sample_rate: int = SR,
) -> np.ndarray:
    """Apply symmetric linear edge fades to a copied NumPy signal."""

    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("fade duration must be finite and non-negative")
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    signal = np.asarray(data, dtype=float).copy()
    fade = min(len(signal) // 2, max(1, round(seconds * sample_rate)))
    if fade > 1:
        signal[:fade] *= np.linspace(0.0, 1.0, fade)
        signal[-fade:] *= np.linspace(1.0, 0.0, fade)
    return signal


def trim_activity(
    data: np.ndarray,
    *,
    sample_rate: int = SR,
    window_seconds: float | None = 0.004,
    threshold_ratio: float = 0.05,
    threshold_floor_db: float | None = None,
    pre_seconds: float = 0.004,
    post_seconds: float | None = 0.010,
) -> np.ndarray:
    """Trim recording silence using configurable onset/activity policy.

    ``window_seconds=None`` selects an absolute-sample envelope.  A ``None``
    post margin trims only the lead and preserves the natural tail.
    """

    signal = np.asarray(data)
    if len(signal) < 4:
        return signal.copy()
    if window_seconds is None:
        envelope = np.abs(signal)
    else:
        window = max(3, round(window_seconds * sample_rate))
        envelope = np.sqrt(uniform_filter1d(signal * signal, window))
    peak = float(np.max(envelope))
    if peak <= 1e-9:
        return signal.copy()
    threshold = peak * threshold_ratio
    if threshold_floor_db is not None:
        threshold = max(threshold, 10.0 ** (threshold_floor_db / 20.0))
    active = np.flatnonzero(envelope >= threshold)
    if len(active) == 0:
        return signal.copy()
    first = max(0, int(active[0]) - round(pre_seconds * sample_rate))
    if post_seconds is None:
        return signal[first:].copy()
    last = min(
        len(signal),
        int(active[-1]) + round(post_seconds * sample_rate) + 1,
    )
    return signal[first:last].copy()


def mix_at_frames(
    destination: np.ndarray,
    source: np.ndarray,
    start_frame: int,
    gain: float,
) -> None:
    """Mix a NumPy source in place at a signed frame offset."""

    source_first = max(0, -start_frame)
    destination_first = max(0, start_frame)
    available = min(
        len(destination) - destination_first,
        len(source) - source_first,
    )
    if available > 0:
        destination[destination_first:destination_first + available] += (
            source[source_first:source_first + available] * gain
        )


def mix_wrapped_frames(
    destination: np.ndarray,
    source: np.ndarray,
    start_frame: int,
    gain: float,
) -> None:
    """Mix a source into a periodic NumPy destination, wrapping its tail."""

    if len(destination) == 0 or len(source) == 0:
        return
    total = len(destination)
    start = start_frame % total
    src = np.asarray(source)
    if len(src) > total:
        folded = np.zeros(total, dtype=np.result_type(src, np.float64))
        for offset in range(0, len(src), total):
            piece = src[offset:offset + total]
            folded[:len(piece)] += piece
        src = folded
    first = min(len(src), total - start)
    destination[start:start + first] += src[:first] * gain
    if first < len(src):
        rest = len(src) - first
        destination[:rest] += src[first:first + rest] * gain


def tile_to_frames(data: np.ndarray, frames: int) -> np.ndarray:
    """Repeat a signal to an exact frame count, or copy-truncate it."""

    signal = np.asarray(data)
    if frames < 0:
        raise ValueError("frame count cannot be negative")
    if frames == 0:
        return signal[:0].copy()
    if len(signal) == 0:
        raise ValueError("cannot tile an empty signal")
    if len(signal) >= frames:
        return signal[:frames].copy()
    repeats = int(np.ceil(frames / len(signal)))
    return np.tile(signal, repeats)[:frames]


def equal_power_crossfade_loop(
    data: np.ndarray,
    frames: int,
    fade_seconds: float,
    *,
    sample_rate: int = SR,
) -> np.ndarray:
    """Fold material after a loop boundary over its head at equal power."""

    if frames <= 0:
        raise ValueError("loop frame count must be positive")
    if not math.isfinite(fade_seconds) or fade_seconds <= 0.0:
        raise ValueError("crossfade duration must be positive and finite")
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    fade_frames = min(frames, max(1, round(fade_seconds * sample_rate)))
    signal = np.asarray(data)
    if len(signal) < frames + fade_frames:
        signal = tile_to_frames(signal, frames + fade_frames)
    body = signal[:frames].copy()
    tail = signal[frames:frames + fade_frames]
    ramp = np.linspace(0.0, 1.0, fade_frames)
    body[:fade_frames] = (
        body[:fade_frames] * np.sqrt(ramp)
        + tail * np.sqrt(1.0 - ramp)
    )
    return body


def random_window(
    data: np.ndarray,
    rng: np.random.Generator,
    seconds: float,
    *,
    sample_rate: int = SR,
) -> np.ndarray:
    """Take a deterministic random window, tiling short material first."""

    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("window duration must be positive and finite")
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    signal = np.asarray(data)
    if len(signal) == 0:
        raise ValueError("cannot window an empty signal")
    frames = max(1, round(seconds * sample_rate))
    if len(signal) <= frames:
        return tile_to_frames(signal, frames)
    offset = int(rng.integers(0, len(signal) - frames + 1))
    return signal[offset:offset + frames].copy()


def darken(data: np.ndarray, cutoff: float) -> np.ndarray:
    """Low-pass a recording without introducing a separate synthetic layer."""

    signal = np.asarray(data, dtype=np.float64)
    if signal.size == 0 or cutoff >= SR * 0.45:
        return signal
    sos = butter(4, max(200.0, cutoff), btype="low", fs=SR, output="sos")
    return sosfilt(sos, signal)


def decay(data: np.ndarray, seconds: float, hold: float = 0.0) -> np.ndarray:
    """Constrain a source tail with an exponential envelope."""

    signal = np.asarray(data, dtype=np.float64)
    if signal.size == 0:
        return signal
    t = np.arange(signal.size) / SR
    return signal * np.exp(-np.maximum(0.0, t - hold) / max(seconds, 1e-4))


def place(destination: np.ndarray, start: float, data: np.ndarray,
          gain: float = 1.0) -> None:
    """Mix ``data`` into ``destination`` at a time in seconds."""

    index = int(round(start * SR))
    source = np.asarray(data, dtype=np.float64)
    if source.size == 0 or index >= destination.size:
        return
    if index < 0:
        source = source[-index:]
        index = 0
    span = min(source.size, destination.size - index)
    destination[index:index + span] += source[:span] * gain


def resonant_sub(rng: random.Random, frequency: float, seconds: float,
                 decay_seconds: float, *, q: float = 4.5,
                 cutoff: float = 400.0) -> np.ndarray:
    """Noise through a low resonance: weight without a pitched sine boom."""

    frames = max(1, int(seconds * SR))
    noise = np.fromiter(
        (rng.gauss(0.0, 1.0) for _ in range(frames)),
        dtype=np.float64,
        count=frames,
    )
    low = sosfilt(
        butter(4, min(cutoff, SR * 0.45), btype="low", fs=SR, output="sos"),
        noise,
    )
    t = np.arange(frames) / SR
    drive = low * np.exp(-t / max(decay_seconds, 1e-4))
    b, a = iirpeak(max(18.0, frequency), q, fs=SR)
    return lfilter(b, a, drive)


def master(signal: np.ndarray, peak_db: float,
           drive: float = 1.25) -> list[float]:
    """Soft-limit, edge-fade, peak-normalise, and return plain float PCM."""

    data = np.asarray(signal, dtype=np.float64)
    if data.size == 0:
        return []
    limited = np.tanh(data * drive) / math.tanh(drive)
    fade_in = min(limited.size, int(SR * 0.0005))
    fade_out = min(limited.size, int(SR * 0.060))
    if fade_in > 1:
        limited[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
    if fade_out > 1:
        limited[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    peak = float(np.max(np.abs(limited)))
    if peak > 1e-12:
        limited *= 10.0 ** (peak_db / 20.0) / peak
    return np.clip(limited, -0.98, 0.98).tolist()


def write_wav(path: str | Path, signal: Sequence[float]) -> Path:
    """Write mono 44.1 kHz signed 16-bit PCM."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0)
    if not np.all(np.isfinite(frames)):
        raise ValueError("audio contains non-finite samples")
    pcm = (frames * 32767.0).round().astype("<i2")
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())
    return target


def write_wav_scipy(path: str | Path, signal: Sequence[float]) -> Path:
    """Write the legacy SciPy PCM16 container used by recording generators."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(list(signal), dtype=float)
    pcm = np.round(np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(target, SR, pcm)
    return target


__all__ = [
    "SR",
    "darken",
    "decay",
    "equal_power_crossfade_loop",
    "fade_edges",
    "load_wav",
    "master",
    "mix_at_frames",
    "mix_wrapped_frames",
    "pcm_to_float",
    "place",
    "resonant_sub",
    "random_window",
    "read_recording",
    "tile_to_frames",
    "trim_activity",
    "varispeed",
    "varispeed_configured",
    "write_wav",
    "write_wav_scipy",
]
