"""Seeded human grunts, gasps, exertions, and layered defeat Foley.

The voice itself is always a real CC0 performance. Retuning and filtering can
change body size and recording character, but this generator never tries to
synthesize a vocal tract. Composite game cues borrow the already-auditioned
unarmed body Foley and the procedural fantasy dissolve through their public cue
registries, so each layer still has exactly one implementation.
"""

from __future__ import annotations

from ..._audio.assets import asset_names

from ..._audio.assets import resolve_asset

from ..._audio.foley_numpy import read_recording

from ..._audio.foley_numpy import write_wav_scipy as write_wav  # noqa: F401

from ..._audio.foley_numpy import fade_edges, mix_at_frames, varispeed_configured


import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, resample, sosfilt

SR = 44_100
SOURCE_DIRECTORY = Path(__file__).resolve().parent / "sources"

MASCULINE = tuple(f"haeldb_male_grunt_{index:02d}.wav" for index in range(1, 7))
FEMININE = ("nocturnal_vanguard_female_hurt.wav",)


@dataclass(frozen=True)
class VoiceProfile:
    name: str
    description: str
    sources: tuple[str, ...]
    speed: float
    warmth: float
    brightness: float


VOICES: dict[str, VoiceProfile] = {
    "masculine": VoiceProfile(
        "masculine", "recorded adult male non-verbal performances", MASCULINE,
        0.98, 0.12, -0.02,
    ),
    "feminine": VoiceProfile(
        "feminine", "recorded adult female hurt and effort performances",
        FEMININE, 1.02, -0.04, 0.08,
    ),
    "neutral": VoiceProfile(
        "neutral", "the complete human pool without a gender preference",
        MASCULINE + FEMININE, 1.0, 0.02, 0.03,
    ),
}




def _active_regions(signal: np.ndarray) -> tuple[np.ndarray, ...]:
    """Split a source reel into isolated utterances and trim its room."""

    window = max(3, round(0.012 * SR))
    envelope = np.sqrt(uniform_filter1d(signal * signal, window))
    threshold = max(float(np.max(envelope)) * 0.035, 10 ** (-48.0 / 20.0))
    active = envelope >= threshold
    indices = np.flatnonzero(active)
    if not len(indices):
        return (signal.copy(),)

    split_gap = round(0.085 * SR)
    gaps = np.flatnonzero(np.diff(indices) > split_gap)
    starts = np.r_[indices[0], indices[gaps + 1]]
    ends = np.r_[indices[gaps], indices[-1]]
    margin = round(0.018 * SR)
    regions: list[np.ndarray] = []
    for first, last in zip(starts, ends):
        first = max(0, int(first) - margin)
        last = min(len(signal), int(last) + margin + 1)
        if last - first >= round(0.07 * SR):
            regions.append(signal[first:last].copy())
    return tuple(regions) or (signal.copy(),)


@lru_cache(maxsize=16)
def _load_source(filename: str) -> tuple[np.ndarray, ...]:
    path = resolve_asset(SOURCE_DIRECTORY / filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing bundled voice source {path}; see sources/provenance.json"
        )
    _, signal = read_recording(
        path, expected_rate=SR, normalize=False, immutable=False,
        label="voice source",
    )
    regions = _active_regions(signal)
    result: list[np.ndarray] = []
    for region in regions:
        peak = float(np.max(np.abs(region))) if len(region) else 0.0
        if peak > 1e-8:
            region = region / peak
            region.setflags(write=False)
            result.append(region)
    if not result:
        raise ValueError(f"voice source {path} contains no audible signal")
    return tuple(result)


def _choose(rng: np.random.Generator, choices: Sequence[str]) -> str:
    return choices[int(rng.integers(0, len(choices)))]


def _varispeed(signal: np.ndarray, speed: float) -> np.ndarray:
    return varispeed_configured(
        signal, speed, minimum=0.65, maximum=1.45, denominator=512
    )


def _fit(signal: np.ndarray, frames: int) -> np.ndarray:
    """Fit an utterance to a game gesture without padding a long silence."""

    if len(signal) > frames:
        signal = resample(signal, frames)
    result = np.zeros(frames)
    result[:min(frames, len(signal))] = signal[:frames]
    return result


def _fades(signal: np.ndarray, seconds: float=0.004) -> np.ndarray:
    return fade_edges(signal, seconds, sample_rate=SR)


def _prepare_voice(profile: VoiceProfile, rng: np.random.Generator,
                   frames: int, *, prefer_long: bool = False) -> tuple[np.ndarray, str]:
    filename = _choose(rng, profile.sources)
    regions = _load_source(filename)
    if prefer_long:
        candidates = sorted(regions, key=len, reverse=True)[:max(1, len(regions) // 2)]
        region = candidates[int(rng.integers(0, len(candidates)))]
    else:
        region = regions[int(rng.integers(0, len(regions)))]
    speed = profile.speed * float(rng.uniform(0.965, 1.036))
    signal = _varispeed(region, speed)
    signal = sosfilt(butter(2, 68.0, btype="highpass", fs=SR, output="sos"), signal)
    signal = sosfilt(butter(2, 10_500.0, btype="lowpass", fs=SR, output="sos"), signal)
    low = sosfilt(butter(2, 360.0, btype="lowpass", fs=SR, output="sos"), signal)
    high = signal - sosfilt(
        butter(2, 3_100.0, btype="lowpass", fs=SR, output="sos"), signal
    )
    signal = signal + profile.warmth * low + profile.brightness * high
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal /= peak
    return _fades(_fit(signal, frames)), filename


def _mix_at(destination: np.ndarray, source: np.ndarray, start: int, gain: float) -> None:
    mix_at_frames(destination, source, start, gain)


def _borrowed_body(rng: np.random.Generator, *, fall: bool,
                   frames: int) -> tuple[np.ndarray, list[str]]:
    from ..unarmed_foley_generator import cues as body_cues
    from ..unarmed_foley_generator import engine as body_engine

    seed = int(rng.integers(0, 2**32 - 1))
    weight = "brutal" if fall else "light"
    options = body_cues.Options(
        weight=body_engine.WEIGHTS[weight], intensity=0.62 if fall else 0.48
    )
    result = body_cues.render("hit", options, np.random.default_rng(seed))
    signal = np.asarray(result.samples)
    if fall:
        signal = resample(signal, min(frames, max(1, round(len(signal) * 1.28))))
        signal = sosfilt(
            butter(2, 680.0, btype="lowpass", fs=SR, output="sos"), signal
        )
    return _fades(_fit(signal, frames)), [
        f"unarmed_foley_generator/{source}" for source in result.sources
    ]


def pain(profile: VoiceProfile, rng: np.random.Generator,
         duration: float = 0.44, intensity: float = 0.9,
         *, body: bool = True) -> tuple[np.ndarray, list[str], list[str]]:
    """A brief human hurt reaction, optionally fused with body contact."""

    frames = round(duration * SR)
    voice_frames = min(frames, round(0.34 * SR))
    voice, voice_source = _prepare_voice(profile, rng, voice_frames)
    out = np.zeros(frames)
    _mix_at(out, voice, round(0.008 * SR), 0.86 * intensity)
    sources = [voice_source]
    layers = ["voice"]
    if body:
        hit, hit_sources = _borrowed_body(
            rng, fall=False, frames=min(frames, round(0.25 * SR))
        )
        _mix_at(out, hit, 0, 0.60 * intensity)
        sources.extend(hit_sources)
        layers.append("unarmed_foley_generator/hit")
    return _fades(out), sources, layers


def gasp(profile: VoiceProfile, rng: np.random.Generator,
         duration: float = 0.52, intensity: float = 0.72,
         **_: object) -> tuple[np.ndarray, list[str], list[str]]:
    frames = round(duration * SR)
    voice, source = _prepare_voice(profile, rng, frames, prefer_long=True)
    # A gasp should lose force after its onset; this also keeps a yell from
    # becoming a sustained scream when a longer source region is selected.
    envelope = np.exp(-np.arange(frames) / SR / max(0.12, duration * 0.48))
    return _fades(voice * envelope * intensity), [source], ["voice"]


def exertion(profile: VoiceProfile, rng: np.random.Generator,
             duration: float = 0.46, intensity: float = 0.82,
             **_: object) -> tuple[np.ndarray, list[str], list[str]]:
    frames = round(duration * SR)
    voice, source = _prepare_voice(profile, rng, frames)
    return _fades(voice * intensity), [source], ["voice"]


def death(profile: VoiceProfile, rng: np.random.Generator,
          duration: float = 1.28, intensity: float = 0.9,
          **_: object) -> tuple[np.ndarray, list[str], list[str]]:
    """Human gasp, soft body fall, and a borrowed magical dissolve."""

    from ..magic_energy_generator import cues as magic_cues

    frames = round(duration * SR)
    out = np.zeros(frames)
    voice, voice_sources, _ = gasp(
        profile, rng, duration=min(0.46, duration), intensity=0.78
    )
    _mix_at(out, voice, 0, 0.88 * intensity)

    fall_frames = min(round(0.48 * SR), frames)
    fall, fall_sources = _borrowed_body(rng, fall=True, frames=fall_frames)
    _mix_at(out, fall, round(0.46 * SR), 0.58 * intensity)

    magic_seed = int(rng.integers(0, 2**32 - 1))
    magic_duration = min(1.08, max(0.12, duration - 0.16))
    magic = magic_cues.render(
        "dissolve",
        magic_cues.Options(intensity=0.78, duration=magic_duration),
        random.Random(magic_seed),
    ).samples
    _mix_at(out, np.asarray(magic), round(0.16 * SR), 0.80 * intensity)
    return _fades(out), voice_sources + fall_sources, [
        "voice",
        "unarmed_foley_generator/hit-as-fall",
        "magic_energy_generator/dissolve",
    ]


def master(signal: np.ndarray, peak_db: float) -> list[float]:
    data = np.asarray(signal, dtype=float)
    if not len(data):
        return []
    drive = 1.15
    data = np.tanh(data * drive) / math.tanh(drive)
    data = _fades(data, 0.003)
    peak = float(np.max(np.abs(data)))
    if peak > 1e-12:
        data *= 10 ** (peak_db / 20.0) / peak
    return [float(value) for value in np.clip(data, -0.98, 0.98)]


def bundled_sources() -> tuple[str, ...]:
    return asset_names(SOURCE_DIRECTORY)


def voice_named(name: str) -> VoiceProfile:
    if name not in VOICES:
        raise ValueError(f"unknown voice {name!r}; expected one of {sorted(VOICES)}")
    return VOICES[name]
