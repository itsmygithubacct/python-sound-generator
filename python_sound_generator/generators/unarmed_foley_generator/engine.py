"""Unarmed impacts built from real CC0 recordings.

A hybrid generator in the same lineage as ``door_sound_generator`` and
``creature_foley_generator``: NumPy and SciPy trim, retune, filter, and layer
bundled recordings rather than synthesizing an impact from oscillators.

Why this exists at all
----------------------
``combat_generators`` already models unarmed combat properly -- a fist is a
striker, a torso is a target, and the whole weapon-by-target matrix applies. Its
*swings* are excellent, because a swing is air and cloth, and air and cloth are
noise, which synthesis is genuinely good at. Its blade-on-blade work is better
than a recording would be.

Its flesh impacts are not, and the reason is structural rather than a tuning
miss. The low end of a struck body there comes from a decaying sine, and a sine
held for a few cycles at 60 Hz is a tuned drum: the cue lands on a *note*. You
can bury the pitch by trading the sine away for filtered noise -- and that helps
-- but what is left is still noise shaped like an impact rather than an impact.
A real punch is a chaotic, non-stationary event: cloth, skin, fat, and bone all
arriving a few milliseconds apart with their own decays. That micro-detail is
what the ear identifies, and it is why Foley artists hit bags and slabs of meat
instead of reaching for a synthesizer.

So this generator owns the impacts and nothing else. ``combat_generators`` stays
stdlib-only and keeps everything it is already better at; the pack registry here
imports it for the swings, so one command still fills a game's whole bank.

The recordings and their licensing live in ``sources/provenance.json``.

Everything returns unmastered mono floating-point PCM plus the source files it
consumed. The cue registry owns mastering levels, so every public render follows
the same contract, and the consumed filenames reach the manifest.
"""

from __future__ import annotations

from ..._audio.assets import asset_names

from ..._audio.assets import resolve_asset

from ..._audio.foley_numpy import read_recording

from ..._audio.foley_numpy import write_wav_scipy as write_wav  # noqa: F401

from ..._audio.foley_numpy import trim_activity as trim_recording_activity

from ..._audio.foley_numpy import fade_edges, mix_at_frames, varispeed_configured


import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfilt

SR = 44_100
SOURCE_DIRECTORY = Path(__file__).resolve().parent / "sources"

# Every bundled recording, by the name provenance.json records.
HIT_SOURCES: tuple[str, ...] = (
    "independent_nu_hit_02.wav",
    "independent_nu_hit_07.wav",
    "independent_nu_hit_08.wav",
    "independent_nu_hit_20.wav",
    "independent_nu_hit_29.wav",
    "independent_nu_hit_34.wav",
)


@dataclass(frozen=True)
class Weight:
    """How much body is behind a blow, and what that does to the recording.

    `speed` is varispeed and does the real work: resampling a recording moves
    its pitch and its decays together, which is what separates a jab from a
    hook. Pitch-shifting alone would leave the decays where they were and turn
    a heavy blow into a slowed-down light one.
    """

    name: str
    label: str
    description: str
    speed: float
    weight: float
    brightness: float


WEIGHTS: dict[str, Weight] = {
    "light": Weight(
        "light", "Light", "a jab: fast, little mass behind it", 1.18, 0.30, 0.16
    ),
    "medium": Weight(
        "medium", "Medium", "a straight punch, most of the shoulder in it",
        1.00, 0.60, 0.06,
    ),
    "heavy": Weight(
        "heavy", "Heavy", "a hook or a kick, the hips behind it", 0.86, 1.00,
        -0.05,
    ),
    "brutal": Weight(
        "brutal", "Brutal", "the one that ends it", 0.74, 1.35, -0.12
    ),
}




def _trim_to_onset(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    return trim_recording_activity(
            signal, sample_rate=sample_rate, window_seconds=0.004,
            threshold_ratio=0.06, pre_seconds=0.004, post_seconds=0.010,
        )


@lru_cache(maxsize=64)
def _load_source(filename: str) -> np.ndarray:
    path = resolve_asset(SOURCE_DIRECTORY / filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing bundled impact source {path}; see sources/provenance.json"
        )
    _, signal = read_recording(
        path, expected_rate=SR,
        trim=lambda data, rate: _trim_to_onset(data, rate),
        label="impact source",
    )
    return signal


def _varispeed(signal: np.ndarray, speed: float) -> np.ndarray:
    return varispeed_configured(
        signal, speed, minimum=0.25, maximum=4.0, denominator=512
    )


def _tilt(signal: np.ndarray, weight: float, brightness: float) -> np.ndarray:
    """Add mass and top end using bands of the recording itself.

    Both terms are lifted out of the source rather than synthesized, so a
    heavier blow stays the same event rather than becoming that event with a
    sine bolted underneath -- which is the failure this generator exists to
    avoid.
    """
    result = signal
    if abs(weight) > 1e-6:
        low = sosfilt(butter(2, 180.0, btype="lowpass", fs=SR, output="sos"),
                      signal)
        result = result + weight * low
    if abs(brightness) > 1e-6:
        high = signal - sosfilt(
            butter(2, 2_200.0, btype="lowpass", fs=SR, output="sos"), signal
        )
        result = result + brightness * high
    return result


def _fades(signal: np.ndarray, seconds: float=0.003) -> np.ndarray:
    return fade_edges(signal, seconds, sample_rate=SR)


def _decay(signal: np.ndarray, seconds: float) -> np.ndarray:
    """Shorten a recording's ring without shortening its attack."""

    if seconds <= 0.0 or len(signal) == 0:
        return signal
    t = np.arange(len(signal)) / SR
    return signal * np.exp(-t / seconds)


def _choose(rng: np.random.Generator, choices: Sequence[str]) -> str:
    return choices[int(rng.integers(0, len(choices)))]


def _mix_at(destination: np.ndarray, source: np.ndarray, start_frame: int, gain: float) -> None:
    mix_at_frames(destination, source, start_frame, gain)


def _prepare(filename: str, weight: Weight, rng: np.random.Generator,
             speed_multiplier: float = 1.0) -> np.ndarray:
    """Load one recording, retuned and tilted for this blow."""

    cached = _load_source(filename)
    speed = weight.speed * speed_multiplier * float(rng.uniform(0.955, 1.048))
    signal = _varispeed(cached, speed)
    signal = _tilt(signal, weight.weight, weight.brightness)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal = signal / peak
    return _fades(signal)


def _frames(seconds: float) -> int:
    return max(1, round(seconds * SR))


# --------------------------------------------------------------------------
# cues
# --------------------------------------------------------------------------

def hit(weight: Weight, rng: np.random.Generator,
        intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """A fist landing on a body.

    Two recordings, not one: the second is laid a few milliseconds behind the
    first, quieter and retuned away from it. A single sample replayed is
    recognisably the same hit every time, and two staggered ones read as one
    thicker impact while giving the rotation somewhere to vary. The offset stays
    under ~12 ms -- past that the ear stops fusing them and hears a double tap.
    """
    primary = _choose(rng, HIT_SOURCES)
    secondary = _choose(rng, tuple(s for s in HIT_SOURCES if s != primary))

    lead = _prepare(primary, weight, rng)
    back = _prepare(secondary, weight, rng, speed_multiplier=0.90)

    total = max(len(lead), len(back) + _frames(0.012)) + _frames(0.02)
    out = np.zeros(total)
    _mix_at(out, lead, 0, 1.0)
    _mix_at(out, back, _frames(float(rng.uniform(0.004, 0.012))),
            0.42 * intensity)
    return _fades(out, 0.004), [primary, secondary]


def block(weight: Weight, rng: np.random.Generator,
          intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """A blow caught on a braced forearm.

    The same recordings as `hit`, made sharp by filtering rather than by mixing
    in a slappier source: a guard is bone close under the skin, so it keeps the
    crack and loses the give. It is the decay that carries this, not the tone --
    a braced arm returns nothing, so the event has to stop dead where a body
    would still be moving.
    """
    primary = _choose(rng, HIT_SOURCES)
    signal = _prepare(primary, weight, rng, speed_multiplier=1.22)
    signal = sosfilt(butter(2, 140.0, btype="highpass", fs=SR, output="sos"),
                     signal)
    signal = _tilt(signal, 0.0, 0.34)
    signal = _decay(signal, 0.075)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal = signal / peak
    return _fades(signal * intensity, 0.004), [primary]


def knockout(weight: Weight, rng: np.random.Generator,
             intensity: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """The hit that ends it, and the body after it.

    Three events, and the spacing is the cue: the impact, a sub-bass drop under
    it, then the fall arriving late enough to read as a consequence rather than
    part of the hit.

    The drop is the one invented thing here. No impact makes a sweep like that;
    it is the convention a fighting game uses to say the lights went out, and
    the cue does not read without it. The fall is the same low band of the
    recordings, slowed hard -- a body landing is a body landing, so it does not
    need a different source, only a heavier one.
    """
    impact, sources = hit(weight, rng, intensity)

    fall_source = _choose(rng, HIT_SOURCES)
    fall = _prepare(fall_source, WEIGHTS["brutal"], rng, speed_multiplier=0.62)
    fall = sosfilt(butter(2, 420.0, btype="lowpass", fs=SR, output="sos"), fall)
    sources.append(fall_source)

    # The fall waits for the impact to finish rather than landing on a fixed
    # offset. `hit` returns whatever length its two recordings came to -- 0.37
    # to 0.58 s -- so a fixed offset drops the body on top of the punch for the
    # long ones and the three events mush into a smear. Keyed to the impact it
    # follows, the gap is always there and the rhythm varies per seed, which is
    # the good kind of variation. The floor is for a hit short enough that the
    # fall would tread on the drop.
    fall_at = max(_frames(0.30), len(impact))
    total = fall_at + len(fall) + _frames(0.05)
    out = np.zeros(total)
    _mix_at(out, impact, 0, 1.0)

    drop_frames = _frames(0.55)
    t = np.arange(drop_frames) / SR
    sweep = np.sin(2.0 * math.pi * np.cumsum(
        np.linspace(92.0, 34.0, drop_frames) / SR))
    drop = sweep * np.exp(-t / 0.20)
    _mix_at(out, drop, _frames(0.004), 0.42)
    _mix_at(out, fall, fall_at, 0.78)
    return _fades(out, 0.004), sources


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def master(signal: np.ndarray, peak_db: float) -> list[float]:
    """Final chain shared by every cue: soft limit, fade, normalize."""

    data = np.asarray(signal, dtype=float)
    if len(data) == 0:
        return []
    drive = 1.25
    data = np.tanh(data * drive) / math.tanh(drive)
    data = _fades(data, 0.003)
    peak = float(np.max(np.abs(data)))
    if peak > 1e-12:
        data = data * (10.0 ** (peak_db / 20.0) / peak)
    return [float(x) for x in np.clip(data, -0.98, 0.98)]


def weight_named(name: str) -> Weight:
    if name not in WEIGHTS:
        raise ValueError(
            f"unknown weight {name!r}; expected one of {sorted(WEIGHTS)}"
        )
    return WEIGHTS[name]


def bundled_sources() -> tuple[str, ...]:
    return asset_names(SOURCE_DIRECTORY)
