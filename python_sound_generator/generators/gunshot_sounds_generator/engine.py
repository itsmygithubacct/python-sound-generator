"""Gunshot context and firearm Foley composed from existing generators.

``weapons_destruction_generator`` remains the only implementation of a firearm
report.  This module adds what happens around that pressure event: dedicated
CC0 firearm Foley, recorded material contacts, reload sequences, suppression
balance, and a space response. Full research masters remain outside the runtime
bank; only auditioned trims are promoted into ``sources/``.
"""

from __future__ import annotations

from ..._audio.assets import resolve_asset

from ..._audio.foley_numpy import write_wav

import math
import random
import wave
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from ..projectile_sounds import engine as material_engine
from ..stone_noise_generator import engine as stone_engine
from ..weapons_destruction_generator import engine as report_engine
from ..wood_noise_generator import engine as wood_engine


SR = 44_100
FIREARMS = report_engine.FIREARMS
FirearmProfile = report_engine.FirearmProfile


@dataclass(frozen=True)
class SpaceProfile:
    """A mono acoustic context for the borrowed dry report."""

    name: str
    label: str
    description: str
    direct_gain: float
    direct_cutoff: float
    tail_seconds: float
    tail_gain: float
    tail_cutoff: float
    decay: float
    early_delays: tuple[float, ...]
    reflection_count: int


@dataclass(frozen=True)
class SuppressionProfile:
    """Report/mechanism balance, not a volume switch."""

    name: str
    label: str
    description: str
    report_gain: float
    report_cutoff: float
    mechanism_gain: float
    tail_gain: float


@dataclass(frozen=True)
class SurfaceProfile:
    """The material a casing lands on."""

    name: str
    label: str
    description: str


SPACES: dict[str, SpaceProfile] = {
    "dry": SpaceProfile(
        "dry", "Dry", "close-miked report with almost no reflected field",
        1.0, 18_000.0, 0.10, 0.08, 9_000.0, 0.045, (0.008,), 0,
    ),
    "indoor": SpaceProfile(
        "indoor", "Indoor", "hard room: early wall slaps and a dense short tail",
        0.90, 14_000.0, 1.20, 0.78, 6_500.0, 0.34,
        (0.012, 0.019, 0.031, 0.047, 0.071), 34,
    ),
    "outdoor": SpaceProfile(
        "outdoor", "Outdoor", "open air with sparse terrain reflections",
        1.0, 16_000.0, 1.55, 0.46, 8_500.0, 0.58,
        (0.038, 0.083, 0.147, 0.245), 16,
    ),
    "distant": SpaceProfile(
        "distant", "Distant", "far report: air loss, weak direct crack, long return",
        0.28, 2_700.0, 2.40, 0.82, 2_100.0, 0.92,
        (0.090, 0.185, 0.330, 0.570), 28,
    ),
}


SUPPRESSIONS: dict[str, SuppressionProfile] = {
    "unsuppressed": SuppressionProfile(
        "unsuppressed", "Unsuppressed", "full muzzle pressure and crack",
        1.0, 18_000.0, 0.62, 1.0,
    ),
    "suppressed": SuppressionProfile(
        "suppressed", "Suppressed",
        "gas report reduced and darkened; action and casing move forward",
        0.34, 2_200.0, 1.20, 0.48,
    ),
}


SURFACES: dict[str, SurfaceProfile] = {
    "concrete": SurfaceProfile(
        "concrete", "Concrete", "hard mineral floor; stone contact underneath metal",
    ),
    "wood": SurfaceProfile(
        "wood", "Wood", "plank or timber floor with a short hollow answer",
    ),
    "metal": SurfaceProfile(
        "metal", "Metal", "plate or machinery deck; bright and ringing",
    ),
}


_BASE_METAL = material_engine.MATERIALS["metal"]
_SMALL_METAL = replace(
    _BASE_METAL,
    name="small_metal",
    contact=("kenney_impact_metal_light_", "kenney_impact_tin_medium_"),
    darken=13_500.0,
)
_MEDIUM_METAL = replace(
    _BASE_METAL,
    name="medium_metal",
    contact=("kenney_impact_metal_medium_", "rubberduck_metal_hit_"),
    darken=11_500.0,
)
_HEAVY_METAL = replace(
    _BASE_METAL,
    name="heavy_metal",
    contact=("kenney_impact_metal_heavy_", "kenney_impact_plate_medium_"),
    darken=9_000.0,
)


def _source(package: str, filename: str) -> str:
    return f"{package}/sources/{filename}"


@lru_cache(maxsize=None)
def _recording(filename: str) -> np.ndarray:
    path = resolve_asset(Path(__file__).resolve().parent / "sources" / filename)
    if not path.is_file():
        raise ValueError(f"missing firearm Foley source: {path}")
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (
            1, 2, SR
        ):
            raise ValueError(f"firearm Foley must be mono 44.1 kHz 16-bit PCM: {path}")
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return data.astype(np.float64) / 32768.0


def _foley(filename: str, intensity: float) -> tuple[np.ndarray, list[str]]:
    intensity = _check_intensity(intensity)
    signal = _recording(filename).copy() * (intensity / 0.85)
    return signal, [_source("gunshot_sounds_generator", filename)]


def _unique(names: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _check_intensity(intensity: float) -> float:
    if not 0.05 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    return intensity


def _place(out: np.ndarray, signal: Sequence[float], seconds: float,
           gain: float = 1.0) -> None:
    data = np.asarray(signal, dtype=np.float64)
    start = int(round(seconds * SR))
    if data.size == 0 or start >= out.size:
        return
    if start < 0:
        data = data[-start:]
        start = 0
    span = min(data.size, out.size - start)
    out[start:start + span] += data[:span] * gain


def _compose(parts: Sequence[tuple[Sequence[float], float, float]],
             tail: float = 0.04) -> np.ndarray:
    length = max(
        (int(round(start * SR)) + len(signal) for signal, start, _ in parts),
        default=1,
    ) + int(round(tail * SR))
    out = np.zeros(max(1, length))
    for signal, start, gain in parts:
        _place(out, signal, start, gain)
    return out


def _lowpass(signal: Sequence[float], cutoff: float, order: int = 4
             ) -> np.ndarray:
    data = np.asarray(signal, dtype=np.float64)
    if data.size == 0 or cutoff >= SR * 0.47:
        return data.copy()
    sos = butter(order, max(80.0, cutoff), btype="low", fs=SR, output="sos")
    return sosfilt(sos, data)


def _highpass(signal: Sequence[float], cutoff: float) -> np.ndarray:
    data = np.asarray(signal, dtype=np.float64)
    if data.size == 0:
        return data.copy()
    sos = butter(2, cutoff, btype="high", fs=SR, output="sos")
    return sosfilt(sos, data)


def _fade(signal: np.ndarray, out_seconds: float = 0.045) -> np.ndarray:
    data = np.asarray(signal, dtype=np.float64).copy()
    if data.size == 0:
        return data
    fade_in = min(data.size, max(2, int(0.0005 * SR)))
    fade_out = min(data.size, max(2, int(out_seconds * SR)))
    data[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
    data[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    return data


def _metal(rng: random.Random, size: str, weight: float,
           intensity: float = 0.85) -> tuple[np.ndarray, list[str]]:
    profile = {
        "small": _SMALL_METAL,
        "medium": _MEDIUM_METAL,
        "heavy": _HEAVY_METAL,
    }[size]
    signal, names = material_engine.material_contact(
        profile, rng, weight=weight, intensity=intensity
    )
    return signal, [_source("projectile_sounds", name) for name in names]


def _surface_contact(surface: SurfaceProfile, rng: random.Random,
                     intensity: float) -> tuple[np.ndarray, list[str]]:
    if surface.name == "concrete":
        signal, names = stone_engine.contact(
            stone_engine.STONES["granite"], stone_engine.SCALES["pebble"],
            np.random.default_rng(rng.randrange(2 ** 63)), intensity,
        )
        return signal, [_source("stone_noise_generator", name) for name in names]
    if surface.name == "wood":
        signal, names = wood_engine.strike(
            wood_engine.FORMS["plank"],
            np.random.default_rng(rng.randrange(2 ** 63)), intensity,
        )
        return signal, [_source("wood_noise_generator", name) for name in names]
    signal, names = _metal(rng, "medium", 0.45, intensity)
    return signal, names


def trigger(rng: random.Random, intensity: float = 0.85
            ) -> tuple[np.ndarray, list[str]]:
    intensity = _check_intensity(intensity)
    click, sources = _metal(rng, "small", 0.08, intensity)
    return _fade(_highpass(click[:int(0.10 * SR)], 1_100.0), 0.015), sources


def safety(rng: random.Random, intensity: float = 0.85
           ) -> tuple[np.ndarray, list[str]]:
    intensity = _check_intensity(intensity)
    first, sources = _metal(rng, "small", 0.10, intensity)
    second, more = _metal(rng, "small", 0.09, intensity * 0.55)
    out = _compose(((first[:int(0.09 * SR)], 0.0, 0.75),
                    (second[:int(0.08 * SR)], 0.018, 0.48)), 0.025)
    return _fade(_highpass(out, 850.0), 0.02), _unique(sources + more)


def bolt(rng: random.Random, intensity: float = 0.85,
         heavy: bool = False, recorded: bool = False
         ) -> tuple[np.ndarray, list[str]]:
    intensity = _check_intensity(intensity)
    if recorded and not heavy:
        return _foley("bolt_rifle.wav", intensity)
    size = "heavy" if heavy else "medium"
    back, sources = _metal(rng, size, 1.15 if heavy else 0.55, intensity)
    home, more = _metal(rng, size, 1.30 if heavy else 0.65, intensity)
    gap = 0.075 if heavy else 0.042
    out = _compose(((back, 0.0, 0.72), (home, gap, 1.0)), 0.05)
    return _fade(_lowpass(out, 11_000.0), 0.035), _unique(sources + more)


def casing_eject(rng: random.Random, intensity: float = 0.85
                 ) -> tuple[np.ndarray, list[str]]:
    intensity = _check_intensity(intensity)
    shell, sources = _metal(rng, "small", 0.07, intensity)
    shell = _highpass(shell[:int(0.16 * SR)], 1_500.0)
    return _fade(shell, 0.025), sources


def casing_land(surface: SurfaceProfile, rng: random.Random,
                intensity: float = 0.85) -> tuple[np.ndarray, list[str]]:
    """A casing hits a surface, bounces twice, then settles."""

    intensity = _check_intensity(intensity)
    first, sources = _metal(rng, "small", 0.07, intensity)
    under, surface_sources = _surface_contact(surface, rng, intensity * 0.30)
    second, more = _metal(rng, "small", 0.05, intensity * 0.55)
    third, last = _metal(rng, "small", 0.05, intensity * 0.30)
    first = _highpass(first, 1_100.0)
    second = _highpass(second, 1_350.0)
    third = _highpass(third, 1_500.0)
    out = _compose((
        (under[:int(0.22 * SR)], 0.0, 0.24),
        (first[:int(0.20 * SR)], 0.0, 1.0),
        (second[:int(0.15 * SR)], 0.082, 0.62),
        (third[:int(0.12 * SR)], 0.132, 0.38),
    ), 0.07)
    return _fade(out, 0.05), _unique(
        sources + surface_sources + more + last
    )


def magazine_out(rng: random.Random, intensity: float = 0.85
                 ) -> tuple[np.ndarray, list[str]]:
    return _foley("magazine_out_ar15.wav", intensity)


def magazine_in(rng: random.Random, intensity: float = 0.85
                ) -> tuple[np.ndarray, list[str]]:
    return _foley("magazine_in_ar15.wav", intensity)


def charging_handle(rng: random.Random, intensity: float = 0.85,
                    heavy: bool = False) -> tuple[np.ndarray, list[str]]:
    intensity = _check_intensity(intensity)
    if not heavy:
        return _foley("charging_handle_ar15.wav", intensity)
    pull, sources = _metal(rng, "heavy" if heavy else "medium",
                           1.1 if heavy else 0.50, intensity)
    release, more = _metal(rng, "heavy" if heavy else "medium",
                           1.25 if heavy else 0.62, intensity)
    gap = 0.18 if heavy else 0.115
    out = _compose(((pull, 0.0, 0.58), (release, gap, 1.0)), 0.08)
    return _fade(_lowpass(out, 9_000.0), 0.06), _unique(sources + more)


def dry_fire(rng: random.Random, intensity: float = 0.85
             ) -> tuple[np.ndarray, list[str]]:
    intensity = _check_intensity(intensity)
    press, sources = trigger(rng, intensity)
    striker, more = _metal(rng, "small", 0.16, intensity)
    out = _compose(((press, 0.0, 0.52),
                    (_lowpass(striker, 6_500.0), 0.012, 0.92)), 0.045)
    return _fade(out, 0.035), _unique(sources + more)


def _space_tail(report: Sequence[float], space: SpaceProfile,
                rng: random.Random) -> np.ndarray:
    """Convolve the report onset with a sparse, seeded mono room response."""

    if space.reflection_count == 0 and not space.early_delays:
        return np.zeros(max(1, int(space.tail_seconds * SR)))
    excitation = np.asarray(report, dtype=np.float64)[:int(0.12 * SR)].copy()
    if excitation.size:
        excitation *= np.exp(-np.arange(excitation.size) / SR / 0.032)
    ir = np.zeros(max(1, int(space.tail_seconds * SR)))
    for index, delay in enumerate(space.early_delays):
        sample = min(ir.size - 1, int(round(delay * SR)))
        decay = math.exp(-delay / max(space.decay, 1e-4))
        ir[sample] += (0.72 / (1.0 + 0.22 * index)) * decay * (
            -1.0 if index % 3 == 2 else 1.0
        )
    start = space.early_delays[0] if space.early_delays else 0.015
    for _ in range(space.reflection_count):
        delay = start + (rng.random() ** 0.72) * max(
            0.001, space.tail_seconds - start
        )
        sample = min(ir.size - 1, int(round(delay * SR)))
        decay = math.exp(-delay / max(space.decay, 1e-4))
        ir[sample] += rng.uniform(-0.32, 0.32) * decay
    ir = _lowpass(ir, space.tail_cutoff, 2)
    tail = fftconvolve(excitation, ir)[:ir.size]
    peak = float(np.max(np.abs(tail))) if tail.size else 0.0
    if peak > 1e-9:
        tail = tail / peak
    return _fade(tail, min(0.16, space.tail_seconds * 0.2))


def report_with_space(profile: FirearmProfile, space: SpaceProfile,
                      suppression: SuppressionProfile, rng: random.Random,
                      intensity: float = 0.85
                      ) -> tuple[np.ndarray, list[str]]:
    """Borrow the canonical report, then place it in a suppression/space mix."""

    intensity = _check_intensity(intensity)
    report = np.asarray(report_engine.fire_weapon(profile, rng, intensity))
    direct = _lowpass(report, min(space.direct_cutoff,
                                 suppression.report_cutoff))
    direct *= space.direct_gain * suppression.report_gain
    tail = _space_tail(report, space, rng)
    out = _compose(((direct, 0.0, 1.0),
                    (tail, 0.0, space.tail_gain * suppression.tail_gain)),
                   0.04)
    return _fade(out, min(0.12, space.tail_seconds * 0.15)), [
        "generator:weapons_destruction_generator/fire",
        f"generated-space:{space.name}",
        f"suppression-balance:{suppression.name}",
    ]


def tail(profile: FirearmProfile, space: SpaceProfile,
         suppression: SuppressionProfile, rng: random.Random,
         intensity: float = 0.85) -> tuple[np.ndarray, list[str]]:
    """The reflected field on its own, for placement and auditioning."""

    intensity = _check_intensity(intensity)
    report = np.asarray(report_engine.fire_weapon(profile, rng, intensity))
    report = _lowpass(report, suppression.report_cutoff)
    response = _space_tail(report, space, rng)
    response *= space.tail_gain * suppression.tail_gain
    return response, [
        "generator:weapons_destruction_generator/fire",
        f"generated-space:{space.name}",
        f"suppression-balance:{suppression.name}",
    ]


def reload(profile: FirearmProfile, rng: random.Random,
           intensity: float = 0.85) -> tuple[np.ndarray, list[str]]:
    """A weapon-correct reload sequence rather than one generic clack."""

    intensity = _check_intensity(intensity)
    parts: list[tuple[Sequence[float], float, float]] = []
    sources: list[str] = []

    def add(renderer, at: float, gain: float = 1.0, **kwargs) -> None:
        signal, used = renderer(rng, intensity=intensity, **kwargs)
        parts.append((signal, at, gain))
        sources.extend(used)

    if profile.name == "rifle":
        return _foley("reload_rifle_ar15.wav", intensity)
    if profile.name == "revolver":
        return _foley("reload_revolver_ruger_gp100.wav", intensity)
    elif profile.name == "shotgun":
        return _foley("reload_shotgun.wav", intensity)
    elif profile.category == "ordnance":
        return _foley("reload_artillery.wav", intensity)
    else:
        add(magazine_out, 0.0, 0.85)
        add(magazine_in, 0.58, 1.0)
        add(charging_handle, 1.02, 1.0)
    return _fade(_compose(parts, 0.15), 0.10), _unique(sources)


def shot(profile: FirearmProfile, space: SpaceProfile,
         suppression: SuppressionProfile, surface: SurfaceProfile,
         rng: random.Random, intensity: float = 0.85
         ) -> tuple[np.ndarray, list[str]]:
    """One complete gunshot: report, space, action, ejection, and landing."""

    intensity = _check_intensity(intensity)
    report, sources = report_with_space(
        profile, space, suppression, rng, intensity
    )
    parts: list[tuple[Sequence[float], float, float]] = [(report, 0.0, 1.0)]
    action_gain = suppression.mechanism_gain * (0.55 + profile.mechanical_gain)

    self_loading = profile.name in {
        "pistol", "smg", "rifle", "machine_gun", "autocannon"
    }
    if self_loading:
        action, used = bolt(rng, intensity, heavy=profile.name == "autocannon")
        parts.append((action, 0.032 + 0.018 * min(profile.scale, 1.5),
                      0.38 * action_gain))
        sources.extend(used)
        eject, used = casing_eject(rng, intensity)
        parts.append((eject, 0.046, 0.42 * action_gain))
        sources.extend(used)
        landed, used = casing_land(surface, rng, intensity)
        land_time = 0.20 + rng.uniform(0.04, 0.18)
        parts.append((landed, land_time, 0.34 * action_gain))
        sources.extend(used)
    elif profile.name in {"cannon", "artillery"}:
        action, used = bolt(rng, intensity, heavy=True)
        parts.append((action, 0.16, 0.28 * action_gain))
        sources.extend(used)
    return _fade(_compose(parts, 0.08), 0.08), _unique(sources)


def burst(profile: FirearmProfile, space: SpaceProfile,
          suppression: SuppressionProfile, surface: SurfaceProfile,
          rng: random.Random, shots: int = 5, intensity: float = 0.85
          ) -> tuple[np.ndarray, list[str]]:
    """Complete shots at the canonical firearm cadence."""

    if not 2 <= shots <= 12:
        raise ValueError("shots must be between 2 and 12")
    intensity = _check_intensity(intensity)
    interval = 1.0 / profile.rounds_per_second
    parts: list[tuple[Sequence[float], float, float]] = []
    sources: list[str] = []
    for index in range(shots):
        signal, used = shot(profile, space, suppression, surface, rng, intensity)
        jitter = 0.0 if index == 0 else rng.uniform(-0.012, 0.012) * interval
        parts.append((signal, max(0.0, index * interval + jitter),
                      rng.uniform(0.94, 1.04)))
        sources.extend(used)
    return _fade(_compose(parts, 0.08), 0.10), _unique(sources)


def firearm_named(name: str) -> FirearmProfile:
    return report_engine.firearm_named(name)


def space_named(name: str) -> SpaceProfile:
    try:
        return SPACES[name]
    except KeyError:
        raise ValueError(
            f"unknown space {name!r}; expected one of {sorted(SPACES)}"
        ) from None


def suppression_named(name: str) -> SuppressionProfile:
    try:
        return SUPPRESSIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown suppression {name!r}; expected one of "
            f"{sorted(SUPPRESSIONS)}"
        ) from None


def surface_named(name: str) -> SurfaceProfile:
    try:
        return SURFACES[name]
    except KeyError:
        raise ValueError(
            f"unknown surface {name!r}; expected one of {sorted(SURFACES)}"
        ) from None


def master(signal: Sequence[float], peak_db: float) -> list[float]:
    """Soft-limit, edge-fade, and master one event to a declared peak."""

    data = np.asarray(signal, dtype=np.float64)
    if data.size == 0:
        return []
    limited = np.tanh(data * 1.18) / math.tanh(1.18)
    limited = _fade(limited, 0.06)
    peak = float(np.max(np.abs(limited)))
    if peak > 1e-12:
        limited *= 10.0 ** (peak_db / 20.0) / peak
    return np.clip(limited, -0.98, 0.98).tolist()




__all__ = [
    "FIREARMS", "SPACES", "SUPPRESSIONS", "SURFACES", "SR",
    "FirearmProfile", "SpaceProfile", "SuppressionProfile", "SurfaceProfile",
    "bolt", "burst", "casing_eject", "casing_land", "charging_handle",
    "dry_fire", "firearm_named", "magazine_in", "magazine_out", "master",
    "reload", "report_with_space", "safety", "shot", "space_named",
    "suppression_named", "surface_named", "tail", "trigger", "write_wav",
]
