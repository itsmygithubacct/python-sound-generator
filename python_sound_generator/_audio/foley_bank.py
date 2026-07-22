"""Shared logical-source selection and shaping for optional Foley layers.

Providers own the meaning of their material pools and the placement of each
layer.  This module owns the repeated mechanics: logical paths, deterministic
selection, varispeed, filtering, tail control, and source reporting.

Procedural providers import this module lazily when their Foley mix is above
zero, preserving their dependency-free default import and render path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
from scipy.signal import butter, sosfilt

from .assets import resolve_asset
from .foley_numpy import SR, decay, fade_edges, load_wav, varispeed_configured


PROVIDER_ROOT = Path(__file__).resolve().parents[1] / "generators"


class RandomSource(Protocol):
    """The small random interface accepted from stdlib and NumPy callers."""

    def uniform(self, low: float, high: float) -> float: ...


@dataclass(frozen=True, slots=True)
class FoleySource:
    """One recording addressed through its provenance-owning provider."""

    provider: str
    filename: str

    def __post_init__(self) -> None:
        provider = Path(self.provider)
        filename = Path(self.filename)
        if (
            not self.provider
            or provider.is_absolute()
            or len(provider.parts) != 1
            or provider.name in {".", ".."}
            or "\\" in self.provider
        ):
            raise ValueError("Foley provider must be one package-directory name")
        if (
            not self.filename
            or filename.is_absolute()
            or ".." in filename.parts
            or filename.name in {"", "."}
            or "\\" in self.filename
            or filename.suffix.lower() != ".wav"
        ):
            raise ValueError("Foley filename must be a safe relative WAV path")

    @property
    def logical_name(self) -> str:
        return f"{self.provider}/sources/{self.filename}"

    @property
    def path(self) -> Path:
        return PROVIDER_ROOT / self.provider / "sources" / self.filename


def sources(provider: str, *filenames: str) -> tuple[FoleySource, ...]:
    """Build a stable logical source group owned by one provider."""

    return tuple(FoleySource(provider, filename) for filename in filenames)


def inventory(
    pools: Mapping[str, Sequence[FoleySource]],
) -> tuple[FoleySource, ...]:
    """Return every distinct source reachable through a pool mapping."""

    by_name = {
        source.logical_name: source
        for pool in pools.values()
        for source in pool
    }
    return tuple(by_name[name] for name in sorted(by_name))


def unresolved(source_inventory: Sequence[FoleySource]) -> tuple[str, ...]:
    """Return logical names whose local or catalogued bytes cannot resolve."""

    return tuple(
        source.logical_name
        for source in source_inventory
        if not resolve_asset(source.path).is_file()
    )


def _index(rng: RandomSource, stop: int) -> int:
    integers = getattr(rng, "integers", None)
    if callable(integers):
        return int(integers(0, stop))
    randrange = getattr(rng, "randrange", None)
    if callable(randrange):
        return int(randrange(stop))
    raise TypeError("Foley RNG must provide integers() or randrange()")


def choose(
    pool: Sequence[FoleySource],
    rng: RandomSource,
) -> FoleySource:
    """Choose one source deterministically with either supported RNG family."""

    if not pool:
        raise ValueError("Foley source pool cannot be empty")
    return pool[_index(rng, len(pool))]


def take(
    pool: Sequence[FoleySource],
    rng: RandomSource,
    *,
    speed: float = 1.0,
    speed_jitter: float = 0.045,
    low_hz: float = 35.0,
    high_hz: float = 18_000.0,
    decay_seconds: float | None = None,
    maximum_seconds: float | None = None,
) -> tuple[np.ndarray, str]:
    """Select, varispeed, filter, crop, and normalize one recording."""

    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("Foley speed must be positive and finite")
    if not math.isfinite(speed_jitter) or not 0.0 <= speed_jitter < 1.0:
        raise ValueError("Foley speed jitter must be finite and below 1")
    if (
        not math.isfinite(low_hz)
        or not math.isfinite(high_hz)
        or not 0.0 <= low_hz < high_hz <= SR * 0.5
    ):
        raise ValueError("Foley filter band must be inside the Nyquist range")
    if maximum_seconds is not None and (
        not math.isfinite(maximum_seconds) or maximum_seconds <= 0.0
    ):
        raise ValueError("maximum Foley duration must be positive and finite")
    if decay_seconds is not None and (
        not math.isfinite(decay_seconds) or decay_seconds <= 0.0
    ):
        raise ValueError("Foley decay must be positive and finite")

    source = choose(pool, rng)
    signal = load_wav(resolve_asset(source.path))
    jitter = float(rng.uniform(1.0 - speed_jitter, 1.0 + speed_jitter))
    if not math.isfinite(jitter) or jitter <= 0.0:
        raise ValueError("Foley RNG returned an invalid speed jitter")
    signal = varispeed_configured(
        signal,
        speed * jitter,
        minimum=0.25,
        maximum=4.0,
        denominator=512,
    )
    if low_hz > 20.0:
        signal = sosfilt(
            butter(3, low_hz, btype="highpass", fs=SR, output="sos"),
            signal,
        )
    if high_hz < SR * 0.48:
        signal = sosfilt(
            butter(3, high_hz, btype="lowpass", fs=SR, output="sos"),
            signal,
        )
    if maximum_seconds is not None:
        signal = signal[:max(1, round(maximum_seconds * SR))].copy()
    if decay_seconds is not None:
        signal = decay(signal, decay_seconds)

    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal = signal / peak
    return fade_edges(signal, 0.003, sample_rate=SR), source.logical_name


__all__ = [
    "PROVIDER_ROOT",
    "FoleySource",
    "RandomSource",
    "choose",
    "inventory",
    "sources",
    "take",
    "unresolved",
]
