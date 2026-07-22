"""Dependency-free primitives for reduced-resolution game-audio synthesis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from . import dsp


Waveform = Literal["sine", "square", "pulse", "triangle", "saw"]


def oscillator_track(
    frequencies: Sequence[float],
    waveform: Waveform = "square",
    phase: float = 0.0,
    *,
    duty: float = 0.32,
    sample_rate: int = dsp.SR,
) -> list[float]:
    """Render a phase-continuous oscillator from a per-frame frequency track."""

    if not 0.05 <= duty <= 0.95:
        raise ValueError("duty must be between 0.05 and 0.95")
    output: list[float] = []
    for raw_frequency in frequencies:
        cycle = (phase / dsp.TAU) % 1.0
        if waveform == "saw":
            value = 2.0 * cycle - 1.0
        elif waveform == "triangle":
            value = 1.0 - 4.0 * abs(cycle - 0.5)
        elif waveform in {"square", "pulse"}:
            value = 1.0 if cycle < duty else -1.0
        elif waveform == "sine":
            value = math.sin(phase)
        else:  # pragma: no cover - guarded for untyped callers
            raise ValueError(f"unsupported waveform: {waveform!r}")
        output.append(value)
        frequency = max(0.0, float(raw_frequency))
        phase = (phase + dsp.TAU * frequency / sample_rate) % dsp.TAU
    return output


def bit_reduce(signal: Sequence[float], bits: int) -> list[float]:
    """Quantise floating PCM to a signed bit depth while retaining floats."""

    if not 2 <= bits <= 24:
        raise ValueError("bits must be between 2 and 24")
    levels = float((1 << (bits - 1)) - 1)
    return [round(max(-1.0, min(1.0, float(value))) * levels) / levels
            for value in signal]


def sample_hold(signal: Sequence[float], factor: int) -> list[float]:
    """Hold every selected sample for ``factor`` frames without resizing."""

    if factor < 1:
        raise ValueError("sample-hold factor must be at least 1")
    source = list(signal)
    output = [0.0] * len(source)
    for start in range(0, len(source), factor):
        output[start:start + factor] = [source[start]] * min(
            factor, len(source) - start
        )
    return output


def reduce(signal: Sequence[float], *, bits: int, hold: int) -> list[float]:
    """Apply bit-depth and sample-rate reduction while preserving frame count."""

    return sample_hold(bit_reduce(signal, bits), hold)


def echo(
    signal: Sequence[float],
    delay_seconds: float,
    feedback: float,
    *,
    wet: float = 0.35,
    sample_rate: int = dsp.SR,
) -> list[float]:
    """Add a fixed-length recursive echo suitable for one-shot game cues."""

    if delay_seconds <= 0.0:
        raise ValueError("echo delay must be positive")
    if not 0.0 <= feedback < 1.0:
        raise ValueError("echo feedback must be in [0, 1)")
    if not 0.0 <= wet <= 1.0:
        raise ValueError("echo wet mix must be in [0, 1]")
    source = [float(value) for value in signal]
    output = list(source)
    delay = max(1, round(delay_seconds * sample_rate))
    for index in range(delay, len(output)):
        output[index] += output[index - delay] * feedback * wet
    return output


__all__ = [
    "Waveform",
    "bit_reduce",
    "echo",
    "oscillator_track",
    "reduce",
    "sample_hold",
]
