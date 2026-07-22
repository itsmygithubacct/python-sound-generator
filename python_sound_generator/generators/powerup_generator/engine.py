"""Core DSP for powerup and item-acquisition sounds. NumPy + SciPy.

Pickup sounds are mostly *musical* gestures rather than physical events: a coin
is a rising interval, a level-up is an arpeggio, a denial is a semitone falling.
So this module is built around pitch and timbre — band-limited oscillators,
modal bars, plucked strings — rather than the impact/material model the combat
and footstep generators need.

Everything is synthesised. No samples, no recordings.

Two notes on the oscillators, which is where naive pickup synths fall over:

- They are **band-limited by additive synthesis**: a square is a sum of odd
  harmonics, not a clipped sine. A naive square at C6 folds every harmonic above
  22 kHz back down as inharmonic mush, which is exactly the frequency range these
  sounds live in. Harmonics above Nyquist are masked out per sample, so a glide
  stays clean as it rises.
- Frequency is accepted **per sample**, so a glide accumulates phase properly
  (`cumsum`) instead of being restarted each block, which would click.
"""

from __future__ import annotations

from ..._audio.foley_numpy import write_wav as write_wav

import math

import numpy as np
from scipy.signal import butter, sosfilt

SR = 44_100
TAU = 2.0 * math.pi

# Additive oscillators sum this many partials at most. At the low end of the
# range these sounds use (~A3), Nyquist allows ~100; past ~64 the extra partials
# are inaudible and just cost time.
MAX_PARTIALS = 64


# --------------------------------------------------------------------------
# pitch
# --------------------------------------------------------------------------

def midi_to_freq(note: float) -> float:
    """MIDI note number -> Hz. A4 (69) = 440."""
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


# --------------------------------------------------------------------------
# buffers
# --------------------------------------------------------------------------

def samples(duration: float) -> int:
    return max(1, int(round(duration * SR)))


def timebase(duration: float) -> np.ndarray:
    return np.arange(samples(duration)) / SR


def silence(duration: float) -> np.ndarray:
    return np.zeros(samples(duration))


def noise(duration: float, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(samples(duration))


def add_at(dst: np.ndarray, start: float, src: np.ndarray,
           gain: float = 1.0) -> None:
    """Mix src into dst in place, starting at `start` seconds. Clips to dst."""
    i = int(round(start * SR))
    if i >= len(dst):
        return
    if i < 0:
        src = src[-i:]
        i = 0
    n = min(len(src), len(dst) - i)
    if n > 0:
        dst[i:i + n] += src[:n] * gain


def mix(*parts: np.ndarray) -> np.ndarray:
    n = max(len(p) for p in parts)
    out = np.zeros(n)
    for part in parts:
        out[:len(part)] += part
    return out


def _freqs(freq, n: int) -> np.ndarray:
    """Accept a scalar, a per-sample array, or a short control curve."""
    arr = np.atleast_1d(np.asarray(freq, dtype=float))
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size == n:
        return arr
    return np.interp(np.linspace(0.0, 1.0, n),
                     np.linspace(0.0, 1.0, arr.size), arr)


# --------------------------------------------------------------------------
# oscillators (band-limited)
# --------------------------------------------------------------------------

def osc(kind: str, freq, duration: float, phase0: float = 0.0) -> np.ndarray:
    """Band-limited oscillator. `freq` may vary per sample (glides are free).

    Partials whose frequency would exceed Nyquist are masked out sample by
    sample, so a rising glide sheds harmonics as it climbs instead of aliasing.
    """
    n = samples(duration)
    f = _freqs(freq, n)
    phase = phase0 + TAU * np.cumsum(f) / SR
    if kind == "sine":
        return np.sin(phase)

    nyquist = SR * 0.5
    out = np.zeros(n)
    if kind == "square":
        for i, k in enumerate(range(1, MAX_PARTIALS * 2, 2)):
            audible = f * k < nyquist
            if not audible.any():
                break
            out += np.where(audible, np.sin(k * phase) / k, 0.0)
        out *= 4.0 / math.pi
    elif kind == "triangle":
        for i, k in enumerate(range(1, MAX_PARTIALS * 2, 2)):
            audible = f * k < nyquist
            if not audible.any():
                break
            sign = -1.0 if i % 2 else 1.0
            out += np.where(audible, sign * np.sin(k * phase) / (k * k), 0.0)
        out *= 8.0 / (math.pi ** 2)
    elif kind == "saw":
        for k in range(1, MAX_PARTIALS + 1):
            audible = f * k < nyquist
            if not audible.any():
                break
            out += np.where(audible, np.sin(k * phase) / k, 0.0)
        out *= 2.0 / math.pi
    else:
        raise ValueError(f"unknown oscillator {kind!r}")
    return out


# --------------------------------------------------------------------------
# envelopes
# --------------------------------------------------------------------------

def exp_env(duration: float, decay: float, attack: float = 0.0) -> np.ndarray:
    """Exponential decay, optionally with a raised-cosine attack."""
    t = timebase(duration)
    env = np.exp(-t / max(decay, 1e-4))
    if attack > 0.0:
        a = min(len(env), samples(attack))
        if a > 1:
            env[:a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / (a - 1))
    return env


def adsr(duration: float, attack: float, decay: float, sustain: float,
         release: float) -> np.ndarray:
    """Linear ADSR clipped to `duration`."""
    n = samples(duration)
    env = np.zeros(n)
    a = min(n, samples(attack)) if attack > 0 else 0
    d = min(n - a, samples(decay)) if decay > 0 else 0
    r = min(n, samples(release)) if release > 0 else 0
    s_len = max(0, n - a - d - r)
    idx = 0
    if a:
        env[:a] = np.linspace(0.0, 1.0, a)
        idx = a
    if d:
        env[idx:idx + d] = np.linspace(1.0, sustain, d)
        idx += d
    if s_len:
        env[idx:idx + s_len] = sustain
        idx += s_len
    if r and idx < n:
        env[idx:] = np.linspace(sustain, 0.0, n - idx)
    return env


def bell_env(duration: float, peak: float, rise: float,
             fall: float) -> np.ndarray:
    """Gaussian swell peaking at `peak` seconds, asymmetric either side."""
    t = timebase(duration)
    width = np.where(t < peak, rise, fall)
    return np.exp(-((t - peak) / width) ** 2)


def ramp(duration: float, start: float, end: float,
         curve: float = 1.0) -> np.ndarray:
    """start -> end over duration. curve>1 eases in, curve<1 eases out."""
    x = np.linspace(0.0, 1.0, samples(duration)) ** curve
    return start + (end - start) * x


# --------------------------------------------------------------------------
# voices
# --------------------------------------------------------------------------

def modal_ring(duration: float, modes, rng: np.random.Generator) -> np.ndarray:
    """Sum of exponentially decaying sinusoids: (freq, decay, gain) each.

    Struck bars and bells are inharmonic, so callers pass ratios that are
    deliberately not integer multiples.
    """
    t = timebase(duration)
    out = np.zeros(len(t))
    for freq, decay, gain in modes:
        if freq >= SR * 0.48:
            continue
        phase = rng.uniform(0.0, TAU)
        out += gain * np.sin(TAU * freq * t + phase) * np.exp(-t / max(decay, 1e-4))
    return out


def pluck(duration: float, freq: float, rng: np.random.Generator,
          damping: float = 0.5, brightness: float = 1.0) -> np.ndarray:
    """Karplus-Strong plucked string.

    `damping` in (0,1] sets how fast the high end dies: 1.0 is a bright harp,
    lower is a duller, faster-decaying pluck. `brightness` lowpasses the initial
    excitation burst, which is what separates a harp from a rubber band.
    """
    n = samples(duration)
    period = max(2, int(round(SR / max(freq, 20.0))))
    buf = rng.uniform(-1.0, 1.0, period)
    if brightness < 1.0:
        # A one-pole lowpass on the excitation only; the loop filter below is
        # what shapes the sustain.
        coef = float(np.clip(brightness, 0.02, 1.0))
        for i in range(1, period):
            buf[i] = buf[i - 1] + coef * (buf[i] - buf[i - 1])
    out = np.zeros(n)
    idx = 0
    loop = float(np.clip(damping, 0.05, 1.0)) * 0.5
    for i in range(n):
        current = buf[idx]
        out[i] = current
        nxt = buf[(idx + 1) % period]
        buf[idx] = loop * (current + nxt)
        idx = (idx + 1) % period
    return out


def sparkle(duration: float, rng: np.random.Generator, count: int = 24,
            low: float = 3_000.0, high: float = 12_000.0,
            decay: float = 0.030, skew: float = 0.7) -> np.ndarray:
    """Scattered high grains — the glitter on top of a pickup.

    Grain onsets are power-curved toward the start (`skew`) so the shimmer
    clusters with the hit and thins into a tail rather than sounding uniform.
    """
    n = samples(duration)
    out = np.zeros(n)
    high = max(high, low + 10.0)
    for _ in range(max(0, count)):
        start = int((rng.random() ** skew) * n)
        if start >= n - 2:
            continue
        freq = low * (high / low) ** rng.random()
        if freq >= SR * 0.48:
            continue
        grain_decay = float(rng.uniform(decay * 0.5, decay * 1.5))
        length = min(n - start, max(2, int(grain_decay * SR * 4)))
        t = np.arange(length) / SR
        out[start:start + length] += (
            np.sin(TAU * freq * t + rng.uniform(0.0, TAU))
            * np.exp(-t / grain_decay)
            * rng.uniform(0.4, 1.0)
        )
    return out


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------

def _safe_fc(cutoff: float) -> float:
    return float(np.clip(cutoff, 20.0, SR * 0.49))


def lowpass(signal: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    sos = butter(order, _safe_fc(cutoff), btype="lowpass", fs=SR, output="sos")
    return sosfilt(sos, signal)


def highpass(signal: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    sos = butter(order, _safe_fc(cutoff), btype="highpass", fs=SR, output="sos")
    return sosfilt(sos, signal)


def bandpass(signal: np.ndarray, low: float, high: float,
             order: int = 4) -> np.ndarray:
    lo = _safe_fc(low)
    hi = _safe_fc(high)
    if hi <= lo * 1.02:
        hi = min(lo * 1.02, SR * 0.49)
        if hi <= lo:
            return lowpass(signal, hi, order)
    sos = butter(order, [lo, hi], btype="bandpass", fs=SR, output="sos")
    return sosfilt(sos, signal)


def resonant_sweep(signal: np.ndarray, freqs, q: float = 2.0,
                   mode: str = "bandpass") -> np.ndarray:
    """Filter whose cutoff moves per sample (topology-preserving SVF).

    scipy's sosfilt cannot retune while running, and a biquad re-tuned per block
    zipper-noises. This is the same TPT structure the combat generator uses for
    its whoosh, kept here so a shield sweep can glide its cutoff cleanly.
    """
    n = len(signal)
    f = _freqs(freqs, n)
    out = np.zeros(n)
    k = 1.0 / max(q, 0.05)
    ic1 = ic2 = 0.0
    nyq_limit = SR * 0.49
    for i in range(n):
        fc = min(max(float(f[i]), 20.0), nyq_limit)
        g = math.tan(math.pi * fc / SR)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = signal[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        out[i] = v1 if mode == "bandpass" else v2
    return out


def dc_block(signal: np.ndarray, cutoff: float = 12.0) -> np.ndarray:
    """Remove subsonic energy without touching the body of the sound."""
    pole = math.exp(-TAU * cutoff / SR)
    out = np.zeros(len(signal))
    prev_in = prev_out = 0.0
    for i, x in enumerate(signal):
        y = x - prev_in + pole * prev_out
        out[i] = y
        prev_in = x
        prev_out = y
    return out


# --------------------------------------------------------------------------
# levels
# --------------------------------------------------------------------------

def amp_db(db: float) -> float:
    return 10.0 ** (db / 20.0)


def normalize(signal: np.ndarray, peak_db: float) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 1e-12:
        signal = signal * (amp_db(peak_db) / peak)
    return np.clip(signal, -0.98, 0.98)


def soft_limit(signal: np.ndarray, drive: float = 1.25) -> np.ndarray:
    return np.tanh(signal * drive) / math.tanh(drive)


def fade_edges(signal: np.ndarray, fade_in_ms: float = 1.0,
               fade_out_ms: float = 20.0) -> np.ndarray:
    out = np.array(signal, dtype=float, copy=True)
    n = len(out)
    fi = min(n, int(SR * fade_in_ms / 1000.0))
    fo = min(n, int(SR * fade_out_ms / 1000.0))
    if fi > 1:
        out[:fi] *= np.linspace(0.0, 1.0, fi)
    if fo > 1:
        out[n - fo:] *= np.linspace(1.0, 0.0, fo)
    return out


def master(signal: np.ndarray, peak_db: float, drive: float = 1.25,
           fade_in_ms: float = 1.0, fade_out_ms: float = 25.0) -> np.ndarray:
    """Final chain shared by every cue: DC-block, limit, fade, normalize."""
    return normalize(
        fade_edges(soft_limit(dc_block(signal), drive), fade_in_ms, fade_out_ms),
        peak_db,
    )


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
