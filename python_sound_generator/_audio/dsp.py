"""Core DSP primitives. Standard library only.

No numpy, no scipy, no sample library: everything downstream is built from the
oscillators, noise and filters in here, so the generators run against a bare
CPython with nothing installed.

Function names and behaviour deliberately mirror the retired NumPy predecessor
so the two read the same way. Two things could not be ported directly and were
rebuilt:

- scipy's butter/sosfilt -> a cascade of RBJ biquads, one section per
  Butterworth pole pair (see _cascade). tests/test_dsp.py checks it against
  scipy: the two agree to ~8.5e-16 in linear magnitude, i.e. they are the same
  filter. That test compares linear magnitude and not dB on purpose -- deep in
  the stopband the response is numerically zero, and dB turns float dust there
  into huge apparent errors.
- a fixed bandpass -> svf_bandpass, a topology-preserving state variable
  filter that can retune while running, which a biquad cannot do cleanly.
"""

from __future__ import annotations

import array
import math
import random
import sys
import wave
from pathlib import Path

SR = 44100
TAU = 2.0 * math.pi


# --------------------------------------------------------------------------
# buffers
# --------------------------------------------------------------------------

def samples(duration: float) -> int:
    return max(1, int(round(duration * SR)))


def timebase(duration: float) -> list[float]:
    return [i / SR for i in range(samples(duration))]


def silence(duration: float) -> list[float]:
    return [0.0] * samples(duration)


def noise(duration: float, rng: random.Random) -> list[float]:
    return [rng.gauss(0.0, 1.0) for _ in range(samples(duration))]


def scale(signal: list[float], gain: float) -> list[float]:
    return [s * gain for s in signal]


def mul(a: list[float], b: list[float]) -> list[float]:
    return [x * y for x, y in zip(a, b)]


def mix(*parts: list[float]) -> list[float]:
    n = max(len(p) for p in parts)
    out = [0.0] * n
    for part in parts:
        for i, s in enumerate(part):
            out[i] += s
    return out


def add_at(dst: list[float], start: float, src: list[float],
           gain: float = 1.0) -> None:
    """Mix src into dst in place, starting at `start` seconds."""
    i = int(round(start * SR))
    if i >= len(dst):
        return
    if i < 0:
        src = src[-i:]
        i = 0
    n = min(len(src), len(dst) - i)
    for k in range(n):
        dst[i + k] += src[k] * gain


# --------------------------------------------------------------------------
# levels
# --------------------------------------------------------------------------

def amp_db(db: float) -> float:
    return 10.0 ** (db / 20.0)


def normalize(signal: list[float], peak_db: float) -> list[float]:
    peak = max((abs(s) for s in signal), default=0.0)
    if peak > 1e-12:
        gain = amp_db(peak_db) / peak
        signal = [s * gain for s in signal]
    return [max(-0.98, min(0.98, s)) for s in signal]


def soft_limit(signal: list[float], drive: float = 1.25) -> list[float]:
    norm = math.tanh(drive)
    return [math.tanh(s * drive) / norm for s in signal]


def fade_edges(signal: list[float], fade_in_ms: float = 3.0,
               fade_out_ms: float = 12.0) -> list[float]:
    out = list(signal)
    fi = min(len(out), int(SR * fade_in_ms / 1000.0))
    fo = min(len(out), int(SR * fade_out_ms / 1000.0))
    for i in range(fi):
        out[i] *= i / (fi - 1) if fi > 1 else 1.0
    for i in range(fo):
        out[len(out) - fo + i] *= 1.0 - (i / (fo - 1)) if fo > 1 else 1.0
    return out


def master(signal: list[float], peak_db: float, drive: float = 1.25,
           fade_in_ms: float = 0.5, fade_out_ms: float = 60.0) -> list[float]:
    """Final chain shared by every cue: limit, fade, normalize."""
    return normalize(fade_edges(soft_limit(signal, drive),
                                fade_in_ms, fade_out_ms), peak_db)


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------

def _butter_qs(order: int) -> list[float]:
    """Q per section so the cascade's poles land on the Butterworth circle."""
    return [1.0 / (2.0 * math.cos(math.pi * (2 * k + 1) / (2 * order)))
            for k in range(order // 2)]


def _biquad_coeffs(kind: str, fc: float, q: float) -> tuple[float, ...]:
    w0 = TAU * fc / SR
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * q)
    if kind == "lowpass":
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
    else:
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
    b2 = b0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _biquad(signal: list[float], b0: float, b1: float, b2: float,
            a1: float, a2: float) -> list[float]:
    out = [0.0] * len(signal)
    x1 = x2 = y1 = y2 = 0.0
    for i, x in enumerate(signal):
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y
        x2, x1 = x1, x
        y2, y1 = y1, y
    return out


def _cascade(signal: list[float], kind: str, cutoff: float,
             order: int) -> list[float]:
    nyq = SR * 0.5
    fc = max(20.0, min(float(cutoff), nyq * 0.98))
    order = max(2, (order // 2) * 2)
    out = signal
    for q in _butter_qs(order):
        out = _biquad(out, *_biquad_coeffs(kind, fc, q))
    return out


def lowpass(signal: list[float], cutoff: float, order: int = 4) -> list[float]:
    return _cascade(signal, "lowpass", cutoff, order)


def highpass(signal: list[float], cutoff: float, order: int = 4) -> list[float]:
    return _cascade(signal, "highpass", cutoff, order)


def bandpass(signal: list[float], low: float, high: float,
             order: int = 4) -> list[float]:
    return highpass(lowpass(signal, high, order), low, order)


def resonator(signal: list[float], freq: float, q: float = 8.0,
              gain: float = 1.0) -> list[float]:
    """Single ringing peak. Cheap way to give noise a pitch."""
    w0 = TAU * max(20.0, min(freq, SR * 0.49)) / SR
    alpha = math.sin(w0) / (2.0 * q)
    b0, b1, b2 = alpha, 0.0, -alpha
    a0, a1, a2 = 1.0 + alpha, -2.0 * math.cos(w0), 1.0 - alpha
    out = _biquad(signal, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    return scale(out, gain)


def svf_bandpass(signal: list[float], freqs: list[float],
                 q: float = 1.0) -> list[float]:
    """Bandpass whose centre moves per sample (topology-preserving SVF).

    A whoosh is mostly this: the blade brightens as it approaches and dulls as
    it recedes, so the filter has to sweep with the envelope.
    """
    out = [0.0] * len(signal)
    k = 1.0 / q
    ic1 = ic2 = 0.0
    nyq_limit = SR * 0.49
    for i, x in enumerate(signal):
        fc = max(20.0, min(freqs[i], nyq_limit))
        g = math.tan(math.pi * fc / SR)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        out[i] = v1
    return out


# --------------------------------------------------------------------------
# envelopes
# --------------------------------------------------------------------------

def exp_env(duration: float, decay: float, attack: float = 0.0) -> list[float]:
    env = [math.exp(-t / max(decay, 1e-4)) for t in timebase(duration)]
    if attack > 0.0:
        a = min(len(env), samples(attack))
        if a > 1:
            for i in range(a):
                env[i] *= 0.5 - 0.5 * math.cos(math.pi * i / (a - 1))
    return env


def bell_env(duration: float, peak: float, rise: float,
             fall: float) -> list[float]:
    """Gaussian swell peaking at `peak` seconds, asymmetric either side."""
    out = []
    for t in timebase(duration):
        width = rise if t < peak else fall
        out.append(math.exp(-((t - peak) / width) ** 2))
    return out


def ramp(duration: float, start: float, end: float,
         curve: float = 1.0) -> list[float]:
    n = samples(duration)
    if n == 1:
        return [end]
    return [start + (end - start) * ((i / (n - 1)) ** curve) for i in range(n)]


# --------------------------------------------------------------------------
# oscillators and voices
# --------------------------------------------------------------------------

def sine(freq: float, duration: float, phase: float = 0.0) -> list[float]:
    return [math.sin(TAU * freq * t + phase) for t in timebase(duration)]


def sine_track(freqs: list[float], phase: float = 0.0) -> list[float]:
    """Sine following a per-sample frequency track (integrates the phase)."""
    out = []
    for f in freqs:
        out.append(math.sin(phase))
        phase += TAU * f / SR
    return out


def modal_ring(duration: float, modes: list[tuple[float, float, float]],
               rng: random.Random) -> list[float]:
    """Sum of exponentially decaying sinusoids, one per mode.

    Modes are (freq, decay, gain). Struck metal is inharmonic, so callers pass
    ratios that are deliberately not integer multiples. The 0.2 % drift keeps
    repeated hits from sounding like the same synthetic tone twice.
    """
    t = timebase(duration)
    out = [0.0] * len(t)
    for freq, decay, gain in modes:
        if freq >= SR * 0.48 or gain <= 0.0:
            continue
        phase = rng.uniform(0.0, TAU)
        drift_hz = rng.uniform(0.7, 2.2)
        for i, x in enumerate(t):
            drift = 1.0 + 0.002 * math.sin(TAU * drift_hz * x + phase)
            out[i] += (gain * math.sin(TAU * freq * drift * x + phase)
                       * math.exp(-x / decay))
    return out


def pluck(freq: float, duration: float, rng: random.Random,
          damping: float = 0.5, blend: float = 1.0) -> list[float]:
    """Karplus-Strong plucked string, for bowstrings.

    `damping` 0..1 shortens the decay; `blend` below 1 detunes the loop into
    noise, which is what stops a bowstring sounding like a harp.
    """
    n = samples(duration)
    period = max(2, int(round(SR / max(freq, 20.0))))
    buf = [rng.uniform(-1.0, 1.0) for _ in range(period)]
    out = [0.0] * n
    loss = 1.0 - 0.5 * damping
    idx = 0
    for i in range(n):
        cur = buf[idx]
        nxt = buf[(idx + 1) % period]
        avg = 0.5 * (cur + nxt)
        if blend < 1.0 and rng.random() > blend:
            avg = -avg
        buf[idx] = avg * loss
        out[i] = cur
        idx = (idx + 1) % period
    return out


def stick_slip(duration: float, rng: random.Random, rate: float = 40.0,
               freq: float = 320.0, jitter: float = 0.6) -> list[float]:
    """Creak: irregular micro-slips ringing a resonance.

    Wood under load does not slide smoothly, it grabs and releases. Each
    release is an impulse; the wood rings at `freq` in response.
    """
    n = samples(duration)
    imp = [0.0] * n
    t = 0.0
    while t < duration:
        i = int(t * SR)
        if 0 <= i < n:
            imp[i] += rng.uniform(0.4, 1.0)
        step = (1.0 / max(rate, 1.0)) * (1.0 + rng.uniform(-jitter, jitter))
        t += max(step, 1.0 / SR)
    return resonator(imp, freq, q=14.0, gain=1.0)


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def write_wav(path: str | Path, signal: list[float]) -> Path:
    """Mono 44.1 kHz 16-bit PCM, which is what game loaders tend to want."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = array.array("h", (int(round(max(-1.0, min(1.0, s)) * 32767.0))
                               for s in signal))
    if sys.byteorder == "big":
        frames.byteswap()
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(frames.tobytes())
    return target
