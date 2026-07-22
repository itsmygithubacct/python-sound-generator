"""A physical syrinx: the Mindlin labial normal form, solved once.

Birdsong is not a sine with a glide on it. The syrinx is a pair of labia driven
into a nonlinear limit cycle by air from the sacs, and the shape of that cycle
-- not an additive harmonic recipe -- is what makes a blackbird sound like a
blackbird. Mindlin's normal form is the standard model, and wild birds have
been shown to respond to songs synthesised from it as though they were real:

    dx/dt = y
    dy/dt = -a*g^2 - b*g^2*x - g^2*x^3 - g*x^2*y + g^2*x^2 - g*x*y

`a` is air-sac pressure, `b` is syringeal muscle tension, `g` is the time
constant. This is ported from an earlier documented model; that implementation
had lost every ``g^2``
term, which drops the natural frequency to ~25 Hz -- it never oscillated, and
what it emitted was the ring of its own 300 Hz cleanup filter. Measured, it
answered a request for 2000-2500 Hz with 310 Hz and 92% of the energy below
600 Hz. The equation above is the one from the research doc, and it does
oscillate: it needs `a` above roughly 0.05, which is also why that file's
frequency table (which reaches down to a = -0.10) could never have worked.

The useful part is a symmetry. Substituting T = g*t removes `g` completely:

    X'' = -a - b*X - X^3 - X^2*X' + X^2 - X*X'

So `g` is a *pure pitch control* -- it rescales time and nothing else. Verified
across a fourfold sweep: f/g holds at 0.0937 and the harmonic ratios hold at
[0.37, 0.12, 0.03] to three digits. Two consequences, and they are the whole
design of this module:

- Pitch and timbre are separable. `g` sets the note; (a, b) set the voice.
- The cycle only has to be solved *once*. One period of the scaled system,
  stored as a table and read back with a phase accumulator, reproduces the
  physics at any pitch for the cost of a lookup -- which is what makes a
  physical model affordable in pure Python at all.

What that trades away is the chaotic regime: reading a fixed cycle assumes the
labia are in periodic motion, so the transition into deterministic chaos that
gives some species their broadest, breathiest notes is not modelled here. In
the oscillating region used below the cycle measures cleanly periodic, so
there is no chaos on the table to lose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..._audio import dsp


SR = dsp.SR
TAU = dsp.TAU

# One period is resampled to this many points. The cycle's harmonics are down
# to a percent of the fundamental by the 5th, so 1024 is far more than the
# waveform carries; it costs 8 kB and removes interpolation error as a worry.
TABLE = 1024

# Integration step in scaled time. The cycle spans ~10.7 scaled units, so this
# is ~10k steps per period -- deep into convergence (halving it moves the
# measured period by <1e-4).
_STEP = 0.001
_SETTLE = 160.0
_MAX_SPAN = 600.0


@dataclass(frozen=True)
class Voice:
    """One (a, b) operating point: a timbre, solved.

    `table` is one period of the limit cycle, peak-normalised and DC-free.
    `ratio` is f/g -- multiply by `g` for the pitch, or divide a wanted pitch
    by it to get the `g` that produces it.
    """

    alpha: float
    beta: float
    table: tuple[float, ...]
    ratio: float

    def gamma_for(self, freq: float) -> float:
        """The time constant that puts this voice at `freq`."""

        return freq / self.ratio


def _derivative(x: float, y: float, alpha: float,
                beta: float) -> tuple[float, float]:
    """The scaled normal form: no gamma, because gamma is only time."""

    return y, (-alpha - beta * x - x * x * x
               - x * x * y + x * x - x * y)


def _integrate(alpha: float, beta: float, span: float,
               x0: float = 0.01, y0: float = 0.0) -> list[float]:
    """Fixed-step RK4 over the scaled system."""

    steps = int(span / _STEP)
    out = [0.0] * steps
    x, y = x0, y0
    h = _STEP
    for i in range(steps):
        k1x, k1y = _derivative(x, y, alpha, beta)
        k2x, k2y = _derivative(x + 0.5 * h * k1x, y + 0.5 * h * k1y, alpha, beta)
        k3x, k3y = _derivative(x + 0.5 * h * k2x, y + 0.5 * h * k2y, alpha, beta)
        k4x, k4y = _derivative(x + h * k3x, y + h * k3y, alpha, beta)
        x += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y += (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        if not (-1e6 < x < 1e6):
            raise ValueError(
                f"syrinx diverged at alpha={alpha}, beta={beta}"
            )
        out[i] = x
    return out


def _upward_crossings(signal: list[float], level: float) -> list[float]:
    """Fractional indices where `signal` crosses `level` going up."""

    found = []
    for i in range(1, len(signal)):
        previous, current = signal[i - 1], signal[i]
        if previous <= level < current:
            span = current - previous
            found.append(i - 1 + (level - previous) / span if span else i - 1.0)
    return found


def solve(alpha: float = 0.20, beta: float = 0.10) -> Voice:
    """Solve one (a, b) operating point into a playable Voice.

    Raises if the pair does not self-oscillate, which is a real answer rather
    than an error: below the Hopf bifurcation the labia sit still and the bird
    is silent. The old frequency table asked for exactly that and got handed
    filter ringing instead.
    """

    trace = _integrate(alpha, beta, _MAX_SPAN)
    settled = trace[int(_SETTLE / _STEP):]
    low, high = min(settled), max(settled)
    if high - low < 0.02:
        raise ValueError(
            f"alpha={alpha}, beta={beta} does not oscillate "
            f"(swing {high - low:.2e}); alpha must be above about 0.05"
        )

    midpoint = 0.5 * (low + high)
    crossings = _upward_crossings(settled, midpoint)
    if len(crossings) < 3:
        raise ValueError(f"alpha={alpha}, beta={beta} has no periodic cycle")

    # Average across every whole cycle rather than trusting one: the period is
    # what sets pitch accuracy, and averaging drops the error to the step size.
    period = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
    ratio = 1.0 / (period * _STEP)  # cycles per scaled time unit, i.e. f/gamma

    start = crossings[0]
    table = []
    for i in range(TABLE):
        position = start + period * i / TABLE
        index = int(position)
        frac = position - index
        if index + 1 >= len(settled):
            table.append(settled[-1])
            continue
        table.append(settled[index] * (1.0 - frac) + settled[index + 1] * frac)

    mean = sum(table) / len(table)
    table = [value - mean for value in table]
    peak = max(abs(value) for value in table)
    if peak > 0.0:
        table = [value / peak for value in table]
    return Voice(alpha, beta, tuple(table), ratio)


_CACHE: dict[tuple[float, float], Voice] = {}


def voice(alpha: float = 0.20, beta: float = 0.10) -> Voice:
    """Solve and cache. Every note of a species shares one solve."""

    key = (round(alpha, 6), round(beta, 6))
    if key not in _CACHE:
        _CACHE[key] = solve(*key)
    return _CACHE[key]


def note(freqs: list[float], voice_: Voice | None = None) -> list[float]:
    """Play the limit cycle along a per-sample frequency track.

    The phase accumulator is the same one `dsp.sine_track` uses; only the
    waveform it reads differs, and that waveform is the physics.
    """

    if voice_ is None:
        voice_ = voice()
    table = voice_.table
    out = [0.0] * len(freqs)
    phase = 0.0
    for i, freq in enumerate(freqs):
        position = phase * TABLE
        index = int(position) % TABLE
        frac = position - int(position)
        following = table[(index + 1) % TABLE]
        out[i] = table[index] * (1.0 - frac) + following * frac
        phase += freq / SR
        if phase >= 1.0:
            phase -= int(phase)
    return out


def spectrum_ratios(voice_: Voice, count: int = 5) -> list[float]:
    """Harmonic amplitudes relative to the fundamental, straight from the table.

    A single period is exactly one cycle of the fundamental, so a plain DFT of
    the table reads the harmonics off directly -- bin k is harmonic k.
    """

    ratios = []
    size = len(voice_.table)
    magnitudes = []
    for k in range(1, count + 2):
        real = sum(v * math.cos(TAU * k * i / size)
                   for i, v in enumerate(voice_.table))
        imaginary = sum(v * math.sin(TAU * k * i / size)
                        for i, v in enumerate(voice_.table))
        magnitudes.append(math.hypot(real, imaginary))
    base = magnitudes[0] or 1.0
    for value in magnitudes[1:count + 1]:
        ratios.append(value / base)
    return ratios
