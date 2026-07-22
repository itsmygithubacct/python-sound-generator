"""Loop-safe extensions to the standard-library DSP core.

An ambience bed is not a one-shot. It runs under a game for minutes and must
never announce where the buffer restarts. Three things put a seam into an
otherwise good render, and each one has a fix in here.

*Filters start cold.* Every filter in :mod:`python_sound_generator._audio.dsp`
begins
from zero state, so filtered noise opens with a startup transient and ends in
steady state. Wrapped end to start that is a swell on every repeat.
:func:`loop_filter` primes a filter with the tail of the very buffer it is
about to process, which is precisely the signal that would have preceded sample
0 had the loop already been running. What comes back is the periodic steady
state, so its first sample genuinely continues its last.

*Oscillators land mid-cycle.* A 60 Hz hum in a 6.5 s buffer ends 390 cycles in
and snaps back to phase zero at the wrap. :func:`quantize` moves a requested
frequency to the nearest one completing a whole number of cycles in the buffer.
Above a few hertz that shift is far too small to hear; below it, a modulator
slower than the loop correctly flattens to no modulation at all, because a
20 s gust does not fit in a 4 s loop no matter how it is synthesised.

*Events fall off the end.* A drip placed near the end is truncated, and its
missing tail is a hole. :func:`add_wrapped` sends that tail around to the
start, where a drip beginning just before the loop point in fact belongs.

Because everything here is periodic by construction, :func:`master` can drop
the fades that ``dsp.master`` applies: a fade to zero at both ends *is* a seam.
:func:`seam` measures what the result achieves.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from . import dsp


SR = dsp.SR
TAU = dsp.TAU

# Long enough for every filter here to settle far below 16-bit resolution:
# the slowest is a q=20 resonator at 40 Hz, whose energy is gone in ~0.16 s.
# Reverb sets its own warmup from its decay time instead; see `reverb`.
WARMUP = 0.5


Signal = list[float]
Transform = Callable[[Signal], Signal]


# --------------------------------------------------------------------------
# periodicity
# --------------------------------------------------------------------------

def period(duration: float) -> float:
    """The buffer's true period in seconds.

    ``dsp.samples`` rounds, so a requested 6.5 s is not necessarily 6.5 s of
    samples. Quantising against the request rather than the buffer would leave
    a fraction of a cycle at the wrap, which is the seam this module exists to
    remove.
    """

    return dsp.samples(duration) / SR


def cycles(freq: float, duration: float, minimum: int = 0) -> int:
    """Whole cycles of `freq` nearest to fitting one loop period."""

    return max(minimum, int(round(abs(freq) * period(duration))))


def quantize(freq: float, duration: float, minimum: int = 0) -> float:
    """Nearest frequency to `freq` that completes whole cycles in one loop.

    `minimum` guards audible voices: a 30 Hz drone in a 0.25 s loop would
    otherwise round to 8 cycles, which is right, but a 0.3 Hz gust in the same
    loop rounds to zero, which is also right and must stay allowed. Pass 1 for
    anything that has to keep sounding, 0 for modulators.
    """

    return cycles(freq, duration, minimum) / period(duration)


def sine(freq: float, duration: float, phase: float = 0.0,
         minimum: int = 1) -> Signal:
    """A sine that arrives back at its starting phase exactly at the wrap."""

    return dsp.sine(quantize(freq, duration, minimum), duration, phase)


def lfo(rate: float, duration: float, depth: float = 1.0,
        phase: float = 0.0) -> Signal:
    """Unipolar modulator in [1 - depth, 1], whole-cycle and seam-free.

    A rate too slow to complete one cycle flattens to a constant, which is the
    honest answer: the loop is shorter than the movement being asked for.
    """

    if depth <= 0.0:
        return [1.0] * dsp.samples(duration)
    wave = sine(rate, duration, phase, minimum=0)
    floor = 1.0 - depth
    return [floor + depth * (0.5 + 0.5 * value) for value in wave]


def drift(duration: float, rates: tuple[float, ...], rng: random.Random,
          depth: float = 1.0, curve: float = 1.0) -> Signal:
    """Several incommensurate LFOs multiplied into one wandering envelope.

    One LFO is a tremolo and the ear hears the rate. Three at unrelated speeds
    read as weather: the pattern only repeats at the loop point, by which time
    it is the loop's job to be inaudible.
    """

    out = [1.0] * dsp.samples(duration)
    for rate in rates:
        part = lfo(rate, duration, depth, rng.uniform(0.0, TAU))
        out = [a * b for a, b in zip(out, part)]
    if curve != 1.0:
        out = [value ** curve for value in out]
    return out


# --------------------------------------------------------------------------
# filters, primed as if the loop had always been running
# --------------------------------------------------------------------------

def _wrapped_prefix(signal: Signal, count: int) -> Signal:
    """The `count` samples that cyclically precede sample 0."""

    n = len(signal)
    if count <= 0 or n == 0:
        return []
    if count <= n:
        return signal[n - count:]
    return (signal * (count // n + 1))[-count:]


def loop_filter(signal: Signal, transform: Transform,
                warmup: float = WARMUP) -> Signal:
    """Run a zero-state filter as though `signal` had been looping forever."""

    count = dsp.samples(warmup) if warmup > 0.0 else 0
    if count <= 0:
        return transform(signal)
    return transform(_wrapped_prefix(signal, count) + signal)[count:]


def lowpass(signal: Signal, cutoff: float, order: int = 4,
            warmup: float = WARMUP) -> Signal:
    return loop_filter(signal, lambda s: dsp.lowpass(s, cutoff, order), warmup)


def highpass(signal: Signal, cutoff: float, order: int = 4,
             warmup: float = WARMUP) -> Signal:
    return loop_filter(signal, lambda s: dsp.highpass(s, cutoff, order), warmup)


def bandpass(signal: Signal, low: float, high: float, order: int = 4,
             warmup: float = WARMUP) -> Signal:
    return loop_filter(
        signal, lambda s: dsp.bandpass(s, low, high, order), warmup
    )


def resonator(signal: Signal, freq: float, q: float = 8.0, gain: float = 1.0,
              warmup: float = WARMUP) -> Signal:
    return loop_filter(
        signal, lambda s: dsp.resonator(s, freq, q, gain), warmup
    )


def svf_bandpass(signal: Signal, freqs: Signal, q: float = 1.0,
                 warmup: float = WARMUP) -> Signal:
    """Moving bandpass primed circularly; the tuning track wraps with it."""

    count = dsp.samples(warmup) if warmup > 0.0 else 0
    if count <= 0:
        return dsp.svf_bandpass(signal, freqs, q)
    primed = _wrapped_prefix(signal, count) + signal
    tuned = _wrapped_prefix(freqs, count) + freqs
    return dsp.svf_bandpass(primed, tuned, q)[count:]


def _svf_lowpass(signal: Signal, freqs: Signal, q: float) -> Signal:
    """The same topology-preserving SVF as ``dsp.svf_bandpass``, tapped at v2.

    A state variable filter computes every response at once and you choose one
    by which node you read: v1 is the bandpass that module returns, v2 is the
    lowpass. Nothing else differs, so this stays exactly in step with it.
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
        out[i] = v2
    return out


def svf_lowpass(signal: Signal, freqs: Signal, q: float = 0.7,
                warmup: float = WARMUP) -> Signal:
    """Lowpass whose cutoff moves per sample, primed circularly.

    This is what "brighter when louder" needs: a cutoff that rides an envelope
    continuously. The obvious alternative -- refiltering in short blocks -- has
    to restart the filter's state at every block boundary, which prints a small
    transient at the block rate; sweeping one filter has no boundaries to
    print.
    """

    count = dsp.samples(warmup) if warmup > 0.0 else 0
    if count <= 0:
        return _svf_lowpass(signal, freqs, q)
    primed = _wrapped_prefix(signal, count) + signal
    tuned = _wrapped_prefix(freqs, count) + freqs
    return _svf_lowpass(primed, tuned, q)[count:]


def _one_pole_periodic(signal: Signal, coeff: float) -> Signal:
    """A one-pole lowpass solved for its exact periodic steady state.

    Priming by running over a wrapped prefix, as `loop_filter` does, needs a
    prefix several time constants long. A modulator's time constant is
    *seconds* -- 10.6 s at 0.015 Hz -- so the prefix would dwarf the loop and
    cost more than the render. It is also unnecessary, because a one-pole can
    be solved rather than converged to.

    Run the filter from zero and the final state is S. Run it from state s and
    the final state is (1-c)^N * s + S. The periodic steady state is the state
    that reproduces itself, so s = S / (1 - (1-c)^N): one pass to find S, one
    to produce the output, and the result is exact rather than approximate.
    """

    n = len(signal)
    if n == 0:
        return []
    state = 0.0
    for x in signal:
        state += coeff * (x - state)
    decay = (1.0 - coeff) ** n
    if 1.0 - decay > 1e-15:
        state /= 1.0 - decay
    out = [0.0] * n
    for i, x in enumerate(signal):
        state += coeff * (x - state)
        out[i] = state
    return out


def smooth(signal: Signal, rate: float, poles: int = 2) -> Signal:
    """Lowpass a signal at a *sub-audio* rate, without a cutoff floor.

    ``dsp``'s filters clamp their cutoff to 20 Hz, which is right for audio --
    it is the bottom of hearing -- and wrong for a modulator, which lives one
    to three decades below it. Asking those filters for 0.015 Hz silently gets
    you 20 Hz: a swell that should breathe over a minute arrives as a flutter,
    which is audible immediately as chop.

    So modulators get their own filter. A one-pole cascade is enough because
    callers here min-max normalise the result, which discards the gain and
    leaves only the shape.
    """

    coeff = 1.0 - math.exp(-TAU * max(rate, 1e-6) / SR)
    out = list(signal)
    for _ in range(max(1, poles)):
        out = _one_pole_periodic(out, coeff)
    return out


def rotate(signal: Signal, seconds: float) -> Signal:
    """Shift a loop in time, wrapping. A delay inside a loop is a rotation."""

    n = len(signal)
    if n == 0:
        return []
    offset = int(round(seconds * SR)) % n
    return signal[-offset:] + signal[:-offset] if offset else list(signal)


# --------------------------------------------------------------------------
# space
# --------------------------------------------------------------------------

# Freeverb's tunings, which are chosen to be mutually prime at 44.1 kHz so the
# combs do not reinforce each other into a ringing pitch.
_COMBS = (1_116, 1_188, 1_277, 1_356)
_ALLPASSES = (556, 441)


def reverb_t60(room: float) -> float:
    """Seconds for the comb bank to fall 60 dB, used to size the warmup."""

    gain = min(max(room, 0.05), 0.97)
    passes = -3.0 / math.log10(gain)
    return passes * (sum(_COMBS) / len(_COMBS)) / SR


def _comb(signal: Signal, delay: int, feedback: float,
          damp: float) -> Signal:
    buf = [0.0] * delay
    out = [0.0] * len(signal)
    store = 0.0
    index = 0
    keep = 1.0 - damp
    for i, x in enumerate(signal):
        value = buf[index]
        out[i] = value
        store = value * keep + store * damp
        buf[index] = x + store * feedback
        index += 1
        if index >= delay:
            index = 0
    return out


def _allpass(signal: Signal, delay: int, feedback: float = 0.5) -> Signal:
    buf = [0.0] * delay
    out = [0.0] * len(signal)
    index = 0
    for i, x in enumerate(signal):
        value = buf[index]
        out[i] = value - x
        buf[index] = x + value * feedback
        index += 1
        if index >= delay:
            index = 0
    return out


def _reverb_core(signal: Signal, room: float, damp: float) -> Signal:
    out = [0.0] * len(signal)
    for delay in _COMBS:
        for i, value in enumerate(_comb(signal, delay, room, damp)):
            out[i] += value
    out = dsp.scale(out, 1.0 / len(_COMBS))
    for delay in _ALLPASSES:
        out = _allpass(out, delay)
    return out


def reverb(signal: Signal, room: float = 0.84, damp: float = 0.35,
           warmup: float | None = None) -> Signal:
    """Loop-safe Schroeder reverb: parallel combs into series allpasses.

    A reverb tail is by far the longest memory in the chain, so it needs a
    warmup sized by its own decay rather than the module default. Twice T60
    leaves the priming error ~120 dB down, well beneath the 16-bit floor the
    WAV is quantised to anyway.
    """

    if warmup is None:
        warmup = 2.0 * reverb_t60(room)
    return loop_filter(
        signal, lambda s: _reverb_core(s, room, damp), warmup
    )


def tail(signal: Signal, room: float = 0.84, damp: float = 0.35) -> Signal:
    """Reverb for a one-shot: no priming, because there is no loop to prime."""

    return _reverb_core(signal, room, damp)


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------

def add_wrapped(dst: Signal, start: float, src: Signal,
                gain: float = 1.0) -> None:
    """Mix `src` into `dst` at `start` seconds, wrapping the end to the start."""

    n = len(dst)
    if n == 0 or not src:
        return
    offset = int(round(start * SR)) % n
    pos = 0
    total = len(src)
    while pos < total:
        span = min(n - offset, total - pos)
        for k in range(span):
            dst[offset + k] += src[pos + k] * gain
        pos += span
        offset = 0


def scatter(duration: float, rng: random.Random, count: int) -> list[float]:
    """`count` random onsets across one loop period."""

    span = period(duration)
    return [rng.uniform(0.0, span) for _ in range(count)]


def impulses(duration: float, rng: random.Random, count: int,
             low: float = 0.25, high: float = 1.0) -> Signal:
    """A bed of single-sample strikes, to be filtered into patter or crackle.

    Filtering one shared impulse bed costs one filter pass; giving every
    droplet its own filter would cost hundreds and sound no different, because
    they would all still share the one impulse response.
    """

    out = [0.0] * dsp.samples(duration)
    if not out:
        return out
    for _ in range(count):
        index = rng.randrange(len(out))
        out[index] += rng.uniform(low, high) * rng.choice((-1.0, 1.0))
    return out


# --------------------------------------------------------------------------
# levels and validation
# --------------------------------------------------------------------------

def level(signal: Signal, rms: float) -> Signal:
    """Scale a layer to a target RMS.

    Filtered noise arrives at a level set by the filter's bandwidth rather than
    by the gain written beside it. The same white noise is 24 dB quieter
    through a 75 Hz lowpass than through a 3.2 kHz highpass, because the
    lowpass threw away 99.7% of the spectrum. Balance the two by eye and the
    hiss buries the rumble -- lava measured brighter than fire that way, which
    is how this function came to exist.

    So a bed declares the level it wants. Where that level carries information
    -- how hard it is raining, how many people are in the room -- the target
    moves with `intensity`, and density still reads as loudness. Discrete
    events stay on plain gains: normalising a layer holding three drips to a
    target RMS would make three drips as loud as thirty, which is precisely
    what the caller was trying to say by only placing three.

    Constant gain, so a periodic layer stays periodic.
    """

    current = _rms(signal)
    if current < 1e-12:
        return list(signal)
    return dsp.scale(signal, rms / current)


def master(signal: Signal, peak_db: float, drive: float = 1.25) -> Signal:
    """Loop mastering: `dsp.master` without the fades, which would be seams.

    Both remaining stages are memoryless, so a periodic input stays periodic.
    """

    return dsp.normalize(dsp.soft_limit(signal, drive), peak_db)


def _rms(signal: Signal) -> float:
    if not signal:
        return 0.0
    return math.sqrt(sum(s * s for s in signal) / len(signal))


MAX_STEP_RATIO = 1.0
MAX_LEVEL_DRIFT = 4.0


@dataclass(frozen=True)
class Seam:
    """What the join between the last sample and the first actually does."""

    step: float
    largest_step: float
    step_ratio: float
    head_rms: float
    tail_rms: float
    level_ratio: float

    def is_seamless(self, max_step_ratio: float = MAX_STEP_RATIO,
                    max_level_drift: float = MAX_LEVEL_DRIFT) -> bool:
        """Both failures have to be absent; neither metric sees the other."""

        return (
            self.step_ratio <= max_step_ratio
            and 1.0 / max_level_drift <= self.level_ratio <= max_level_drift
        )


def seam(signal: Signal, window: float = 0.05) -> Seam:
    """Measure the loop point for the two ways it fails: a click and a swell.

    `step_ratio` is the jump across the wrap over the largest jump between any
    neighbouring pair inside the buffer. The normalisation looks odd until you
    notice what it buys: a genuinely periodic buffer *cannot* score much above
    1, because its wrap is simply one more of its own sample steps and cannot
    outrank all the rest. Measured here the worst honest render scores 0.72
    and most are far below it, while cold-started filters run 2.4 to 106 -- a
    gap wide enough that 1.0 gates it without tuning. (Normalising by the mean
    step instead leaves narrowband beds scoring 3+ purely from their own quiet
    passages, which is why this uses the peak.)

    `level_ratio`, the energy of the closing window over the opening one,
    exists because `step_ratio` is blind to the more damaging failure. An
    unprimed reverb opens in silence and closes under a full tail: its wrap
    step is tiny -- it scores 0.002, better than most honest renders -- while
    it swells by nine orders of magnitude on every repeat. A slow unquantised
    LFO hides the same way. Sparse events make this ratio jitter either side of
    1 on content alone, so the band around it has to stay wide; broken loops
    miss it by factors of hundreds, not percent.
    """

    n = len(signal)
    if n < 2:
        raise ValueError("a seam needs at least two samples")
    largest = max(
        (abs(signal[i] - signal[i - 1]) for i in range(1, n)), default=0.0
    )
    step = abs(signal[0] - signal[n - 1])
    width = max(1, min(n // 2, dsp.samples(window)))
    head = _rms(signal[:width])
    tail_level = _rms(signal[n - width:])
    return Seam(
        step=step,
        largest_step=largest,
        step_ratio=step / max(largest, 1e-12),
        head_rms=head,
        tail_rms=tail_level,
        level_ratio=tail_level / max(head, 1e-12),
    )
