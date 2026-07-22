"""Lava, built as a hybrid: a recorded roar with synthesized bubbles on top.

A sibling of ``fire_sound_generator``, and deliberately so. Molten rock is,
per the roadmap, "a slow viscous burn, nearer fire than weather", and fire's
burning bed at ``--scale inferno`` is most of what lava is: a chaotic aperiodic
roar of broadband turbulence. That part is *recorded* -- it is the same 29 s of
wood burning fire uses, borrowed by copy into this generator's own ``sources/``
so lava stays self-contained, then slowed until the turbulence reads as viscous
and low, rolled off at the top, and de-flickered so the fast crackle of a
fireplace settles into the heavy churn of a lake of rock.

What lava has that a fireplace does not is *events*: a bubble of gas working its
way up through the melt and surfacing with a resonant pitched "bloop". Those
have a rate -- so many a second -- but a rate is not a period, and this is the
sharpened rule the whole generator turns on: the roar is recorded because it is
chaotic, and the bubbles are synthesized because a bubble is a clean resonant
transient with a pitch. Each bloop is an exponentially decaying tone gliding
*down* as the cavity opens, with a wet noise burst as the skin breaks. They are
scattered at a slow irregular rate over the recorded bed.

So nothing sustained here is oscillators, and nothing impulsive here is a
recording. That split is the point.

Loops
-----
The bed is a recording, so a loop cannot be seamless by construction the way a
synthesized one can. It is made seamless the way fire's is -- an equal-power
crossfade of the tail into the head -- and then *measured* rather than asserted,
with ``environment_ambience_generator``'s seam metric, imported rather than
reimplemented so every generator in the project agrees on what seamless means.
Bubbles in a loop are kept clear of the wrap so the join stays pure bed.

Everything returns unmastered mono floating-point PCM plus the source files it
consumed. The cue registry owns mastering levels, so every public render follows
the same contract, and the consumed filenames reach the manifest.
"""

from __future__ import annotations

from ..._audio.assets import asset_names

from ..._audio.assets import resolve_asset

from ..._audio.foley_numpy import read_recording

from ..._audio.foley_numpy import write_wav_scipy as write_wav  # noqa: F401

from ..._audio.foley_numpy import equal_power_crossfade_loop, random_window, tile_to_frames

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

# The one bundled recording, borrowed by copy from fire_sound_generator. Both
# generators point at the same CC0 origin (PagDev's Fireplace Sound loop); see
# sources/provenance.json. Lava keeps its own copy so it does not depend on a
# sibling package's files staying put.
BED = "pagdev_fireplace.wav"


@dataclass(frozen=True)
class Scale:
    """How big the lava is, and what that does to the recording and the bubbles.

    `speed` is varispeed and does the real work, exactly as in fire: a small
    vent is not a lake turned down, it is faster and higher, because the eddies
    and the bubbles are smaller. Resampling moves rate and pitch together, which
    is what keeps that true. Every lava speed sits below 1.0 because a lake of
    rock is slower and heavier than the fireplace the bed was recorded from.

    `top` is the lowpass that rolls the crackle off the top -- molten rock has no
    bright snap -- and `weight` is the low-end mass added back. `bubble_rate` and
    `bubble_pitch` set how often the melt surfaces and how low it sounds when it
    does; a bigger pool bubbles lower and, at the eruption end, harder.
    """

    name: str
    label: str
    description: str
    speed: float
    weight: float
    top: float
    bubble_rate: float
    bubble_pitch: float


SCALES: dict[str, Scale] = {
    "vent": Scale("vent", "Vent", "a small molten vent, spitting and quick",
                  0.70, 0.55, 1_600.0, 1.6, 210.0),
    "pool": Scale("pool", "Pool", "a contained pool of melt",
                  0.56, 0.90, 1_300.0, 1.1, 150.0),
    "lake": Scale("lake", "Lake", "a wide lava lake, slow and heavy",
                  0.46, 1.20, 1_050.0, 0.8, 120.0),
    "caldera": Scale("caldera", "Caldera", "a vast molten caldera, sub and slow",
                     0.38, 1.50, 850.0, 0.55, 92.0),
    "eruption": Scale("eruption", "Eruption", "an erupting vent, low and violent",
                      0.31, 1.80, 700.0, 1.3, 72.0),
}




@lru_cache(maxsize=8)
def _load_source(filename: str) -> np.ndarray:
    path = resolve_asset(SOURCE_DIRECTORY / filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing bundled lava source {path}; see sources/provenance.json"
        )
    _, signal = read_recording(path, expected_rate=SR, label="lava source")
    return signal


def _frames(seconds: float) -> int:
    return max(1, round(seconds * SR))


def _varispeed(signal: np.ndarray, speed: float) -> np.ndarray:
    return varispeed_configured(
        signal, speed, minimum=0.25, maximum=4.0, denominator=512
    )


def _lowpass(signal: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    return sosfilt(
        butter(order, cutoff, btype="lowpass", fs=SR, output="sos"), signal
    )


def _highpass(signal: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    return sosfilt(
        butter(order, cutoff, btype="highpass", fs=SR, output="sos"), signal
    )


def _bandpass(signal: np.ndarray, low: float, high: float,
              order: int = 2) -> np.ndarray:
    low = max(20.0, low)
    high = min(SR * 0.45, high)
    if high <= low:
        return signal
    return sosfilt(
        butter(order, [low, high], btype="bandpass", fs=SR, output="sos"),
        signal,
    )


def _tilt(signal: np.ndarray, weight: float) -> np.ndarray:
    """Add mass using a low band of the recording itself.

    Fire's tilt also adds a bright band; lava never does -- brightness is what
    the `top` lowpass exists to remove. So this only ever pulls the low end up,
    which is the bulk of molten rock a fireplace does not have.
    """

    if abs(weight) <= 1e-6:
        return signal
    low = _lowpass(signal, 220.0)
    return signal + weight * low


def _deflicker(signal: np.ndarray, amount: float,
               slow: float = 4.0, fast: float = 28.0) -> np.ndarray:
    """Tame the fast amplitude flicker toward the slow breathing.

    A fireplace flickers -- its loudness jitters several times a second as small
    pockets flare and fail. Lava does not; it heaves. This measures the signal's
    amplitude envelope at two rates, a slow contour (the heave) and a faster one
    (the flicker), and applies the gain that would drag the fast one onto the
    slow one. Sign and fine waveform structure are untouched, so the timbre
    survives; only the loudness wobble between `slow` and `fast` Hz is flattened,
    which is exactly the fireplace tell that has to go.

    Run on the whole recording before any window is cut, so its filter transient
    lives at material t=0 where no window reaches -- the same discipline fire
    learned the hard way with its post-window filters.
    """

    if amount <= 0.0:
        return signal
    envelope = np.abs(signal)
    slow_env = _lowpass(envelope, slow)
    fast_env = _lowpass(envelope, fast)
    gain = np.ones_like(signal)
    active = fast_env > 1e-6
    gain[active] = np.clip(slow_env[active] / fast_env[active], 0.5, 1.7)
    return signal * (1.0 - amount + amount * gain)


def _fades(signal: np.ndarray, seconds: float=0.004) -> np.ndarray:
    return fade_edges(signal, seconds, sample_rate=SR)


def _tile(signal: np.ndarray, frames: int) -> np.ndarray:
    return tile_to_frames(signal, frames)


def _crossfade_loop(signal: np.ndarray, frames: int, fade: float=1.2) -> np.ndarray:
    return equal_power_crossfade_loop(
        signal, frames, fade, sample_rate=SR
    )


def _heave(frames: int, rng: np.random.Generator, depth: float,
           rate: float) -> np.ndarray:
    """A slow irregular amplitude wander, quantised to whole cycles.

    Lava's version of fire's flutter, but slower and shallower: molten rock
    heaves, it does not flicker. Quantised so the modulator itself does not break
    the loop -- an LFO that does not close on a whole cycle swells across the
    wrap, which the seam metric catches as level drift rather than as a click.
    """

    if depth <= 0.0:
        return np.ones(frames)
    seconds = frames / SR
    out = np.zeros(frames)
    t = np.arange(frames) / SR
    for _ in range(3):
        cycles = max(1, round(rate * seconds * float(rng.uniform(0.6, 1.6))))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        out += np.sin(2.0 * math.pi * cycles * t / seconds + phase)
    out /= 3.0
    return 1.0 + depth * out


def _whole_cycle_swell(frames: int, cycles: int, depth: float,
                       phase: float = 0.0) -> np.ndarray:
    """A single slow swell that closes exactly on the loop wrap.

    Whole cycles by construction, so a loop multiplied by it stays a loop. Used
    for the passing wave in `flow`, where the movement is one deliberate rise and
    fall rather than the three unrelated ones `_heave` blends into weather.
    """

    cycles = max(1, cycles)
    t = np.arange(frames) / SR
    seconds = frames / SR
    wave = np.sin(2.0 * math.pi * cycles * t / seconds + phase)
    return 1.0 + depth * wave


def _choose(rng: np.random.Generator, choices: Sequence[str]) -> str:
    return choices[int(rng.integers(0, len(choices)))]


def _mix_at(destination: np.ndarray, source: np.ndarray, start_frame: int, gain: float) -> None:
    mix_at_frames(destination, source, start_frame, gain)


@lru_cache(maxsize=32)
def _molten_bed(scale: Scale, speed_multiplier: float = 1.0) -> np.ndarray:
    """The whole fireplace, slowed and worked into molten rock.

    Deliberately the *whole* recording and not a window of it: every filter a cue
    wants runs here, on the full material, before `_window` cuts into it.
    Filtering after the cut would start the filter cold at the head of the loop
    while its tail carried full state -- a swell at the join, which is exactly
    the seam failure this project keeps re-learning. Filter here and the
    transient stays at t=0 of material no window reaches.

    Order matters: slow it down first (which already halves the flicker rate and
    drops the spectrum), de-flicker what is left, add low mass, then roll off the
    top. What comes back is heavy, dark, viscous churn.
    """

    signal = _varispeed(_load_source(BED), scale.speed * speed_multiplier)
    signal = _deflicker(signal, 0.6)
    signal = _tilt(signal, scale.weight)
    signal = _lowpass(signal, scale.top)
    peak = float(np.max(np.abs(signal)))
    signal = signal / peak if peak > 1e-9 else signal
    signal.setflags(write=False)
    return signal


def _window(signal: np.ndarray, rng: np.random.Generator, seconds: float) -> np.ndarray:
    return random_window(signal, rng, seconds, sample_rate=SR)


def _bed(scale: Scale, rng: np.random.Generator, seconds: float,
         speed_multiplier: float = 1.0) -> np.ndarray:
    """A random window of the molten bed. The root of every cue here."""

    return _window(_molten_bed(scale, speed_multiplier), rng, seconds)


# --------------------------------------------------------------------------
# the bubble: the synthesized half of the hybrid
# --------------------------------------------------------------------------

def _bloop(rng: np.random.Generator, pitch: float, decay: float,
           wetness: float = 0.5) -> np.ndarray:
    """One molten bubble surfacing: a resonant tone that glides down, and a
    wet burst as the skin breaks.

    The pitch falls rather than rises because the cavity *opens* as the bubble
    bursts -- a heavy viscous "blorp", not the rising chirp of a water drop
    closing. A little second partial gives the tone body, and a band of noise
    around the pitch, decaying faster than the tone, is the wet break. Nothing
    here is a recording; a bubble is a clean resonant event and synthesis is what
    that half of the hybrid is for.
    """

    total = _frames(min(0.7, max(0.05, decay * 5.0)))
    t = np.arange(total) / SR
    high = pitch
    low = pitch * float(rng.uniform(0.42, 0.58))
    glide_tau = decay * 0.55
    instantaneous = low + (high - low) * np.exp(-t / glide_tau)
    phase = 2.0 * np.pi * np.cumsum(instantaneous) / SR
    tone = np.sin(phase) * np.exp(-t / decay)
    tone += 0.30 * np.sin(2.0 * phase) * np.exp(-t / (decay * 0.7))

    noise = rng.standard_normal(total)
    noise = _bandpass(noise, pitch * 0.5, pitch * 6.0)
    wet = noise * np.exp(-t / (decay * 0.30)) * wetness

    return _fades(tone + wet, 0.003)


def _bubble_pitch(scale: Scale, rng: np.random.Generator,
                  spread: float = 0.5) -> float:
    """A pitch near the scale's base, scattered so no two bubbles match."""

    return scale.bubble_pitch * float(rng.uniform(1.0 - spread, 1.0 + spread))


# --------------------------------------------------------------------------
# loops: the recorded roar
# --------------------------------------------------------------------------

def molten(scale: Scale, seconds: float,
           rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """A lava bed, as a seamless loop: the roar, heaving, with sparse bubbles.

    This is the bed under everything -- a lake of rock churning. Two windows of
    the recording at different speeds so a short loop does not repeat one
    recognisable stretch, a slow heave over the top, and a handful of bubbles
    scattered in the interior. The bubbles stay clear of the first and last of
    the loop so the wrap the crossfade repaired is left as pure bed, which is
    what keeps the seam measurable and clean.
    """

    frames = _frames(seconds)
    fade = 1.2
    bed = _crossfade_loop(_bed(scale, rng, seconds + fade), frames, fade)
    bed = bed * _heave(frames, rng, 0.18, 0.35)

    # A second, faster window laid under it, offset by a third so the two passes
    # do not line up on the wrap. Same trick fire's `burning` uses.
    second = _crossfade_loop(
        _bed(scale, rng, seconds + fade, speed_multiplier=1.27), frames, fade
    )
    out = bed + 0.45 * np.roll(second, frames // 3)

    out = _scatter_bubbles(out, scale, rng, frames, seconds,
                           rate=scale.bubble_rate, gain=0.5)
    return out, [BED]


def flow(scale: Scale, seconds: float,
         rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """A slow surge of moving lava: the bed, with a wave passing through it.

    Moving rock is not a still pool. The difference is a single slow swell that
    closes on the loop wrap -- one deliberate rise and fall, the crest of the
    flow going by -- plus a thin bright band that lifts on that crest, because a
    moving surface catches and tears more than a settled one does. It breaks the
    skin more too, so the bubbles come faster. The bright band is highpassed off
    the full recording before the window is cut, for the reason in `_molten_bed`.
    """

    frames = _frames(seconds)
    fade = 1.2
    bed = _crossfade_loop(
        _bed(scale, rng, seconds + fade, speed_multiplier=1.15), frames, fade
    )
    bed = bed * _heave(frames, rng, 0.12, 0.5)

    swell_cycles = max(1, round(0.6 * seconds * float(rng.uniform(0.8, 1.2))))
    swell = _whole_cycle_swell(frames, swell_cycles, 0.35,
                               float(rng.uniform(0.0, 2.0 * math.pi)))

    bright_material = _highpass(_molten_bed(scale, 1.15), scale.top * 0.8)
    bright = _crossfade_loop(_window(bright_material, rng, seconds + fade),
                             frames, fade)

    out = bed * swell + 0.4 * bright * swell
    out = _scatter_bubbles(out, scale, rng, frames, seconds,
                           rate=scale.bubble_rate * 1.6, gain=0.55)
    return out, [BED]


def _scatter_bubbles(bed: np.ndarray, scale: Scale, rng: np.random.Generator,
                     frames: int, seconds: float, rate: float,
                     gain: float) -> np.ndarray:
    """Place bubbles at an irregular rate, kept clear of the loop wrap.

    The onsets live in [0.2 s, seconds - 0.7 s] so every bubble rings out inside
    the buffer and the head and tail windows the seam metric reads stay pure bed.
    A rate is not a period: the onsets are drawn irregularly, so the loop's
    bubbles do not tick.
    """

    first = _frames(0.2)
    last = frames - _frames(0.7)
    if last <= first:
        return bed
    count = int(round(rate * seconds))
    for _ in range(count):
        onset = int(rng.integers(first, last))
        decay = float(rng.uniform(0.10, 0.24))
        pitch = _bubble_pitch(scale, rng)
        bloop = _bloop(rng, pitch, decay, wetness=0.4)
        _mix_at(bed, bloop, onset, gain * float(rng.uniform(0.6, 1.0)))
    return bed


# --------------------------------------------------------------------------
# events: the synthesized bubbles, over a recorded surge
# --------------------------------------------------------------------------

def lava(scale: Scale, rng: np.random.Generator,
         intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """The game hazard: a molten splash and surge, ~0.85 s.

    This is what Flying Joust loads as ``sfx/lava.wav`` -- a one-shot the lava tile
    plays, its built-in synth a 0.62 s decaying gloop. Here it is a short window
    of the churning bed swelling up and settling, with a foreground splash: one
    big low bubble at the front and a couple of smaller ones behind it, plus a
    wet noise break as the surface tears. Four variants come from four seeds:
    different bed windows and different bubbles, genuinely different stretches.
    """

    seconds = 0.85
    frames = _frames(seconds)
    bed = _bed(scale, rng, seconds)
    t = np.arange(frames) / SR
    surge = (1.0 - np.exp(-t / 0.03)) * np.exp(-t / 0.55)
    out = bed * surge * 0.8

    # the splash: a wet noise break right at the front
    splash = rng.standard_normal(_frames(0.18))
    splash = _bandpass(splash, scale.bubble_pitch * 0.6, scale.top)
    st = np.arange(len(splash)) / SR
    splash = splash * np.exp(-st / 0.06)
    _mix_at(out, _fades(splash, 0.002), _frames(0.005), 0.6 * intensity)

    # one big low bloop leading, then a couple of smaller ones surfacing behind
    big = _bloop(rng, _bubble_pitch(scale, rng, spread=0.25) * 0.8,
                 decay=float(rng.uniform(0.20, 0.30)), wetness=0.6)
    _mix_at(out, big, _frames(float(rng.uniform(0.01, 0.05))), intensity)
    for _ in range(int(rng.integers(1, 3))):
        onset = _frames(float(rng.uniform(0.12, 0.45)))
        follow = _bloop(rng, _bubble_pitch(scale, rng),
                        decay=float(rng.uniform(0.10, 0.18)), wetness=0.4)
        _mix_at(out, follow, onset, 0.55 * intensity)

    return _fades(out), [BED]


def bubble(scale: Scale, rng: np.random.Generator,
           intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """A single molten bloop: one bubble surfacing, and nothing else.

    A tiny window of the bed sits under it for context -- a bubble in a vacuum is
    a sound effect, a bubble over a churn is lava -- but the bed is quick and low
    and the bloop is the cue.
    """

    bloop = _bloop(rng, _bubble_pitch(scale, rng, spread=0.35),
                   decay=float(rng.uniform(0.16, 0.26)), wetness=0.55)
    total = max(len(bloop), _frames(0.4))
    out = np.zeros(total)

    bed = _bed(scale, rng, total / SR)
    t = np.arange(total) / SR
    bed = bed * np.exp(-t / 0.18)
    _mix_at(out, bed, 0, 0.35)
    _mix_at(out, bloop, _frames(0.01), intensity)
    return _fades(out), [BED]


def burst(scale: Scale, rng: np.random.Generator,
          intensity: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """A cluster of bubbles erupting: several surfacing in quick succession.

    A pocket of gas reaching the surface all at once. The bed swells under it as
    the melt is displaced, and four to seven bubbles surface across the first
    two-thirds of a second at an irregular rate -- a rate, again, not a beat.
    """

    seconds = 1.2
    frames = _frames(seconds)
    bed = _bed(scale, rng, seconds)
    t = np.arange(frames) / SR
    swell = np.clip(t / 0.08, 0.0, 1.0) * np.exp(-t / 0.7)
    out = bed * swell * 0.7

    count = int(rng.integers(4, 8))
    for index in range(count):
        onset = _frames(float(rng.uniform(0.0, 0.66)))
        decay = float(rng.uniform(0.10, 0.22))
        pitch = _bubble_pitch(scale, rng)
        bloop = _bloop(rng, pitch, decay, wetness=0.5)
        gain = intensity * float(rng.uniform(0.5, 1.0))
        _mix_at(out, bloop, onset, gain)

    return _fades(out), [BED]


def crust_crack(scale: Scale, rng: np.random.Generator,
                intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Hot rock cracking: a short bright crack over the dark bed.

    The one bright event here, and it earns its place by being a *clean*
    transient -- cooling crust fracturing is a snap and a ring, not a chaotic
    roar, which is the kind of thing synthesis does well. A fast burst of
    highpassed noise is the fracture; a short decaying resonance above the bed's
    ceiling is the rock ringing; the bed underneath, quick and dark, says the
    crack is on the surface of something molten.
    """

    total = _frames(0.7)
    out = np.zeros(total)
    t = np.arange(total) / SR

    # the fracture: a fast bright noise crack
    crack = rng.standard_normal(total)
    crack = _highpass(crack, 2_200.0)
    crack = crack * np.exp(-t / 0.012)
    _mix_at(out, crack, 0, 0.7 * intensity)

    # the ring: the rock resonating for a moment after
    ring_freq = float(rng.uniform(900.0, 1_600.0))
    ring = np.sin(2.0 * np.pi * ring_freq * t) * np.exp(-t / 0.08)
    ring += 0.4 * np.sin(2.0 * np.pi * ring_freq * 1.5 * t) * np.exp(-t / 0.05)
    _mix_at(out, ring, _frames(0.001), 0.5 * intensity)

    # the bed: a quick dark window so the crack sits on molten rock
    bed = _bed(scale, rng, 0.7)
    bed = bed * np.exp(-t / 0.30)
    _mix_at(out, bed, 0, 0.35)

    return _fades(out), [BED]


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def master(signal: np.ndarray, peak_db: float,
           loop: bool = False) -> list[float]:
    """Final chain shared by every cue: soft limit, fade, normalize.

    A loop is not faded at its edges: fading one would put a hole at exactly the
    join the crossfade just repaired.
    """

    data = np.asarray(signal, dtype=float)
    if len(data) == 0:
        return []
    drive = 1.25
    data = np.tanh(data * drive) / math.tanh(drive)
    if not loop:
        data = _fades(data, 0.004)
    peak = float(np.max(np.abs(data)))
    if peak > 1e-12:
        data = data * (10.0 ** (peak_db / 20.0) / peak)
    return [float(x) for x in np.clip(data, -0.98, 0.98)]


def scale_named(name: str) -> Scale:
    if name not in SCALES:
        raise ValueError(
            f"unknown scale {name!r}; expected one of {sorted(SCALES)}"
        )
    return SCALES[name]


def bundled_sources() -> tuple[str, ...]:
    return asset_names(SOURCE_DIRECTORY)
