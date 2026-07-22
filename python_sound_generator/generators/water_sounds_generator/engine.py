"""Water, built from real CC0 recordings.

A hybrid generator in the lineage of ``door_sound_generator``,
``creature_foley_generator``, ``unarmed_foley_generator``, ``fire_sound_generator``
and ``stone_noise_generator``: NumPy and SciPy trim, retune, filter and layer
bundled recordings rather than synthesizing water from noise.

Why this is not synthesized
---------------------------
``environment_ambience_generator`` had six carefully designed water cues. All
six were auditioned on 2026-07-15 before this recording-driven generator was
written. **Five of the six were rejected.**

The one that matters most is `drip`, because it is the cleanest refutation this
project has of the rule it has been leaning on. A drip *has a rate* -- it is a
decaying bounce train, the textbook case for synthesis, which is exactly why item
4 modelled it that way. The verdict was "a ball harder than a tennis ball
bouncing off of cement". Not a poor drip: a different object. The rate was right
and the sound was wrong, which is item 9's trap stated precisely.

The one that survived is `ocean` -- the *most* chaotic cue of the six, broadband
turbulent surf with no rate worth naming. So in a single sitting the coarse rule
("procedural wins on anything periodic and loses on chaotic transients") called
both extremes backwards.

What did call all six correctly is ``vehicle_foley_generator``'s reading, offered
there as a caution rather than a rule:

    what decided it was not whether the sound has a rate, but whether the ear
    holds a precise template for it. Tread noise is a texture nobody can identify
    closely, so a plausible band passes. A tire squeal is a voice everyone knows
    on contact.

Nobody holds a template for one particular surf, so any plausible broadband wash
*is* surf. Everybody holds one for a drip, a splash, a bubble, a fish tank and
rain -- all heard a thousand times from a foot away -- and nothing plausible will
do. That is the question to ask of a new cue here: **has the player heard this
from a foot away?**

What this generator owns, and what it does not
----------------------------------------------
- **Taken from item 4**, all five rejected: `splash`, `drip`, `aquarium`,
  `bubbles`, `rain`. Item 9 took fire from that generator on the same grounds and
  in the same way.
- **Left with item 4**: `ocean`. It passed. It is also the cue hardest to better
  from recordings -- its 90 s default exists so its wave sets do not repeat
  audibly, and nothing on this machine is 90 s of CC0 surf.
- **Left with item 11**: `water` as an *impact material*, a projectile arriving.
  That is the event side of the material/event cut and it is unchanged.

Mass is partly recorded, which was a surprise
---------------------------------------------
Item 10's standing lesson is that library recordings are of small objects and
have nothing below ~80 Hz, so weight must be synthesised. Item 13 already found
one exception; this is the second, in the same archive that had been sitting on
this machine for a day.

The splash bank is not fifteen takes of one event. Measured, it spans a real mass
range: ``splash_08`` peaks at 8.6 kHz (a droplet), ``splash_04`` peaks at **92 Hz**
with 86 % of its energy below 500 Hz (a body going in). So `Mass` picks a *family*
first and varispeeds second, and the synthetic sub does very little -- there is
already a recording of something heavy hitting water.

Everything returns unmastered mono floating-point PCM plus the source files it
consumed. The cue registry owns mastering levels, so every public render follows
the same contract, and the consumed filenames reach the manifest.
"""

from __future__ import annotations

from ..._audio.assets import asset_names, resolve_asset

from ..._audio.foley_numpy import read_recording

from ..._audio.foley_numpy import write_wav_scipy as write_wav  # noqa: F401

from ..._audio.foley_numpy import trim_activity as trim_recording_activity

from ..._audio.foley_numpy import equal_power_crossfade_loop, mix_wrapped_frames, random_window, tile_to_frames

from ..._audio.foley_numpy import fade_edges, mix_at_frames, varispeed_configured


import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import butter, iirpeak, lfilter, sosfilt

SR = 44_100

# Two source directories, and the split is deliberate rather than untidy.
#
# Splash and bubble recordings are already bundled in projectile_sounds/sources,
# so this provider consumes that vetted bank rather than duplicating it.
#
# The six loop beds are bundled locally because no other runtime provider owns
# them. Their origin is recorded in sources/provenance.json.
#
# This is a compromise and it should be read as one. A cross-package read couples
# this generator to item 11's layout, which the material/event cut says should be
# the other way round -- water is a material, and a material generator should own
# its recordings while an event generator consumes them. Reconciling that is
# a larger source-bank consolidation concern rather than a rendering decision.
LOCAL_SOURCES = Path(__file__).resolve().parent / "sources"
SHARED_SOURCES = (
    Path(__file__).resolve().parent.parent / "projectile_sounds" / "sources"
)

# Entries, by what the recording is of. The bank spans a genuine mass range and
# these three families are that range, measured rather than guessed: the split
# points are spectral centroid 3.8 kHz and a low-frequency fraction of 0.3.
ENTRY_LIGHT: tuple[str, ...] = (
    "rubberduck_splash_01.wav",   # centroid 5838, peaks 6.1 kHz
    "rubberduck_splash_06.wav",   # centroid 5545, peaks 6.1 kHz
    "rubberduck_splash_08.wav",   # centroid 6279, peaks 8.6 kHz -- the smallest
    "rubberduck_splash_13.wav",
    "rubberduck_splash_14.wav",
    "rubberduck_splash_15.wav",
)
ENTRY_MID: tuple[str, ...] = (
    "rubberduck_splash_02.wav",
    "rubberduck_splash_05.wav",
    "rubberduck_splash_10.wav",
    "rubberduck_splash_11.wav",
    "rubberduck_splash_12.wav",
)
ENTRY_HEAVY: tuple[str, ...] = (
    "rubberduck_splash_03.wav",   # 0.55 below 500 Hz, peaks 132 Hz
    "rubberduck_splash_04.wav",   # 0.86 below 500 Hz, peaks 92 Hz -- the largest
    "rubberduck_splash_07.wav",
    "rubberduck_splash_09.wav",
)
ENTRY_SOURCES = ENTRY_LIGHT + ENTRY_MID + ENTRY_HEAVY

BUBBLE_SOURCES: tuple[str, ...] = (
    "rubberduck_bubble_01.wav",
    "rubberduck_bubble_02.wav",
    "rubberduck_bubble_03.wav",
)

# The beds nobody had unpacked. Local.
WATER_BEDS: tuple[str, ...] = (
    "rubberduck_loop_water_01.wav",
    "rubberduck_loop_water_02.wav",
    "rubberduck_loop_water_03.wav",
)
BUBBLE_BEDS: tuple[str, ...] = (
    "rubberduck_loop_bubbles_01.wav",
    "rubberduck_loop_bubbles_02.wav",
)
RAIN_BED = "rubberduck_loop_rain.wav"

LOCAL_NAMES = WATER_BEDS + BUBBLE_BEDS + (RAIN_BED,)


@dataclass(frozen=True)
class Mass:
    """How much of something goes in, and what that does to the recording.

    `sources` comes first and `speed` second, which is the opposite of every
    other generator here and is why this registry looks different. Elsewhere mass
    is varispeed because the library only ever recorded small objects. Water is
    the exception: this bank contains an actual recording of something heavy
    going in (``splash_04``, peaking at 92 Hz) *and* an actual droplet
    (``splash_08``, peaking at 8.6 kHz). Picking the right recording is worth more
    than any amount of resampling the wrong one, and varispeed then does the
    in-between.

    `spread` is how much water is thrown *after* the entry -- the difference
    between a pellet, which makes a hole that closes, and a body, which makes a
    hole and then a sheet of water that comes back down.
    """

    name: str
    label: str
    description: str
    sources: tuple[str, ...]
    speed: float
    spread: float
    weight: float
    sub_hz: float


MASSES: dict[str, Mass] = {
    "droplet": Mass("droplet", "Droplet", "a single drop: a hole that closes at once",
                    ENTRY_LIGHT, 2.10, 0.00, 0.00, 220.0),
    "pellet": Mass("pellet", "Pellet", "food, a pebble, a coin",
                   ENTRY_LIGHT, 1.45, 0.10, 0.00, 170.0),
    "fish": Mass("fish", "Fish", "a fish breaking the surface, a dart, a hook",
                 ENTRY_MID, 1.15, 0.25, 0.10, 130.0),
    "body": Mass("body", "Body", "a person or a creature going in",
                 ENTRY_MID + ENTRY_HEAVY, 0.88, 0.65, 0.45, 84.0),
    "boat": Mass("boat", "Boat", "a hull landing: displacement, not penetration",
                 ENTRY_HEAVY, 0.66, 0.90, 0.80, 58.0),
    "shell": Mass("shell", "Shell", "artillery: a column, then everything falls back",
                  ENTRY_HEAVY, 0.52, 1.30, 1.10, 44.0),
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _resolve(filename: str) -> Path:
    """Find a recording in either directory, local first.

    The error when this fails names both places and says why there are two,
    because the failure mode is a stripped or moved ``projectile_sounds`` and the
    next person should not have to read this module to understand it.
    """
    local = resolve_asset(LOCAL_SOURCES / filename)
    if local.is_file():
        return local
    shared = resolve_asset(SHARED_SOURCES / filename)
    if shared.is_file():
        return shared
    raise FileNotFoundError(
        f"missing water source {filename!r}: not in {LOCAL_SOURCES} and not in "
        f"{SHARED_SOURCES}. This generator bundles its six loop beds locally and "
        f"reads the splashes and bubbles from projectile_sounds to avoid "
        f"duplicating the vetted bank. If that provider moved or was stripped, "
        f"restore it or bundle the 18 files here and update provenance.json."
    )




def _trim_to_onset(signal: np.ndarray) -> np.ndarray:
    return trim_recording_activity(
            signal, sample_rate=SR, window_seconds=0.004,
            threshold_ratio=0.05, pre_seconds=0.004, post_seconds=0.010,
        )


@lru_cache(maxsize=48)
def _load_source(filename: str, trim: bool=True) -> np.ndarray:
    path = _resolve(filename)
    _, signal = read_recording(
        path, expected_rate=SR,
        trim=(lambda data, rate: _trim_to_onset(data)) if trim else None,
        label="water source",
    )
    return signal


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def _frames(seconds: float) -> int:
    return max(1, round(seconds * SR))


def _varispeed(signal: np.ndarray, speed: float) -> np.ndarray:
    return varispeed_configured(
        signal, speed, minimum=0.25, maximum=4.0, denominator=512
    )


def _shelf(signal: np.ndarray, gain_db: float, corner: float,
           high: bool) -> np.ndarray:
    """One RBJ shelving biquad. Lifted from stone_noise_generator._shelf."""

    if abs(gain_db) < 0.05 or len(signal) == 0:
        return signal
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * max(20.0, min(corner, SR * 0.45)) / SR
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / 2.0 * math.sqrt(2.0)
    root = 2.0 * math.sqrt(amp) * alpha
    sign = 1.0 if high else -1.0
    b = np.array([
        amp * ((amp + 1.0) + sign * (amp - 1.0) * cos_w0 + root),
        -2.0 * sign * amp * ((amp - 1.0) + sign * (amp + 1.0) * cos_w0),
        amp * ((amp + 1.0) + sign * (amp - 1.0) * cos_w0 - root),
    ])
    a = np.array([
        (amp + 1.0) - sign * (amp - 1.0) * cos_w0 + root,
        2.0 * sign * ((amp - 1.0) - sign * (amp + 1.0) * cos_w0),
        (amp + 1.0) - sign * (amp - 1.0) * cos_w0 - root,
    ])
    return lfilter(b / a[0], a / a[0], signal)


def _lowpass(signal: np.ndarray, cutoff: float) -> np.ndarray:
    if len(signal) == 0 or cutoff >= SR * 0.45:
        return signal
    return sosfilt(butter(4, max(60.0, cutoff), btype="low", fs=SR,
                          output="sos"), signal)


def _highpass(signal: np.ndarray, cutoff: float) -> np.ndarray:
    if len(signal) == 0 or cutoff <= 20.0:
        return signal
    return sosfilt(butter(2, cutoff, btype="high", fs=SR, output="sos"), signal)


def _decay(signal: np.ndarray, seconds: float, hold: float = 0.0) -> np.ndarray:
    if seconds <= 0.0 or len(signal) == 0:
        return signal
    t = np.arange(len(signal)) / SR
    return signal * np.exp(-np.maximum(0.0, t - hold) / max(seconds, 1e-4))


def _fades(signal: np.ndarray, seconds: float=0.004) -> np.ndarray:
    return fade_edges(signal, seconds, sample_rate=SR)


def _choose(rng: np.random.Generator, choices: Sequence[str]) -> str:
    return choices[int(rng.integers(0, len(choices)))]


def _mix_at(destination: np.ndarray, source: np.ndarray, start_frame: int, gain: float) -> None:
    mix_at_frames(destination, source, start_frame, gain)


def _mix_wrapped(destination: np.ndarray, source: np.ndarray, start_frame: int, gain: float) -> None:
    mix_wrapped_frames(destination, source, start_frame, gain)


def _tile(signal: np.ndarray, frames: int) -> np.ndarray:
    return tile_to_frames(signal, frames)


def _crossfade_loop(signal: np.ndarray, frames: int, fade: float=0.8) -> np.ndarray:
    return equal_power_crossfade_loop(
        signal, frames, fade, sample_rate=SR
    )


def _window(signal: np.ndarray, rng: np.random.Generator, seconds: float) -> np.ndarray:
    return random_window(signal, rng, seconds, sample_rate=SR)


def _flutter(frames: int, rng: np.random.Generator, depth: float,
             rate: float) -> np.ndarray:
    """A slow irregular amplitude wander, quantised to whole cycles per loop."""

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


# How loud the synthesised weight sits under a recorded entry.
#
# Lower than stone's 0.28, and for a reason this bank earned: `splash_04` peaks at
# 92 Hz and carries 86 % of its energy below 500 Hz. There is already a recording
# of something heavy hitting water, so the resonator is filling a much smaller
# hole than it was in stone -- and stacking a sub under a recording that already
# has one is the mistake stone made twice and measured both times.
_SUB_LEVEL = 0.18


def sub(mass: Mass, rng: np.random.Generator, seconds: float,
        decay: float = 0.12) -> np.ndarray:
    """The weight, synthesised: noise through a resonance, never a sine.

    Returns silence for `droplet` and `pellet`, whose weight is 0.0. A drop has no
    low end and a generator that gives it one is describing a boulder.
    """
    frames = _frames(seconds)
    if mass.weight <= 1e-6:
        return np.zeros(frames)
    noise = rng.standard_normal(frames)
    noise = _lowpass(noise, 300.0)
    t = np.arange(frames) / SR
    drive = noise * np.exp(-t / max(decay, 1e-4)) * (1.0 - np.exp(-t / 0.002))
    b, a = iirpeak(max(18.0, mass.sub_hz), 4.0, fs=SR)
    resonant = lfilter(b, a, drive)
    peak = float(np.max(np.abs(resonant)))
    if peak > 1e-9:
        resonant = resonant / peak
    return resonant * mass.weight


def _prepare(filename: str, mass: Mass, rng: np.random.Generator,
             speed_multiplier: float = 1.0) -> np.ndarray:
    cached = _load_source(filename)
    speed = mass.speed * speed_multiplier * float(rng.uniform(0.93, 1.07))
    signal = _varispeed(cached, speed)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal = signal / peak
    return signal


# --------------------------------------------------------------------------
# layers -- public, for item 10 and anything else composing water
# --------------------------------------------------------------------------

def entry(mass: Mass, rng: np.random.Generator,
          intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """The moment of contact: a hole punched in a surface.

    One recording, chosen from the family that *is* this mass, rather than one
    recording varispeeded to pretend. See the `Mass` docstring.
    """
    name = _choose(rng, mass.sources)
    signal = _prepare(name, mass, rng)
    return signal * intensity, [name]


def spray(mass: Mass, rng: np.random.Generator,
          seconds: float, intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """What comes back down: the sheet, the droplets, the surface closing.

    Built from the *light* entries regardless of `mass`, because that is what
    falling water is at any scale -- a shell's column comes down as a thousand
    small splashes, not as one big one. Scaled by `mass.spread`, which is a
    separate axis from mass for the same reason item 6 needed weight separate
    from mass: a boat displaces a lot of water without penetrating, and a bullet
    penetrates without displacing.
    """
    frames = _frames(seconds)
    out = np.zeros(frames)
    sources: list[str] = []
    if mass.spread <= 1e-6:
        return out, sources
    count = max(1, int(round(4 + 26 * mass.spread * intensity)))
    for _ in range(count):
        name = _choose(rng, ENTRY_LIGHT)
        drop = _prepare(name, mass, rng,
                        speed_multiplier=float(rng.uniform(1.3, 2.6)) / max(
                            mass.speed, 0.3))
        drop = _highpass(drop, 700.0)
        drop = _decay(drop, float(rng.uniform(0.04, 0.16)))
        start = 0.06 + float(rng.random()) ** 0.8 * seconds * 0.55
        _mix_at(out, drop, _frames(start),
                float(rng.uniform(0.06, 0.26)) * mass.spread * intensity)
        if name not in sources:
            sources.append(name)
    return out, sources


def body(mass: Mass, rng: np.random.Generator, seconds: float,
         intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """The water itself moving: the bed, gated to an event.

    A `loop_water` bed windowed and enveloped rather than a filtered noise burst.
    It is the layer that says the hole is *in water* and not in gravel, and it is
    the one an entry sounds thin without.
    """
    frames = _frames(seconds)
    name = _choose(rng, WATER_BEDS)
    bed = _window(_load_source(name, trim=False), rng, seconds)
    bed = _varispeed(bed, mass.speed * 1.1)
    bed = _tile(bed, frames)
    t = np.arange(frames) / SR
    envelope = np.clip(t / 0.02, 0.0, 1.0) * np.exp(-t / (0.10 + 0.5 * mass.spread))
    return bed * envelope * intensity * 0.7, [name]


# --------------------------------------------------------------------------
# cues -- events
# --------------------------------------------------------------------------

def splash(mass: Mass, rng: np.random.Generator,
           intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Something entering water. Replaces item 4's, which was rejected by ear.

    Contact leads, the water body arrives under it, and the spray outlives both --
    which is the order it happens in, and the order item 6 learned to respect when
    its touchdown put the regolith first and read as a dropped bag of gravel.
    """
    seconds = 0.5 + 1.4 * mass.spread + 0.4 * intensity
    out = np.zeros(_frames(seconds))

    hit, sources = entry(mass, rng, intensity)
    _mix_at(out, hit, 0, 1.0)

    wash, wash_sources = body(mass, rng, min(seconds, 1.2), intensity)
    _mix_at(out, wash, _frames(0.004), 0.55 + 0.3 * mass.spread)

    drops, drop_sources = spray(mass, rng, seconds, intensity)
    _mix_at(out, drops, 0, 1.0)

    weight = sub(mass, rng, min(seconds, 0.6),
                 decay=0.05 + 0.06 * mass.weight)
    _mix_at(out, weight, _frames(0.003), _SUB_LEVEL * (0.6 + 0.3 * intensity))

    for name in wash_sources + drop_sources:
        if name not in sources:
            sources.append(name)
    return _fades(out), sources


# There is no `drip` cue, and its absence is a decision made by ear on
# 2026-07-15, not an omission.
#
# There is no CC0 recording of a drip anywhere on this machine. A drip was built
# anyway, from the brightest small splash in the bank (``splash_08``, 8.6 kHz)
# varispeeded up and given a real reflected tail, on the theory that a plausible
# room on a real droplet might get there. It was auditioned against the same
# question item 4's synthesized drip failed -- "is this a drip, or a tick?" -- and
# it failed the same way. The verdict was: make it absent.
#
# That is the right call and the one this repository has made five times before.
# A drip is a voice everyone knows from a foot away -- the plink of a cavity
# resonance as the surface closes over -- and neither a modelled bounce (item 4)
# nor a retuned splash (this) is that sound. item 8 left ``fireball`` empty rather
# than ship a phaser, and item 2 removed its impacts rather than ship a bell; a
# game falls back to its own built-in for an absent cue, so an absent cue is a
# working bank and a wrong one is not.
#
# What would fill it: one real CC0 drip recording. rubberduck's water archive does
# not contain one. Freesound does, per-sound-licensed, behind a login -- audition
# the public preview first (see sources/provenance.json). Until then, absent.
#
# The renderer is kept as a raising stub rather than deleted so that anyone who
# reaches for `engine.drip` finds the reason rather than an AttributeError.
def drip(*_args, **_kwargs) -> tuple[np.ndarray, list[str]]:
    raise NotImplementedError(
        "drip was auditioned on 2026-07-15 and rejected: with no CC0 drip "
        "recording on this machine, a retuned splash reads as a tick, not a "
        "drip. It is deliberately absent rather than approximated. See the "
        "comment above this stub and sources/provenance.json."
    )


# There is no `surface` cue. It was built -- the bed alone, swelling and falling,
# with the entry transient filtered off, meaning to be a ripple or a lap or a fish
# rolling -- and auditioned on 2026-07-15. The verdict was "sounded like nothing,
# vague". That is the correct verdict and the cue is dropped rather than tuned:
# a disturbance with no entry and no debris has no event in it for the ear to hold
# onto, and every plausible rendering of "water moving, quietly" is the same
# ambiguous wash. What reads as water being disturbed is water being *entered* --
# which is `splash` at a light mass -- or water being *swum through*, which needs a
# body and a rhythm this could not supply. Not deleted-and-forgotten: it is in
# NOT_MADE with the reason, so it is not re-attempted by the next person who
# notices the gap.


def bubble(mass: Mass, rng: np.random.Generator,
           intensity: float = 0.8) -> tuple[np.ndarray, list[str]]:
    """Bubbles as an event: a single pop, or a burst of them.

    `intensity` is how many. One bubble is a pop; a dozen is a gulp, a vent, a
    thing sinking. Item 4 owned bubbles only as a bed, which cannot do either.
    """
    seconds = 0.4 + 1.0 * intensity
    frames = _frames(seconds)
    out = np.zeros(frames)
    sources: list[str] = []
    count = max(1, int(round(1 + 9 * intensity)))
    for index in range(count):
        name = _choose(rng, BUBBLE_SOURCES)
        pop = _prepare(name, mass, rng,
                       speed_multiplier=float(rng.uniform(0.75, 1.7)))
        pop = _decay(pop, float(rng.uniform(0.08, 0.30)))
        start = float(rng.random()) ** 0.85 * seconds * 0.7 if index else 0.0
        _mix_at(out, pop, _frames(start),
                float(rng.uniform(0.5, 1.0)) * intensity)
        if name not in sources:
            sources.append(name)
    return _fades(out), sources


def pour(mass: Mass, rng: np.random.Generator,
         intensity: float = 0.8) -> tuple[np.ndarray, list[str]]:
    """Water poured from a vessel: a stream landing in more water.

    The bed carries it and a scatter of small entries rides on top, because a
    pour is not a smooth flow -- it is a continuous train of tiny impacts, and
    the bed alone reads as a tap running into a sink that is already full.
    """
    seconds = 1.4 + 1.2 * intensity
    frames = _frames(seconds)
    name = _choose(rng, WATER_BEDS)
    bed = _tile(_varispeed(_window(_load_source(name, trim=False), rng, seconds),
                           1.25), frames)
    t = np.arange(frames) / SR
    envelope = (np.clip(t / 0.12, 0.0, 1.0)
                * np.clip((seconds - t) / 0.35, 0.0, 1.0))
    out = bed * envelope * intensity
    sources = [name]

    for _ in range(int(round(18 * intensity * seconds))):
        drop_name = _choose(rng, ENTRY_LIGHT)
        drop = _prepare(drop_name, mass, rng,
                        speed_multiplier=float(rng.uniform(1.6, 3.0)))
        drop = _highpass(drop, 800.0)
        drop = _decay(drop, float(rng.uniform(0.03, 0.10)))
        _mix_at(out, drop, _frames(float(rng.uniform(0.05, seconds * 0.88))),
                float(rng.uniform(0.05, 0.20)) * intensity)
        if drop_name not in sources:
            sources.append(drop_name)
    return _fades(out), sources


def wade(mass: Mass, rng: np.random.Generator,
         intensity: float = 0.8) -> tuple[np.ndarray, list[str]]:
    """One step through shallow water: in, through, out.

    Three events rather than one, and the last is the cue: water does not let go
    cleanly. The foot comes out with a suck and a run-off that outlasts the step,
    and without it this is `splash --mass fish` with extra latency.

    Deliberately *one step*. `footstep_generator` owns gait -- the rate, the
    stride, the surface registry -- and a water generator that grew a cadence
    would be a second footstep generator with worse coverage.
    """
    seconds = 0.9 + 0.5 * intensity
    frames = _frames(seconds)
    out = np.zeros(frames)

    down_name = _choose(rng, ENTRY_MID)
    down = _prepare(down_name, mass, rng, speed_multiplier=1.15)
    _mix_at(out, down, 0, intensity)
    sources = [down_name]

    wash, wash_sources = body(mass, rng, 0.5, intensity * 0.8)
    _mix_at(out, wash, _frames(0.01), 0.5)

    # The lift: quieter, later, and duller than the step that made it.
    up_name = _choose(rng, ENTRY_LIGHT)
    up = _prepare(up_name, mass, rng, speed_multiplier=1.5)
    up = _lowpass(up, 5_000.0)
    up = _decay(up, 0.22)
    _mix_at(out, up, _frames(0.28 + float(rng.uniform(0.0, 0.10))),
            0.55 * intensity)

    drops, drop_sources = spray(mass, rng, seconds, intensity * 0.5)
    _mix_at(out, drops, _frames(0.30), 0.7)

    for name in wash_sources + drop_sources + [up_name]:
        if name not in sources:
            sources.append(name)
    return _fades(out), sources


# --------------------------------------------------------------------------
# cues -- loops
# --------------------------------------------------------------------------

def aquarium(mass: Mass, seconds: float, rng: np.random.Generator,
             intensity: float = 0.5) -> tuple[np.ndarray, list[str]]:
    """Inside a fish tank: water body, airstone, pump. Replaces item 4's.

    This replacement passed its audition on 2026-07-15. Aquarium chose the
    quieter intermittent ``bubbler`` instead and now plays it through its
    in-process mixer; ``aquarium`` remains the general busy-tank model.

    Three layers, because a tank is three things and item 4's registry knew that;
    what it lacked was recordings. The water body and the airstone are beds. The
    pump is the one *deliberately* synthetic thing here and it is not a
    contradiction of this generator's thesis: a pump is a motor, it has a rate, it
    is periodic, and it is exactly the half of the split synthesis is good at.
    Nobody holds a template for one particular aquarium pump either -- it is a hum
    behind glass. It is noise through a resonance, never a sine.
    """
    frames = _frames(seconds)
    fade = 0.8

    water_name = _choose(rng, WATER_BEDS)
    water = _crossfade_loop(
        _varispeed(_window(_load_source(water_name, trim=False), rng,
                           seconds + fade + 1.0), 0.82),
        frames, fade)
    water = _lowpass(water, 5_000.0)
    out = water * (0.55 + 0.25 * intensity)

    # A second bed at a different speed, offset, so a short loop does not repeat
    # one recognisable gurgle. The beds are 3.4-7.3 s and the loop is usually
    # longer, so tiling is unavoidable -- this is what hides it.
    other = _choose(rng, tuple(n for n in WATER_BEDS if n != water_name))
    second = _crossfade_loop(
        _varispeed(_window(_load_source(other, trim=False), rng,
                           seconds + fade + 1.0), 1.19),
        frames, fade)
    out = out + 0.4 * np.roll(_lowpass(second, 4_000.0), frames // 3)

    stone_name = _choose(rng, BUBBLE_BEDS)
    airstone = _crossfade_loop(
        _window(_load_source(stone_name, trim=False), rng, seconds + fade + 1.0),
        frames, fade)
    airstone = _highpass(airstone, 300.0)
    out = out + airstone * (0.34 + 0.30 * intensity) * _flutter(
        frames, rng, 0.12, 0.4)

    # The pump. Periodic, so it is synthesised, and it wraps by construction:
    # a whole number of cycles per loop cannot drift across the join.
    t = np.arange(frames) / SR
    cycles = max(1, round(96.0 * seconds))
    noise = rng.standard_normal(frames)
    b, a = iirpeak(96.0, 9.0, fs=SR)
    hum = lfilter(b, a, _lowpass(noise, 700.0))
    peak = float(np.max(np.abs(hum)))
    if peak > 1e-9:
        hum = hum / peak
    hum = hum * (1.0 + 0.10 * np.sin(2.0 * math.pi * cycles * t / seconds))
    out = out + hum * 0.16 * intensity
    return out, [water_name, other, stone_name]


def bubbler(mass: Mass, seconds: float, rng: np.random.Generator,
            intensity: float = 0.45) -> tuple[np.ndarray, list[str]]:
    """An aquarium bubbler that pulses rather than runs: intermittent bubbles.

    The difference from `aquarium` and `bubbles` is the *silence between*. A quiet
    tank does not roar and it does not run a continuous column -- an air pump on a
    cheap timer, or an airstone with an uneven feed, lets go in bursts with plain
    water between them, and it is that rhythm the ear reads as "a tank sitting in
    the corner" rather than "a water feature". `aquarium` is the busy version;
    this is the one you forget is on.

    So the bed is deliberately *faint* -- a low, dull wash well under the bursts,
    present only so the gaps are water rather than dead air -- and the bursts are
    sparse: clusters of a few pops every few seconds, `intensity` setting how
    often. Everything is placed with `_mix_wrapped`, so a burst that starts near
    the end finishes at the head and the loop has no seam.

    `mass` is accepted for the loop-cue signature and unused: a tank's bubbles are
    a tank's bubbles at one size.
    """
    frames = _frames(seconds)
    fade = 0.8

    # A faint water presence in the gaps. Dull and quiet -- if this is audible as
    # anything but "not silence", it is too loud and the tank sounds busy.
    bed_name = _choose(rng, WATER_BEDS)
    bed = _crossfade_loop(
        _window(_load_source(bed_name, trim=False), rng, seconds + fade + 1.0),
        frames, fade)
    bed = _lowpass(bed, 2_400.0)
    out = bed * (0.06 + 0.06 * intensity)
    sources = [bed_name]

    # Bursts at intervals. Lower intensity means longer gaps: the mean spacing
    # runs ~2.4 s busy to ~5.5 s sparse. Placed on a wandering clock rather than a
    # fixed one, so no burst recurs on a period a listener could lock onto.
    mean_gap = 2.4 + 3.1 * (1.0 - intensity)
    position = float(rng.uniform(0.0, mean_gap))
    guard = 0
    while position < seconds and guard < 4096:
        guard += 1
        cluster = 1 + int(rng.integers(0, 4))  # 1-4 bubbles per burst
        for _ in range(cluster):
            pop_name = _choose(rng, BUBBLE_SOURCES)
            pop = _prepare(pop_name, mass, rng,
                           speed_multiplier=float(rng.uniform(0.7, 1.9)))
            pop = _decay(pop, float(rng.uniform(0.08, 0.32)))
            at = position + float(rng.uniform(0.0, 0.45))
            _mix_wrapped(out, pop, _frames(at),
                         float(rng.uniform(0.5, 1.0)) * (0.5 + 0.5 * intensity))
            if pop_name not in sources:
                sources.append(pop_name)
        position += mean_gap * float(rng.uniform(0.6, 1.5))

    # A very slow wander so even the quiet bed is not perfectly stationary.
    return out * _flutter(frames, rng, 0.05, 0.2), sources


def bubbles(mass: Mass, seconds: float, rng: np.random.Generator,
            intensity: float = 0.7) -> tuple[np.ndarray, list[str]]:
    """A close rising column of bubbles, as a bed. Replaces item 4's.

    The bed carries it and discrete pops ride on top at a rate, wrapped so they
    cross the loop point. A column is not a texture -- you can count the bubbles
    near the bottom -- and a bed alone reads as a hiss.
    """
    frames = _frames(seconds)
    fade = 0.8
    name = _choose(rng, BUBBLE_BEDS)
    bed = _crossfade_loop(
        _window(_load_source(name, trim=False), rng, seconds + fade + 1.0),
        frames, fade)
    out = bed * (0.7 + 0.3 * intensity)
    sources = [name]

    for _ in range(max(2, int(round(7 * intensity * seconds)))):
        pop_name = _choose(rng, BUBBLE_SOURCES)
        pop = _prepare(pop_name, mass, rng,
                       speed_multiplier=float(rng.uniform(0.8, 1.9)))
        pop = _decay(pop, float(rng.uniform(0.06, 0.22)))
        _mix_wrapped(out, pop, int(rng.integers(0, frames)),
                     float(rng.uniform(0.15, 0.5)) * intensity)
        if pop_name not in sources:
            sources.append(pop_name)
    return out * _flutter(frames, rng, 0.10, 0.5), sources


def rain(mass: Mass, seconds: float, rng: np.random.Generator,
         intensity: float = 0.6) -> tuple[np.ndarray, list[str]]:
    """Rain. Replaces item 4's, which was rejected by ear.

    One recording, and `intensity` is how hard it is coming down -- reached by
    filtering and speed rather than by adding synthetic drops, because a real
    rain's near drops are already in the take and a synthesised one arriving among
    them is the thing that gives it away. Heavier rain is faster and duller: more
    water, more distant hiss, less individual drop.

    **The bed is 6.8 s and it was heard looping in an 8 s render.** A single
    tiled recording repeats itself audibly the moment the loop is longer than the
    material, and rain is unforgiving of it: the ear locks onto one distinct near
    drop and hears it return on a clock. The fix is the one `aquarium` already
    uses -- a second copy of the same recording at a different speed, rolled to a
    non-commensurate offset, so the two passes never realign and no single drop
    recurs on a period. It is one recording still, layered against itself, which
    is why the source list has one entry.
    """
    frames = _frames(seconds)
    fade = 0.8
    material = _load_source(RAIN_BED, trim=False)
    material = _varispeed(material, 0.88 + 0.42 * intensity)
    material = _shelf(material, -7.0 * intensity, 2_200.0, high=True)
    bed = _crossfade_loop(_window(material, rng, seconds + fade + 1.0), frames,
                          fade)

    # The decorrelating layer: same take, ~19 % slower, offset by an irrational
    # fraction of the loop so it does not share a period with the first pass. Its
    # near drops land between the first pass's, which is what breaks the clock.
    second_src = _varispeed(material, 0.81)
    second = _crossfade_loop(_window(second_src, rng, seconds + fade + 1.0),
                             frames, fade)
    offset = int(frames * 0.41)
    out = bed + 0.85 * np.roll(second, offset)

    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out = out / peak
    return out * (0.6 + 0.4 * intensity) * _flutter(frames, rng, 0.06, 0.3), [
        RAIN_BED]


# There is no `swim` cue. It was built as a loop -- wrapped entries at a rate over
# a highpassed wake bed, meant to vary by mass so a fish and a boat swim
# differently -- and auditioned on 2026-07-15. The verdict was "too sparse,
# vague". Tuning the rate up does not fix it: the failure is that discrete splashes
# on a rate read as *someone throwing things into water*, not as a body moving
# through it, and the thing that separates the two is a stroke's continuous drag,
# which is a Foley performance rather than a scatter of entries. That is the
# `wade` boundary again -- gait belongs to a cadence this generator does not model
# -- and the honest scope is one step (`wade`), kept, and a swim, dropped. In
# NOT_MADE with the reason so it is not re-attempted blind.


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def master(signal: np.ndarray, peak_db: float,
           loop: bool = False) -> list[float]:
    """Final chain shared by every cue: soft limit, fade, normalize."""

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


def mass_named(name: str) -> Mass:
    if name not in MASSES:
        raise ValueError(
            f"unknown mass {name!r}; expected one of {sorted(MASSES)}"
        )
    return MASSES[name]


def local_sources() -> tuple[str, ...]:
    return asset_names(LOCAL_SOURCES)


def consumed_sources() -> tuple[str, ...]:
    """The files read out of projectile_sounds rather than bundled here."""

    return tuple(sorted(ENTRY_SOURCES + BUBBLE_SOURCES))


def bundled_sources() -> tuple[str, ...]:
    return tuple(sorted(local_sources() + consumed_sources()))
