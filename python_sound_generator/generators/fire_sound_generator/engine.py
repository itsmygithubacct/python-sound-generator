"""Fire, built from real CC0 recordings.

A hybrid generator in the lineage of ``door_sound_generator``,
``creature_foley_generator``, and ``unarmed_foley_generator``: NumPy and SciPy
trim, retune, filter, and layer bundled recordings.

Why this is not synthesized
---------------------------
``environment_ambience_generator`` had a fire, and it was a careful one: a
breathing roar, a hiss, a sparse irregular crackle, resonant pops, an "embers to
bonfire" intensity axis. It was listened to on 2026-07-15 and rejected outright,
which is why this generator exists and why that one no longer ships fire.

That is the fifth time this project has reached the same conclusion -- after the
door, the creature vocalisations, the aircraft touchdown and crash, and the
flesh impacts -- and fire is the clearest case of it. Procedural DSP wins on
anything with a rate and loses on chaotic transients. Fire has no rate. It is
turbulent combustion: broadband, non-stationary, correlated across frequency in
ways filtered noise is not. The parts you would expect to synthesize well
(a crackle is a train of tiny transients, and a roar is broadband noise) are
exactly the parts that gave it away, because a real crackle's pops are *not*
independent -- they come in bursts as a pocket fails, and each one rings the
same log.

So: everything sustained here is the same 29 s of wood burning, retuned and
filtered. Nothing is oscillators.

Loops
-----
The bed is a recording, so a loop cannot be seamless by construction the way a
synthesized one can. It is made seamless by an equal-power crossfade of the tail
into the head, and then *measured* rather than asserted -- with
``environment_ambience_generator``'s seam metric, imported rather than
reimplemented, so both generators agree on what seamless means.

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

BED = "pagdev_fireplace.wav"
FIREBALL = "julien_matthey_fireball.wav"
BLASTS: tuple[str, ...] = (
    "rubberduck_bang_02.wav",
    "rubberduck_bang_05.wav",
    "rubberduck_bang_08.wav",
    "rubberduck_cannon_01.wav",
    "rubberduck_cannon_03.wav",
)
MUD: tuple[str, ...] = tuple(
    f"rubberduck_mud_{index:02d}.wav" for index in range(1, 26)
)


@dataclass(frozen=True)
class Scale:
    """How big the fire is, and what that does to the recording.

    `speed` is varispeed and does the real work. A small flame is not a big one
    turned down: it is faster and higher, because the eddies are smaller. A
    torch flutters where a bonfire breathes, and resampling moves the rate and
    the pitch together, which is what keeps that true.
    """

    name: str
    label: str
    description: str
    speed: float
    weight: float
    brightness: float
    flutter: float


SCALES: dict[str, Scale] = {
    "ember": Scale("ember", "Ember", "dying coals, barely alive",
                   0.92, 0.10, -0.10, 0.05),
    "torch": Scale("torch", "Torch", "handheld, small and restless",
                   1.55, 0.14, 0.30, 0.85),
    "campfire": Scale("campfire", "Campfire", "a contained open fire",
                      1.00, 0.55, 0.05, 0.20),
    "bonfire": Scale("bonfire", "Bonfire", "big, breathing, and low",
                     0.74, 1.10, -0.14, 0.42),
    "inferno": Scale("inferno", "Inferno", "a building going up",
                     0.60, 1.60, -0.20, 0.55),
}




@lru_cache(maxsize=32)
def _load_source(filename: str) -> np.ndarray:
    path = resolve_asset(SOURCE_DIRECTORY / filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing bundled fire source {path}; see sources/provenance.json"
        )
    _, signal = read_recording(path, expected_rate=SR, label="fire source")
    return signal


def _trim_to_onset(signal: np.ndarray) -> np.ndarray:
    return trim_recording_activity(
            signal, sample_rate=SR, window_seconds=None,
            threshold_ratio=0.04, pre_seconds=0.004, post_seconds=None,
        )


def _varispeed(signal: np.ndarray, speed: float) -> np.ndarray:
    return varispeed_configured(
        signal, speed, minimum=0.25, maximum=4.0, denominator=512
    )


def _tilt(signal: np.ndarray, weight: float, brightness: float) -> np.ndarray:
    """Add mass and top end using bands of the recording itself."""

    result = signal
    if abs(weight) > 1e-6:
        low = sosfilt(butter(2, 220.0, btype="lowpass", fs=SR, output="sos"),
                      signal)
        result = result + weight * low
    if abs(brightness) > 1e-6:
        high = signal - sosfilt(
            butter(2, 2_400.0, btype="lowpass", fs=SR, output="sos"), signal
        )
        result = result + brightness * high
    return result


def _fades(signal: np.ndarray, seconds: float=0.004) -> np.ndarray:
    return fade_edges(signal, seconds, sample_rate=SR)


def _frames(seconds: float) -> int:
    return max(1, round(seconds * SR))


def _tile(signal: np.ndarray, frames: int) -> np.ndarray:
    return tile_to_frames(signal, frames)


def _crossfade_loop(signal: np.ndarray, frames: int, fade: float=1.2) -> np.ndarray:
    return equal_power_crossfade_loop(
        signal, frames, fade, sample_rate=SR
    )


def _flutter(frames: int, rng: np.random.Generator, depth: float,
             rate: float) -> np.ndarray:
    """A slow irregular amplitude wander, quantised to whole cycles.

    Quantised so the modulator itself does not break the loop: an LFO that does
    not close on a whole cycle swells across the wrap, which the seam metric
    catches as level drift rather than as a click.
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


def _choose(rng: np.random.Generator, choices: Sequence[str]) -> str:
    return choices[int(rng.integers(0, len(choices)))]


def _mix_at(destination: np.ndarray, source: np.ndarray, start_frame: int, gain: float) -> None:
    mix_at_frames(destination, source, start_frame, gain)


@lru_cache(maxsize=64)
def _retuned_bed(scale: Scale, speed_multiplier: float = 1.0) -> np.ndarray:
    """The whole fireplace, retuned and tilted to a scale.

    Deliberately the *whole* 29 s and not a window of it: any filtering a cue
    wants has to happen here, on the full recording, before `_window` cuts into
    it. Filtering after the cut would start the filter cold at the head of the
    loop while its tail carried full state, which is a swell at the join --
    exactly the failure environment_ambience_generator primes its filters
    circularly to avoid. Filter here and the transient stays at t=0 of material
    the window never reaches.
    """
    signal = _varispeed(_load_source(BED), scale.speed * speed_multiplier)
    signal = _tilt(signal, scale.weight, scale.brightness)
    peak = float(np.max(np.abs(signal)))
    signal = signal / peak if peak > 1e-9 else signal
    signal.setflags(write=False)
    return signal


def _window(signal: np.ndarray, rng: np.random.Generator, seconds: float) -> np.ndarray:
    return random_window(signal, rng, seconds, sample_rate=SR)


def _bed(scale: Scale, rng: np.random.Generator, seconds: float,
         speed_multiplier: float = 1.0) -> np.ndarray:
    """A random window of the retuned fireplace. The root of every cue here."""

    return _window(_retuned_bed(scale, speed_multiplier), rng, seconds)


# --------------------------------------------------------------------------
# sustained cues
# --------------------------------------------------------------------------

def burning(scale: Scale, seconds: float,
            rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """A fire burning, as a seamless loop. Ember through inferno.

    This is `fire` under its honest name: the cue is the sound of something
    already alight, which is a bed and not an event.
    """
    frames = _frames(seconds)
    fade = 1.2
    bed = _crossfade_loop(_bed(scale, rng, seconds + fade), frames, fade)
    bed = bed * _flutter(frames, rng, scale.flutter * 0.30, 0.6)

    # A second window of the same recording, retuned and offset, so a short loop
    # does not repeat one recognisable log. Offset by a third so the two passes
    # do not line up on the wrap.
    second = _crossfade_loop(
        _bed(scale, rng, seconds + fade, speed_multiplier=1.31), frames, fade
    )
    out = bed + 0.55 * np.roll(second, frames // 3)
    return out, [BED]


def torch(scale: Scale, seconds: float,
          rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """A handheld flame: small, restless, and close.

    A torch is not a small campfire. It is carried, so the flame is dragged
    through air and flutters at a rate a fire on the ground never does; that
    flutter is the whole cue, and it is why this is not just `burning --scale
    torch`.

    The highpass runs on the full recording before the window is cut, not on the
    finished loop: see `_retuned_bed`. Cold-starting it after the cut put a
    filter transient at the head of the loop and full state at its tail, which
    measured as a broken seam on the bigger scales.
    """
    frames = _frames(seconds)
    fade = 1.2
    material = sosfilt(
        butter(2, 260.0, btype="highpass", fs=SR, output="sos"),
        _retuned_bed(scale),
    )
    bed = _crossfade_loop(_window(material, rng, seconds + fade), frames, fade)
    return bed * _flutter(frames, rng, 0.42, 1.9), [BED]


def flame(scale: Scale, seconds: float,
          rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """A pressurised jet: a flamethrower, a burner, a dragon's breath.

    Under pressure the fire stops breathing. A jet is forced, so the slow wander
    of an open fire is wrong for it: the bed is sped up until the turbulence
    reads as continuous, the low end is pulled out because there is no bulk of
    coals to make it, and what is left is roar and hiss.

    Both filters run before the window is cut, for the reason in `_retuned_bed`.
    """
    frames = _frames(seconds)
    fade = 1.2
    jet_material = sosfilt(
        butter(2, 180.0, btype="highpass", fs=SR, output="sos"),
        _retuned_bed(scale, 1.9),
    )
    jet_material = _tilt(jet_material, 0.0, 0.45)
    jet = _crossfade_loop(_window(jet_material, rng, seconds + fade), frames,
                          fade)

    # A jet's own hiss: the bed's top end, sped up further and laid under it.
    hiss_material = sosfilt(
        butter(2, 2_600.0, btype="highpass", fs=SR, output="sos"),
        _retuned_bed(scale, 3.1),
    )
    hiss = _crossfade_loop(_window(hiss_material, rng, seconds + fade), frames,
                           fade)
    out = jet + 0.5 * hiss
    return out * _flutter(frames, rng, 0.10, 3.2), [BED]


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------

def ignite(scale: Scale, rng: np.random.Generator,
           intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Catching light: the whoomp, and the fire that stays.

    Two parts. The whoomp is the fireball recording's attack and nothing after
    it -- ignition is that same event, just without a projectile to follow. The
    tail is the bed swelling in behind it, which is what makes it read as
    something catching rather than something detonating.
    """
    ball = _trim_to_onset(_load_source(FIREBALL))
    whoomp = _varispeed(ball[:_frames(0.42)], scale.speed * 1.15)
    whoomp = _tilt(whoomp, scale.weight * 0.6, 0.0)

    tail_frames = _frames(0.9)
    tail = _crossfade_loop(_bed(scale, rng, 0.9 + 0.3), tail_frames,
                           fade=0.3)
    swell = np.clip(np.linspace(0.0, 1.4, tail_frames), 0.0, 1.0) ** 1.6
    tail = tail * swell * np.exp(-np.arange(tail_frames) / SR / 0.55)

    total = max(len(whoomp), tail_frames) + _frames(0.05)
    out = np.zeros(total)
    _mix_at(out, whoomp, 0, intensity)
    _mix_at(out, tail, _frames(0.06), 0.7 * intensity)
    return _fades(out), [FIREBALL, BED]


def fireball(scale: Scale, rng: np.random.Generator,
             intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """A flame projectile: the throw, the body in flight, and the burn.

    The recording is the throw. What it does not have is *travel*, so the bed
    rides underneath it with a falling filter, which is what says the thing left
    the hand and went somewhere -- the same job the departure layer did in
    combat_generators' phaser blast, done with fire instead of a sweep.
    """
    ball = _trim_to_onset(_load_source(FIREBALL))
    throw = _varispeed(ball, scale.speed)
    throw = _tilt(throw, scale.weight * 0.8, scale.brightness)

    flight_frames = max(len(throw), _frames(0.85))
    body = _crossfade_loop(
        _bed(scale, rng, flight_frames / SR + 0.25, speed_multiplier=1.6),
        flight_frames, fade=0.25,
    )
    # receding: the top end goes first, as it does over distance
    t = np.arange(flight_frames) / SR
    body = sosfilt(butter(2, 3_200.0, btype="lowpass", fs=SR, output="sos"),
                   body)
    body = body * np.exp(-t / 0.42)

    total = flight_frames + _frames(0.05)
    out = np.zeros(total)
    _mix_at(out, throw, 0, intensity)
    _mix_at(out, body, _frames(0.02), 0.62 * intensity)
    return _fades(out), [FIREBALL, BED]


def fire_explosion(scale: Scale, rng: np.random.Generator,
                   intensity: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """Fuel going up: a blast, and then everything is on fire.

    The distinction from `weapons_destruction_generator`'s explosion is the
    tail, not the front. That one is shock, pressure, and fragments -- a shell.
    This is a whoomp with a fire behind it that outlives the bang by a second,
    which is what separates a petrol tank from an artillery round. The blast
    itself is a recording rather than a synthesized shock, because the thing
    that gives a synthesized explosion away is the same thing that gives a
    synthesized punch away.
    """
    blast_name = _choose(rng, BLASTS)
    blast = _trim_to_onset(_load_source(blast_name))
    blast = _varispeed(blast, scale.speed * 0.85)
    blast = _tilt(blast, scale.weight * 1.2, -0.05)

    ball = _trim_to_onset(_load_source(FIREBALL))
    whoomp = _varispeed(ball[:_frames(0.5)], scale.speed * 0.8)

    roar_frames = _frames(1.6)
    roar = _crossfade_loop(_bed(scale, rng, 1.6 + 0.3), roar_frames,
                           fade=0.3)
    t = np.arange(roar_frames) / SR
    roar = roar * np.clip(t / 0.10, 0.0, 1.0) * np.exp(-t / 0.85)

    total = max(len(blast), roar_frames) + _frames(0.05)
    out = np.zeros(total)
    _mix_at(out, whoomp, 0, 0.75 * intensity)
    _mix_at(out, blast, _frames(0.012), intensity)
    _mix_at(out, roar, _frames(0.05), 0.85 * intensity)
    return _fades(out), [blast_name, FIREBALL, BED]


def extinguish(scale: Scale, rng: np.random.Generator,
               intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Going out: the roar collapses and the steam outlives it.

    The steam is the bed's own top end with the roar filtered off it and a
    longer decay than the fire underneath. Nothing is added -- what is left when
    a fire dies is what was already in it, minus the low end that was the fire.
    """
    frames = _frames(1.1)
    bed = _crossfade_loop(_bed(scale, rng, 1.1 + 0.3), frames, fade=0.3)
    t = np.arange(frames) / SR

    collapse = bed * np.exp(-t / 0.16)
    steam = sosfilt(butter(2, 3_000.0, btype="highpass", fs=SR, output="sos"),
                    bed)
    steam = steam * np.exp(-t / 0.45)
    out = collapse + 0.8 * steam
    return _fades(out * intensity), [BED]


def lava(scale: Scale, rng: np.random.Generator,
         intensity: float = 0.95) -> tuple[np.ndarray, list[str]]:
    """Armoured body into lava: viscous splash, sizzle, and two bubbles.

    The body is real mud slowed to roughly 0.45x -- the CC0 source already
    auditioned for this job. Fire contributes only its top end as the sudden
    heat/armour hiss. Two more mud events, slowed and low-passed, arrive late
    enough to read as separate molten bubbles instead of one watery splash.

    The 0.85 s length is Flying Joust's: the game's `lava.wav` files are 850 ms and
    it plays this at 0.8x, so the render matches the shape it replaces rather
    than dropping a shorter clip into a longer slot. The sizzle settles across
    the tail instead of leaving it silent.
    """

    frames = _frames(0.85)
    out = np.zeros(frames)

    primary_name = _choose(rng, MUD)
    primary = _trim_to_onset(_load_source(primary_name))
    primary = _varispeed(primary, float(rng.uniform(0.43, 0.54)))
    primary = sosfilt(
        butter(2, 2_300.0, btype="lowpass", fs=SR, output="sos"), primary
    )
    primary = _tilt(primary, scale.weight * 0.50 + 0.35, -0.10)
    if len(primary) > frames:
        primary = primary[:frames]
    _mix_at(out, primary, 0, 0.95 * intensity)

    # Sudden sizzle and armour hiss from real fire, with the body filtered out.
    fire = _bed(scale, rng, 0.85, speed_multiplier=1.65)
    fire = sosfilt(
        butter(2, 2_100.0, btype="highpass", fs=SR, output="sos"), fire
    )
    t = np.arange(frames) / SR
    hiss_env = np.clip(t / 0.012, 0.0, 1.0) * np.exp(-t / 0.30)
    _mix_at(out, fire[:frames] * hiss_env, _frames(0.015), 0.72 * intensity)

    used = [primary_name, BED]
    available = tuple(name for name in MUD if name != primary_name)
    for when, gain in ((0.26, 0.54), (0.52, 0.42)):
        bubble_name = _choose(rng, available)
        available = tuple(name for name in available if name != bubble_name)
        bubble = _trim_to_onset(_load_source(bubble_name))
        bubble = _varispeed(bubble, float(rng.uniform(0.38, 0.48)))
        bubble = sosfilt(
            butter(3, 920.0, btype="lowpass", fs=SR, output="sos"), bubble
        )
        maximum = max(1, frames - _frames(when))
        if len(bubble) > maximum:
            bubble = bubble[:maximum]
        _mix_at(out, bubble, _frames(when), gain * intensity)
        used.append(bubble_name)

    return _fades(out), used


def lava_flow(scale: Scale, seconds: float,
              rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """Molten rock as a seamless bed: a slow, low, constant rumble.

    Not the splash of a body dropping in -- that is `lava` -- but the pit
    itself, always there. The fire bed's low end is slowed to roughly 0.6x
    until the flames read as a thick churn rather than a crackle; a slower,
    lower sub sits under it; and a few molten bubbles surface at random,
    wrapped around the loop so none is clipped at the seam.

    Much of the rumble lives below where a small speaker reproduces it -- the
    bubbles are what carries on a laptop, the rumble what a game with a real
    low end adds -- and both are meant to be present. Signed off by ear on
    2026-07-16 as the looping half of Flying Joust's lava, paired with the `lava`
    one-shot the game already triggers.
    """
    frames = _frames(seconds)
    fade = 1.2

    churn = sosfilt(
        butter(2, 340.0, btype="lowpass", fs=SR, output="sos"),
        _retuned_bed(scale, 0.6),
    )
    bed = _crossfade_loop(_window(churn, rng, seconds + fade), frames, fade)
    bed = bed * _flutter(frames, rng, 0.14, 0.22)

    sub = _crossfade_loop(
        sosfilt(butter(2, 120.0, btype="lowpass", fs=SR, output="sos"),
                _retuned_bed(scale, 0.5)),
        frames, fade,
    )
    out = bed * 0.9 + sub * 0.5

    used = [BED]
    available = list(MUD)
    rng.shuffle(available)
    for index in range(3):
        name = available[index % len(available)]
        bubble = _varispeed(_trim_to_onset(_load_source(name)),
                            float(rng.uniform(0.30, 0.42)))
        bubble = sosfilt(
            butter(3, 640.0, btype="lowpass", fs=SR, output="sos"), bubble
        )
        at = int(rng.integers(0, frames))
        buf = np.zeros(frames + len(bubble))
        buf[at:at + len(bubble)] += bubble
        # Send the tail that overhangs the loop back to the head, so a bubble
        # that surfaces near the seam is continuous across it.
        buf[:len(bubble)] += buf[frames:frames + len(bubble)]
        out += buf[:frames] * 0.28
        used.append(name)
    return out, used


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
