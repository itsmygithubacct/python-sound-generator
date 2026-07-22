"""Stone, built from real CC0 recordings.

A hybrid generator in the lineage of ``door_sound_generator``,
``creature_foley_generator``, ``unarmed_foley_generator`` and
``fire_sound_generator``: NumPy and SciPy trim, retune, darken and layer bundled
recordings rather than synthesizing an impact from oscillators.

Why this is not synthesized
---------------------------
``weapons_destruction_generator`` had stone impacts and stone debris. They were
auditioned by ear on 2026-07-15 and removed; the retired comparison was
preserved outside the runtime package. That audit is the reason this generator
exists, and it is worth being precise about
what it did and did not show.

Eleven of twelve renders were rejected. Stone was the *single survivor* -- shell
into stone, on ``modal_ring`` + ``sine_track``. It would be a mistake to read
that as a licence to keep synthesizing stone. Its total mode gain is 0.42
against metal's 0.86, so its ring contributes almost nothing and noise carries
the cue: it passed by being the material that least resembled what the technique
actually does. Two finer theories were formed during that audit and both were
falsified -- that the pure-sine primitives were to blame (dirt is sine-free and
failed anyway) and that mode gain predicted the verdict (wood sits between stone
and metal and failed). One survivor in twelve is an outlier, not a pattern, and
a matrix with one synthesized cell and eleven recorded ones would be incoherent
besides. ``W2_impact_shell_stone.wav`` is kept as the reference a replacement has
to beat, not as a thing to preserve -- and ``projectile_sounds`` has since beaten
it by ear with a Foley stone of its own, which settles the outlier question. That
generator, not W2, is the bar this one has to clear.

What the rule actually says, coarsely, and it has resisted refinement seven
times now: procedural DSP wins on anything periodic and loses on chaotic
transients. Stone is the archetype of the losing half. A rock breaking is
contact, fracture and debris arriving milliseconds apart, each with its own
decay, and ``modal_ring`` sums decaying sinusoids, which is how a bell is
synthesised.

Read ``vehicle_foley_generator`` before trusting that rule, though. It predicted
both of that generator's verdicts backwards. The rule is a guide to where to
*look*, and never a substitute for listening.

What a material generator owns
------------------------------
The providers cut the same territory two ways. Crash and projectile generators
are *event*-centric -- a structure failing, a projectile arriving. Stone and
wood generators are *material*-centric. A bullet striking a plank belongs to
both domains.

The resolution this generator implements, from item 13:

- a **material** generator owns what a material *is*: its contact, its ring, its
  fracture, its debris. That is the natural unit of a recording library, and it
  is the unit ``sources/`` already stores.
- an **event** generator owns the *composition*: which materials, at what scale,
  in what order, with the sub-bass underneath.

So the four layer functions below -- `contact`, `fracture`, `debris`, `sub` --
are public, and they are the actual deliverable for items 10 and 11. The cues
are those layers composed into the events stone makes on its own account. An
event generator composing a *plank* and a *plate* into one crash should call the
layers and do its own composing; it should not call `shatter` and mix the result
under something, because `shatter` has already made composition decisions that
are only right for stone being the subject.

Scale, and why weight is not a separate axis here
-------------------------------------------------
Mass is varispeed: resampling moves pitch and decay together, which is what makes
a big object big. Pitch-shifting alone leaves the decay behind and yields a
slowed-down small object. Borrowed from item 5, confirmed by item 6.

Item 6 needed a second axis, because *weight* is not mass: a lander massing 0.9
barely varispeeds, so how much of the machine arrives on one contact patch had to
be modelled separately. Stone does not need that axis, and the reason is that a
stone is not a machine. A boulder is heavy on its contact patch *because* it is a
boulder -- there is no chassis distributing it -- so mass and weight are the same
number and `Scale` carries one. A cliff face is the one case where they part
company, and it is handled by the cue rather than the registry: `rockfall`
spreads its arrivals over many contacts instead of driving `sub` harder.

The low end, and a lesson that turned out to be false
-----------------------------------------------------
An earlier working rule said that library impacts of small objects have nothing
below roughly 80 Hz and therefore need weight synthesized underneath as noise
through a resonance, never a sine.

It is true of fifteen of the twenty recordings here -- under 15 % of their energy
below 500 Hz, centroids 1.3 to 7.1 kHz. It is **false** of the other five.
Kenney's ``impact_mining_*`` is a tool going into rock and it peaks at 59-98 Hz,
with 22-85 % of its energy below 80 Hz. That is a recorded stone low end, which
is the exact thing the lesson says has to be synthesised. See `contact`, which
layers the two families rather than replaying one, because measured, neither is a
struck rock on its own.

The synthetic weight stays, for two reasons that survive the correction: nothing
recorded here reaches a *cliff's* low end, and a pebble must have none at all. It
is noise through a resonance, never a sine, for the reason the module docstring of
``aircraft_spacecraft_generator.hybrid`` gives -- the procedural model earning its
place *under* the foley rather than being replaced by it, and the only synthetic
thing in this file. It simply does less than it was built to: `_SUB_LEVEL` fell
from 0.55 to 0.28 once the recorded body arrived, because stacking a resonator
under a real low end only doubles it.

Loops
-----
`grind` and `rockfall` are seamless **by construction** rather than by
crossfading: they are built by scattering grains at modular offsets, so a grain
landing near the end wraps to the head and the join is simply more of the same
material. ``fire_sound_generator`` had to crossfade because its bed is one
continuous 29 s recording that cannot be cut anywhere; these cues are made of
discrete contacts, which is the case where wrapping works. The seam is
*measured* rather than asserted, with ``environment_ambience_generator``'s
metric, imported by the CLI rather than reimplemented.

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

from ..._audio.foley_numpy import mix_wrapped_frames

from ..._audio.foley_numpy import fade_edges, mix_at_frames, varispeed_configured


import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import butter, iirpeak, lfilter, sosfilt

SR = 44_100
SOURCE_DIRECTORY = Path(__file__).resolve().parent / "sources"

# The four families, by what the recording is of rather than by what a cue wants.
# provenance.json records the same split, and it is why a twenty-file
# library covers as much as it does: between them, Kenney and rubberduck recorded
# stone being struck, stone touching, stone failing and stone landing as separate
# takes, which is exactly the decomposition a material generator needs.
#
# CONTACT and BODY are one event in two bands and they are always used together;
# see `contact` for why neither is a stone on its own.
CONTACT_SOURCES: tuple[str, ...] = (
    "rubberduck_sfx2_stones_01.wav",
    "rubberduck_sfx2_stones_02.wav",
    "rubberduck_sfx2_stones_03.wav",
)
BODY_SOURCES: tuple[str, ...] = (
    "kenney_impact_mining_000.wav",
    "kenney_impact_mining_001.wav",
    "kenney_impact_mining_002.wav",
    "kenney_impact_mining_003.wav",
    "kenney_impact_mining_004.wav",
)
FRACTURE_SOURCES: tuple[str, ...] = (
    "rubberduck_break_rock_breaking_01.wav",
    "rubberduck_break_rock_breaking_02.wav",
    "rubberduck_break_rock_breaking_03.wav",
)
DEBRIS_SOURCES: tuple[str, ...] = (
    "rubberduck_break_rock_falling_01.wav",
    "rubberduck_break_rock_falling_02.wav",
    "rubberduck_break_rock_falling_03.wav",
    "rubberduck_break_rock_falling_04.wav",
    "rubberduck_break_rock_falling_05.wav",
    "rubberduck_break_rock_falling_06.wav",
    "rubberduck_break_rock_falling_07.wav",
    "rubberduck_break_rock_falling_08.wav",
    "rubberduck_break_rock_falling_09.wav",
)


@dataclass(frozen=True)
class Stone:
    """What a stone *is*. The axis a material generator exists to own.

    These are not free parameters. Each one is a thing the rock does that the
    ear can actually hear the difference of:

    `brightness` is hardness, **in decibels** of high shelf. Granite is quartz
    and feldspar and it returns everything above a couple of kilohertz; sandstone
    is cemented sand and returns almost none of it. It is the recording's own top
    end being moved, never synthetic top end being added -- which is what keeps a
    hard stone the same event rather than that event with a hiss on it.

    The range is wide (-9 to +8 dB) and it is meant to be. This is the axis the
    generator exists for, and at the +/-3 dB the sibling generators use it
    measured as a 4 % shift in a strike's A-weighted centroid across the entire
    granite-to-sandstone range, which is nothing. See `_shelf`.

    `damping` is how fast contact dies -- what a rock does *after* it is hit, and
    the thing people mean by a stone "ringing". It multiplies every decay in the
    file, so it separates a struck slab of marble from a struck lump of chalk
    without touching the spectrum at all.

    It is also the whole of the ring, deliberately. A ring is what `modal_ring`
    was for and it is exactly what the audit killed: summing decaying sinusoids
    is how a bell is synthesised, and leading an impact with one yields a chime
    with a click in front of it. So granite rings by holding its own recorded
    decay longer, and nothing here ever adds a partial.

    `grit` is how much fine material a fracture throws. Concrete is aggregate in
    a matrix and it comes apart in dust and sand; slate cleaves clean and throws
    almost none.

    `chunkiness` is the size distribution of what breaks off: low means many
    small pieces, high means few large ones. It drives `debris` piece counts and
    their varispeed spread, not their level.
    """

    name: str
    label: str
    description: str
    brightness: float
    damping: float
    grit: float
    chunkiness: float


STONES: dict[str, Stone] = {
    "granite": Stone(
        "granite", "Granite", "hard igneous rock: bright, rings, breaks chunky",
        6.0, 1.25, 0.30, 0.85,
    ),
    "limestone": Stone(
        "limestone", "Limestone", "the default building stone: middling in every way",
        0.0, 1.00, 0.50, 0.60,
    ),
    "sandstone": Stone(
        "sandstone", "Sandstone", "soft and dead: no ring, crumbles, throws sand",
        -9.0, 0.62, 0.95, 0.25,
    ),
    "slate": Stone(
        "slate", "Slate", "cleaves in sheets: sharp, thin, almost no grit",
        8.0, 0.78, 0.12, 0.45,
    ),
    "concrete": Stone(
        "concrete", "Concrete", "aggregate in a matrix: hard but damped, dusty",
        -3.5, 0.70, 0.85, 0.70,
    ),
    "gravel": Stone(
        "gravel", "Gravel", "loose aggregate: no monolith, all contact and shift",
        3.0, 0.40, 0.70, 0.10,
    ),
}


@dataclass(frozen=True)
class Scale:
    """How much stone, and what that does to the recording.

    `speed` is varispeed and does the real work; see the module docstring on why
    mass and weight are one axis here where item 6 needed two.

    `sub_hz` and `weight` are the synthesised low end. A pebble's weight is 0.0
    on purpose: a pebble has no sub, and a generator that puts one under it is
    describing a boulder. This is the field most likely to be wrong by ear, and
    it is the field to reach for first if these come back sounding like someone
    dropping a bag of gravel -- which is the exact note item 6 got on its
    touchdown, built from these same recordings.

    `darken_hz` rolls the top off with mass. The sources are bright close-mic'd
    pebbles; played straight at boulder speed they stay pebbles that happen to be
    slow.

    It is deliberately gentler than the 1.4 kHz item 6 used on these same
    recordings, and the difference is the subject. That cue is a footpad settling
    into regolith -- dust over grit, and dull because the dust is the point. A
    struck boulder is bare rock and it still cracks up top; darkened to item 6's
    numbers the contact disappears under the synthesised weight entirely, which
    measured as 99.7 % of a `strike`'s energy below 500 Hz and a spectral centroid
    of 95 Hz. That is not a boulder, it is a kick drum.
    """

    name: str
    label: str
    description: str
    speed: float
    weight: float
    sub_hz: float
    darken_hz: float


SCALES: dict[str, Scale] = {
    "pebble": Scale("pebble", "Pebble", "a stone you could throw", 1.50, 0.00,
                    170.0, 16_000.0),
    "cobble": Scale("cobble", "Cobble", "a stone you could lift", 1.12, 0.18,
                    120.0, 11_000.0),
    "boulder": Scale("boulder", "Boulder", "a stone you could not move", 0.78,
                     0.60, 74.0, 7_000.0),
    "slab": Scale("slab", "Slab", "masonry: a wall, a pillar, a lintel", 0.60,
                  0.95, 52.0, 5_200.0),
    "cliff": Scale("cliff", "Cliff", "a rock face coming down", 0.44, 1.35,
                   38.0, 3_400.0),
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------



def _trim_to_onset(signal: np.ndarray) -> np.ndarray:
    return trim_recording_activity(
            signal, sample_rate=SR, window_seconds=0.004,
            threshold_ratio=0.05, pre_seconds=0.004, post_seconds=0.010,
        )


@lru_cache(maxsize=32)
def _load_source(filename: str) -> np.ndarray:
    path = resolve_asset(SOURCE_DIRECTORY / filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing bundled stone source {path}; see sources/provenance.json"
        )
    _, signal = read_recording(
        path, expected_rate=SR,
        trim=lambda data, rate: _trim_to_onset(data),
        label="stone source",
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
    """One RBJ shelving biquad. `high` picks which end of the spectrum moves."""

    if abs(gain_db) < 0.05 or len(signal) == 0:
        return signal
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * max(20.0, min(corner, SR * 0.45)) / SR
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    # S = 1: the steepest shelf that does not overshoot into a peak.
    alpha = sin_w0 / 2.0 * math.sqrt((amp + 1.0 / amp) * (1.0 / 1.0 - 1.0) + 2.0)
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


def _hardness(signal: np.ndarray, gain_db: float, pivot: float) -> np.ndarray:
    """Hardness: rotate the recording's spectrum about `pivot`, in decibels.

    A *tilt* -- high shelf up by half, low shelf down by half -- rather than one
    shelf, and rather than the band-mixing `_tilt` that
    ``unarmed_foley_generator`` and ``fire_sound_generator`` use. Both of those
    alternatives were measured on this material and both failed, in instructive
    ways:

    - Band-mixing (`signal + brightness * highpassed(signal)`) tops out near
      +3 dB before it reads as a copy of the hit rather than as a harder rock. It
      moved a `strike`'s A-weighted centroid by 4 % across the whole
      granite-to-sandstone range. That is not an axis, it is a rounding error.
    - A single high shelf at 2 kHz got to 17 %, and stalled there for a reason
      worth writing down: these recordings live at 4-7 kHz, so a shelf *below*
      their energy is very nearly a flat gain, and `_prepare` normalises the gain
      straight back out. Boosting a band that is already all of the signal
      changes nothing.

    A tilt has no such problem, because it always cuts one side while it lifts
    the other -- there is no net gain for normalisation to eat, and the pivot
    sits inside the band that carries the sound rather than under it.

    Nothing synthetic is added either way: this is only ever the recording's own
    energy, redistributed. Hardness is what a rock returns, so what a hard rock
    needs is more of what is already there, higher up.
    """
    if abs(gain_db) < 0.05 or len(signal) == 0:
        return signal
    lifted = _shelf(signal, gain_db * 0.5, pivot, high=True)
    return _shelf(lifted, -gain_db * 0.5, pivot, high=False)


def _pivot_for(scale: Scale) -> float:
    """Where the hardness tilt turns, for a given mass.

    It has to follow `darken_hz`. Pinned at 4 kHz, a granite cliff's tilt would
    be operating an octave above anything left in the signal after `_darken` had
    run, and the material axis would quietly vanish at exactly the scales that
    most need to sound like something.
    """
    return max(700.0, min(4_200.0, scale.darken_hz * 0.62))


def _darken(signal: np.ndarray, cutoff: float) -> np.ndarray:
    """Roll the top off. Big stone is dull stone.

    Item 6 found this building its lander touchdown out of these same files:
    "The CC0 stone recordings are bright, close-mic'd pebbles, and played
    straight they read as a handful of gravel being dropped rather than as a
    surface being displaced by something heavy landing on it."
    """
    if len(signal) == 0 or cutoff >= SR * 0.45:
        return signal
    sos = butter(4, max(80.0, min(cutoff, SR * 0.45)), btype="low", fs=SR,
                 output="sos")
    return sosfilt(sos, signal)


def _decay(signal: np.ndarray, seconds: float, hold: float = 0.0) -> np.ndarray:
    """Impose an exponential decay, so a library tail cannot outstay its cue."""

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


# --------------------------------------------------------------------------
# layers -- the public core; see the module docstring
# --------------------------------------------------------------------------

def _prepare(filename: str, stone: Stone, scale: Scale,
             rng: np.random.Generator,
             speed_multiplier: float = 1.0) -> np.ndarray:
    """One recording, retuned, tilted and darkened for this stone and scale."""

    cached = _load_source(filename)
    speed = scale.speed * speed_multiplier * float(rng.uniform(0.94, 1.06))
    signal = _varispeed(cached, speed)
    # Hardness first, then mass. The order is not arbitrary: `_darken` is a
    # property of how much stone there is and it has to have the last word, or a
    # granite cliff comes out brighter than a granite pebble.
    signal = _hardness(signal, stone.brightness, _pivot_for(scale))
    signal = _darken(signal, scale.darken_hz)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal = signal / peak
    return signal


# How loud the synthesised weight sits under a recorded contact, before the cue
# applies its own gain on top.
#
# This number is the single most likely thing in the file to be wrong by ear, and
# it is worth knowing which way it has already been wrong -- twice, in the same
# direction.
#
# `sub` returns a signal peaking at `scale.weight`, and the contact it sits under
# is peak-normalised to 1.0, so mixing it at unity gain -- which is what this
# generator did first -- puts a 74 Hz resonator within 2 dB of the rock.
# Measured, that gave a `strike --scale boulder` a spectral centroid of 95 Hz and
# 88 % of its energy under 120 Hz: the recording was still in there and inaudible
# under it. A narrowband resonator carries far more energy per unit amplitude
# than a broadband transient does, so "the same peak" is nowhere near "the same
# loudness".
#
# It came down again when `contact` grew its recorded body layer. The mining
# takes peak at 59-98 Hz, so most of the weight that this constant used to invent
# now arrives as an actual recording of a rock being hit, and stacking a
# full-strength resonator under a real low end just doubles it. Less synthesis is
# the right direction for this generator on principle: `sub` exists to fill what
# the recordings lack, and it turned out they lack less than item 10's notes
# promised.
_SUB_LEVEL = 0.28

# Peak of `grind`'s hum, against grains that peak at ~1.0, before the weight and
# intensity terms in `_hum` scale it. Set by ear on 2026-07-15: the first
# audition passed `grind` conditionally -- "fine if it had a deep stone hum
# underneath" -- against a version whose hum peaked around 0.07 at boulder scale.
# It is a *floor* the cue stands on, so it wants to be unmistakably present and
# still clearly under the grind; the failure on either side is audible within a
# second, which is more than can be said for most of the numbers in this file.
_HUM_LEVEL = 0.72


def sub(scale: Scale, rng: np.random.Generator, seconds: float,
        decay: float = 0.09) -> np.ndarray:
    """The weight, synthesised: noise through a resonance, never a sine.

    The one thing the recordings genuinely lack, and the only synthetic layer
    here. A sine under an exponential envelope is the textbook way to draw a
    boom and also the textbook way to draw a computer beep -- they are the same
    signal, which is what "bells and computer beeps" meant when item 6's first
    listener said it. What separates a thud from a beep is that a real one is
    noise being shaped by a resonance, so there is no pure tone in it at any
    point. Lifted from `aircraft_spacecraft_generator.engine._thud`, in NumPy
    rather than in that generator's stdlib DSP.

    Returns silence at `pebble`, whose weight is 0.0: a pebble has no low end
    and inventing one is how these recordings stop being what they are.
    """
    frames = _frames(seconds)
    if scale.weight <= 1e-6:
        return np.zeros(frames)
    noise = rng.standard_normal(frames)
    noise = sosfilt(butter(4, 320.0, btype="lowpass", fs=SR, output="sos"),
                    noise)
    t = np.arange(frames) / SR
    drive = noise * np.exp(-t / max(decay, 1e-4)) * (
        1.0 - np.exp(-t / 0.0015))
    b, a = iirpeak(max(18.0, scale.sub_hz), 4.5, fs=SR)
    resonant = lfilter(b, a, drive)
    peak = float(np.max(np.abs(resonant)))
    if peak > 1e-9:
        resonant = resonant / peak
    return resonant * scale.weight


def contact(stone: Stone, scale: Scale, rng: np.random.Generator,
            intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Stone being touched: the hit itself, before anything breaks.

    Two recordings from *different families*, and that pairing is the whole cue.
    Neither is a struck stone on its own, and their measured spectra say so
    plainly:

    - `rubberduck_sfx2_stones_*` is loose stones knocking. Spectral centroids
      4.1-4.9 kHz, and **0-3 % of their energy below 500 Hz**. All crack, no
      body: played alone at boulder speed it is a pebble that happens to be slow.
    - `kenney_impact_mining_*` is a tool going into rock. It peaks at **59-98 Hz**
      with 97-98 % of its energy below 500 Hz. All body, no crack.

    So the bright one leads and the low one arrives underneath it a few
    milliseconds later, which is the order the event actually happens in: the
    surface fails first and the mass answers. Item 6's third lesson, which it
    learned by putting the regolith in front of the plate -- lead with the
    contact, not the clatter.

    **The mining family is also a standing exception to the earlier working rule
    that recordings have nothing below roughly 80 Hz because they are library
    impacts of small objects.** That is true of
    every other recording in this repository, including the other fifteen here.
    It is not true of these five: 22-85 % of their energy is under 80 Hz and they
    peak below 100. That is a *recorded* stone low end, which is precisely the
    thing `sub` exists to fake, so wherever these reach `sub` has less to do.

    The stagger stays under ~12 ms for the reason item 8 worked out: past that
    the ear stops fusing two layers and hears a double tap.
    """
    primary = _choose(rng, CONTACT_SOURCES)
    secondary = _choose(rng, tuple(s for s in CONTACT_SOURCES if s != primary))
    body_name = _choose(rng, BODY_SOURCES)

    lead = _prepare(primary, stone, scale, rng)
    lead = _decay(lead, (0.09 + 0.05 / max(scale.speed, 0.3)) * stone.damping)
    back = _prepare(secondary, stone, scale, rng, speed_multiplier=0.88)
    back = _decay(back, (0.07 + 0.04 / max(scale.speed, 0.3)) * stone.damping)

    # The mass answering. Its own recorded weight is why this is not just `sub`
    # with extra steps: a resonator rings one frequency, and this is a real rock
    # absorbing a blow. It varispeeds with scale like everything else, so a cliff
    # gets a deeper recorded body and not only a deeper synthetic one.
    body = _prepare(body_name, stone, scale, rng, speed_multiplier=0.92)
    body = _decay(body, (0.11 + 0.09 * scale.weight) * stone.damping)

    total = max(len(lead), len(back) + _frames(0.012),
                len(body) + _frames(0.014)) + _frames(0.02)
    out = np.zeros(total)
    _mix_at(out, lead, 0, 1.0)
    _mix_at(out, back, _frames(float(rng.uniform(0.003, 0.012))),
            0.40 * intensity)
    # Tracks weight almost all the way down, and the floor is 0.10 rather than
    # something comfortable like 0.4: a struck pebble has a body, but it is a
    # pebble's body. At a flat 0.42 the mining take put 37 % of a *pebble*
    # strike's energy under 120 Hz, which is a fist-sized stone with a kick drum
    # inside it -- the same mistake as an over-loud `sub`, made with a recording
    # instead of a resonator.
    _mix_at(out, body, _frames(float(rng.uniform(0.002, 0.008))),
            (0.10 + 0.75 * scale.weight) * intensity)
    return out, [primary, secondary, body_name]


def fracture(stone: Stone, scale: Scale, rng: np.random.Generator,
             intensity: float = 0.9,
             completeness: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """Stone failing: the crack running through it.

    `completeness` is how far the fracture gets. At 1.0 the recording runs out
    and the thing is in pieces; below it the tail is cut and damped, which is a
    crack that propagates and arrests -- a stone that is now cracked rather than
    a stone that is now gravel. That is the whole difference between `crack` and
    `shatter`, and it is one number rather than two renderers because it is one
    event heard at two different lengths.
    """
    name = _choose(rng, FRACTURE_SOURCES)
    signal = _prepare(name, stone, scale, rng)
    completeness = float(min(max(completeness, 0.05), 1.0))
    if completeness < 1.0:
        keep = max(_frames(0.03), int(len(signal) * completeness))
        signal = signal[:keep]
        signal = _decay(signal, 0.05 * stone.damping, hold=0.008)
    else:
        signal = _decay(signal, (0.30 + 0.25 * stone.chunkiness) * stone.damping,
                        hold=0.02)

    # Sandstone crumbles where granite cracks: the grit is the fracture's own
    # top end, sped up and laid back under it. Lifted from the recording rather
    # than synthesized, for the reason `_tilt` gives.
    sources = [name]
    if stone.grit > 0.05:
        grit_name = _choose(rng, DEBRIS_SOURCES)
        grit = _prepare(grit_name, stone, scale, rng, speed_multiplier=1.7)
        grit = sosfilt(butter(2, 1_800.0, btype="highpass", fs=SR, output="sos"),
                       grit)
        grit = _decay(grit, 0.18 * completeness + 0.04)
        combined = np.zeros(max(len(signal), len(grit) + _frames(0.01)))
        _mix_at(combined, signal, 0, 1.0)
        _mix_at(combined, grit, _frames(0.008), 0.45 * stone.grit * intensity)
        signal = combined
        sources.append(grit_name)
    return signal * intensity, sources


def debris(stone: Stone, scale: Scale, rng: np.random.Generator,
           seconds: float = 1.4, count: int | None = None,
           intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """The pieces afterwards: scatter, tumble, settle.

    `chunkiness` sets both how many pieces there are and how far apart they are
    in size. Slate throws a few flat plates; sandstone throws a hundred grains.
    The onsets are skewed early (`** 0.7`) because debris does not arrive
    uniformly -- most of it lands at once and the stragglers are what you hear
    last.
    """
    frames = _frames(seconds)
    out = np.zeros(frames)
    if count is None:
        count = int(round((28 - 18 * stone.chunkiness) * (0.4 + 0.9 * intensity)))
    count = max(1, count)
    sources: list[str] = []
    for index in range(count):
        name = _choose(rng, DEBRIS_SOURCES)
        # Small pieces are fast and bright; big ones are slow. chunkiness widens
        # the spread rather than shifting it.
        spread = 0.5 + 1.4 * stone.chunkiness
        piece = _prepare(name, stone, scale, rng,
                         speed_multiplier=float(
                             rng.uniform(1.0, 1.0 + spread)))
        piece = _decay(piece, float(rng.uniform(0.05, 0.20)) * stone.damping)
        start = float(rng.random()) ** 0.7 * seconds * 0.82
        gain = float(rng.uniform(0.10, 0.38)) * (1.0 - 0.5 * start / seconds)
        _mix_at(out, piece, _frames(start), gain * intensity)
        if name not in sources:
            sources.append(name)
    return out, sources


# --------------------------------------------------------------------------
# cues -- events
# --------------------------------------------------------------------------

def strike(stone: Stone, scale: Scale, rng: np.random.Generator,
           intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Stone struck, and it holds. A pick, a hammer, a round off a wall.

    Contact and weight, and nothing else: nothing broke, so nothing follows. The
    contact leads and the sub sits under it rather than in front -- item 6's
    third lesson, learned when its touchdown put the regolith first and read as
    a dropped bag of gravel.
    """
    body, sources = contact(stone, scale, rng, intensity)
    weight = sub(scale, rng, min(len(body) / SR, 0.5),
                 decay=(0.045 + 0.05 * scale.weight) * stone.damping)
    out = np.zeros(max(len(body), len(weight)))
    _mix_at(out, body, 0, 1.0)
    _mix_at(out, weight, _frames(0.003), _SUB_LEVEL * (0.55 + 0.25 * intensity))
    return _fades(out), sources


def crack(stone: Stone, scale: Scale, rng: np.random.Generator,
          intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Stone failing, but holding together. A fissure, not a collapse.

    The cue is the *absence* of debris. A crack that throws pieces is a shatter;
    what makes this one read is that the fracture arrests and the silence after
    it is where the rubble would have been.
    """
    body, sources = contact(stone, scale, rng, intensity * 0.8)
    split, split_sources = fracture(stone, scale, rng, intensity,
                                    completeness=0.35)
    sources = sources + [s for s in split_sources if s not in sources]

    total = max(len(body), len(split) + _frames(0.02)) + _frames(0.05)
    out = np.zeros(total)
    _mix_at(out, body, 0, 1.0)
    _mix_at(out, split, _frames(float(rng.uniform(0.010, 0.030))),
            0.85 * intensity)
    weight = sub(scale, rng, min(total / SR, 0.6),
                 decay=(0.05 + 0.05 * scale.weight) * stone.damping)
    _mix_at(out, weight, _frames(0.004), _SUB_LEVEL * (0.5 + 0.25 * intensity))
    return _fades(out), sources


def shatter(stone: Stone, scale: Scale, rng: np.random.Generator,
            intensity: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """Stone coming apart: contact, fracture, and everything afterwards.

    The order is the cue. Contact, then the fracture a few milliseconds behind
    it, then debris that outlives both -- and the debris starts while the
    fracture is still going, because a rock does not finish breaking before the
    first piece lands.
    """
    body, sources = contact(stone, scale, rng, intensity)
    split, split_sources = fracture(stone, scale, rng, intensity,
                                    completeness=1.0)
    seconds = 1.1 + 1.3 * intensity + 0.5 * scale.weight
    rubble, rubble_sources = debris(stone, scale, rng, seconds=seconds * 0.85,
                                    intensity=intensity)
    for name in split_sources + rubble_sources:
        if name not in sources:
            sources.append(name)

    out = np.zeros(_frames(seconds))
    _mix_at(out, body, 0, 1.0)
    _mix_at(out, split, _frames(float(rng.uniform(0.006, 0.020))),
            0.95 * intensity)
    _mix_at(out, rubble, _frames(float(rng.uniform(0.05, 0.11))), 1.0)
    weight = sub(scale, rng, min(seconds, 0.75),
                 decay=(0.07 + 0.07 * scale.weight) * stone.damping)
    _mix_at(out, weight, _frames(0.004), _SUB_LEVEL * (0.7 + 0.3 * intensity))
    return _fades(out), sources


def fall(stone: Stone, scale: Scale, rng: np.random.Generator,
         intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """One stone landing: the arrival, the bounces, the settle.

    Not `strike` with a different name. A strike is something arriving *at*
    stone; this is stone arriving at something, and what separates them is that a
    falling rock does not stop where it lands. The bounces decay in level and
    stretch in spacing, which is the pattern the ear knows from a lifetime of
    dropping things.
    """
    seconds = 0.7 + 0.9 * scale.weight + 0.4 * intensity
    out = np.zeros(_frames(seconds))
    landing, sources = contact(stone, scale, rng, intensity)
    _mix_at(out, landing, 0, 1.0)
    weight = sub(scale, rng, min(seconds, 0.5),
                 decay=(0.04 + 0.05 * scale.weight) * stone.damping)
    _mix_at(out, weight, _frames(0.003), _SUB_LEVEL * (0.6 + 0.25 * intensity))

    # Each bounce later and quieter than the last, and faster, because what is
    # left bouncing has less energy and reads smaller.
    onset = 0.0
    bounces = 1 + int(round(2 * intensity * (1.0 - 0.4 * scale.weight)))
    for index in range(bounces):
        onset += float(rng.uniform(0.09, 0.20)) * (1.0 + 0.5 * index)
        if onset >= seconds - 0.05:
            break
        name = _choose(rng, DEBRIS_SOURCES)
        piece = _prepare(name, stone, scale, rng,
                         speed_multiplier=1.0 + 0.22 * (index + 1))
        piece = _decay(piece, 0.09 * stone.damping)
        _mix_at(out, piece, _frames(onset),
                0.45 * intensity * (0.6 ** index))
        if name not in sources:
            sources.append(name)
    return _fades(out), sources


def scatter(stone: Stone, scale: Scale, rng: np.random.Generator,
            intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Rubble: pieces arriving, with nothing in front of them.

    `debris` on its own, which is what an event generator wants when the thing
    that broke was somewhere else -- a wall coming down over there, or the tail
    of an explosion that this generator had no part in. It is deliberately
    headless: adding a contact would make it a `shatter` heard from the wrong
    end.
    """
    seconds = 0.9 + 1.1 * intensity + 0.4 * scale.weight
    out, sources = debris(stone, scale, rng, seconds=seconds,
                          intensity=intensity)
    if scale.weight > 0.4:
        # Enough stone moving to be felt rather than only heard.
        weight = sub(scale, rng, min(seconds, 0.9), decay=0.22)
        _mix_at(out, weight, _frames(0.02), 0.5 * scale.weight * intensity)
    return _fades(out), sources


# --------------------------------------------------------------------------
# cues -- loops
# --------------------------------------------------------------------------

def _grain(material: np.ndarray, rng: np.random.Generator,
           seconds: float) -> np.ndarray:
    """A windowed slice from somewhere inside a recording.

    Hann-windowed rather than fade-in/fade-out: a grain this short is mostly
    envelope, and a linear fade leaves a corner at each end that reads as a tick
    at the grain rate -- which is a buzz, and a buzz is the thing that makes
    granular material sound synthetic.
    """
    need = _frames(seconds)
    if len(material) <= need:
        piece = np.resize(material, need).astype(float)
    else:
        offset = int(rng.integers(0, len(material) - need))
        piece = material[offset:offset + need].astype(float)
    return piece * np.hanning(len(piece))


def grind(stone: Stone, scale: Scale, seconds: float,
          rng: np.random.Generator,
          intensity: float = 0.7) -> tuple[np.ndarray, list[str]]:
    """Stone dragged over stone: a millstone, a lid, a boulder shifting.

    **This is the cue most likely to come back rejected, and it should be
    listened to first.** It is the only one here with no recording behind it:
    the vetted CC0 supply has no sustained stone-on-stone grinding in it at all,
    so this is built by granulating the loose-stones recordings into a dense
    train of tiny contacts.

    The argument for it is that a grind physically *is* that -- thousands of
    small asperities catching and releasing -- and the grains are real contacts
    rather than filtered noise. The argument against is item 3, whose `skid` is
    stick-slip at a definite rate, was modelled explicitly, and was rejected
    twice, as "a whoosh" and then "a grinder". A grind being called "a grinder"
    would be a pass; the same cue being called "a whoosh" is the failure mode to
    listen for, and it means the grain density has flattened into noise.

    Seamless by construction: grains are placed modularly, so the wrap is one
    more grain boundary among thousands.
    """
    frames = _frames(seconds)
    out = np.zeros(frames)
    material = _prepare(_choose(rng, CONTACT_SOURCES), stone, scale, rng)
    second = _prepare(_choose(rng, CONTACT_SOURCES), stone, scale, rng,
                      speed_multiplier=0.77)
    sources = list(CONTACT_SOURCES)

    # Density is the whole cue. Too sparse and it is pebbles rattling; too dense
    # and the grains overlap into the whoosh item 3 was told off for. Harder
    # stone catches more often and lets go more cleanly. Bigger stone drags
    # slower, so its asperities catch less often per second.
    density = (170.0 + 260.0 * intensity) * (1.0 + 0.5 * stone.brightness)
    density = density / max(0.5, scale.speed) ** 0.5
    total = max(8, int(density * seconds))
    for index in range(total):
        source = material if index % 3 else second
        grain = _grain(source, rng, float(rng.uniform(0.012, 0.045)))
        _mix_wrapped(out, grain, int(rng.integers(0, frames)),
                     float(rng.uniform(0.25, 1.0)) ** 2 * 0.9)

    # Sandstone drags where granite chatters: a bed of fine material under the
    # contacts, taken from the same recordings sped up past recognition.
    if stone.grit > 0.05:
        fine = _prepare(_choose(rng, DEBRIS_SOURCES), stone, scale, rng,
                        speed_multiplier=2.4)
        for _ in range(int(120 * seconds)):
            grain = _grain(fine, rng, float(rng.uniform(0.006, 0.018)))
            _mix_wrapped(out, grain, int(rng.integers(0, frames)),
                         float(rng.uniform(0.05, 0.30)) * stone.grit)

    out += _hum(stone, scale, frames, rng, intensity)
    return out, sources


def _hum(stone: Stone, scale: Scale, frames: int, rng: np.random.Generator,
         intensity: float) -> np.ndarray:
    """The deep stone hum under a drag: the mass, ringing continuously.

    This layer has now been wrong in both directions, which is worth recording
    because they are opposite mistakes with the same cause -- treating a
    continuous event as a series of discrete ones.

    First it was **sparse and loud**: ~3 subs a second, which is a boulder being
    dropped repeatedly rather than one being pushed. It measured as a broken loop
    too (`grind granite cliff`, level ratio 0.233 against a 0.25 floor), because
    a handful of loud sparse events cannot be stationary across a wrap.

    The fix overcorrected into **dense and inaudible** -- 28 hits a second at
    0.16 gain, gated off below `weight > 0.3` entirely. That passed the seam
    metric and lost the sound: on the first by-ear audition the verdict on
    `grind` was "fine if it had a deep stone hum underneath", which is this layer
    saying it was not there.

    So: dense *and* present. Seventy overlapping rings a second at a 0.14 s decay
    is a ~10x overlap factor, which sums to something genuinely continuous rather
    than to a flutter, and it is loud enough to be the floor of the cue instead of
    a rumour of one. It is still `sub` underneath -- noise through a resonance,
    never a sine -- so the hum has no pure tone in it at any point, which is what
    keeps a dragged boulder from turning into an organ pedal.

    No `weight` gate. A pebble ground against stone has no hum because
    `scale.weight` is 0.0 and `sub` returns silence for it; every scale above it
    has some, in proportion. A gate at 0.3 only ever meant "cobble is silent",
    which is not true of any real stone being dragged.
    """
    out = np.zeros(frames)
    if scale.weight <= 1e-6:
        return out
    for _ in range(max(8, int(70 * frames / SR))):
        ring = sub(scale, rng, float(rng.uniform(0.16, 0.30)), decay=0.14)
        _mix_wrapped(out, ring, int(rng.integers(0, frames)),
                     float(rng.uniform(0.55, 1.0)))

    # Normalise *after* summing, because the overlap that makes this a hum also
    # multiplies it. Ten overlapping rings of incoherent noise sum to roughly
    # sqrt(10) times one of them, so a level set per-ring lands wherever the
    # density happens to leave it: measured, a `cliff` hum built that way peaked
    # at 4.2 against grains at 1.0 and buried the cue it was supposed to sit
    # under. Setting the peak here instead makes the density a property of the
    # *texture* and the level a separate decision, which is the only way to tune
    # either one without breaking the other.
    peak = float(np.max(np.abs(out)))
    if peak <= 1e-9:
        return out

    # Compressed rather than linear in weight. A cliff has far more mass than a
    # boulder but not proportionally more hum, because what is dragging is still
    # one contact patch -- and left linear the hum overtakes the grind at slab
    # and never gives it back. A softer stone drags where a hard one chatters, so
    # its hum runs a shade louder relative to the contacts above it.
    softness = 1.0 - min(stone.damping, 1.25) / 1.25
    target = (_HUM_LEVEL + 0.22 * softness) * (scale.weight ** 0.45) * (
        0.7 + 0.45 * intensity)
    return out * (target / peak)


def rockfall(stone: Stone, scale: Scale, seconds: float,
             rng: np.random.Generator,
             intensity: float = 0.7) -> tuple[np.ndarray, list[str]]:
    """A slope coming down and not stopping: a sustained cascade.

    Distinct from `scatter` by having no end. Scatter is an event that decays;
    this is a bed that plays under a scene while a cliff is failing, which is why
    it is a loop and why its arrivals are stationary in time rather than skewed
    early.

    This is the `cliff` case the module docstring flags: a rock face does not
    arrive on one contact patch, so the weight comes from many wrapped subs
    spread through the loop rather than from driving one harder. That is what
    keeps it a rumble rather than a heartbeat.

    ``environment_ambience_generator`` also ships a procedural one-shot named
    `rockfall`. The two cues have different ownership and loop semantics, so no
    automatic replacement occurs here.
    """
    frames = _frames(seconds)
    out = np.zeros(frames)
    sources: list[str] = []

    rate = (7.0 + 26.0 * intensity) / max(0.6, scale.speed)
    for _ in range(max(4, int(rate * seconds))):
        name = _choose(rng, DEBRIS_SOURCES)
        piece = _prepare(name, stone, scale, rng,
                         speed_multiplier=float(
                             rng.uniform(0.8, 1.6 + 1.2 * stone.chunkiness)))
        piece = _decay(piece, float(rng.uniform(0.06, 0.26)) * stone.damping)
        _mix_wrapped(out, piece, int(rng.integers(0, frames)),
                     float(rng.uniform(0.12, 0.55)))
        if name not in sources:
            sources.append(name)

    # The fine material between the rocks, which is most of what a real slope
    # sounds like and none of what a volley of impacts sounds like.
    fine = _prepare(_choose(rng, CONTACT_SOURCES), stone, scale, rng,
                    speed_multiplier=1.9)
    for _ in range(int((90 + 150 * intensity) * seconds)):
        grain = _grain(fine, rng, float(rng.uniform(0.008, 0.030)))
        _mix_wrapped(out, grain, int(rng.integers(0, frames)),
                     float(rng.uniform(0.05, 0.35)) * (0.4 + stone.grit))
    for name in CONTACT_SOURCES:
        if name not in sources:
            sources.append(name)

    if scale.weight > 1e-6:
        for _ in range(max(3, int(4 * seconds * intensity))):
            weight = sub(scale, rng, float(rng.uniform(0.4, 0.9)), decay=0.30)
            _mix_wrapped(out, weight, int(rng.integers(0, frames)),
                         float(rng.uniform(0.3, 0.8)) * scale.weight)
    return out, sources


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def master(signal: np.ndarray, peak_db: float,
           loop: bool = False) -> list[float]:
    """Final chain shared by every cue: soft limit, fade, normalize.

    A loop is not faded at its edges: fading one would put a hole at exactly the
    join the wrapped placement made seamless.
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


def stone_named(name: str) -> Stone:
    if name not in STONES:
        raise ValueError(
            f"unknown stone {name!r}; expected one of {sorted(STONES)}"
        )
    return STONES[name]


def scale_named(name: str) -> Scale:
    if name not in SCALES:
        raise ValueError(
            f"unknown scale {name!r}; expected one of {sorted(SCALES)}"
        )
    return SCALES[name]


def bundled_sources() -> tuple[str, ...]:
    return asset_names(SOURCE_DIRECTORY)
