"""Human non-verbal voice, built from real CC0 recordings.

A hybrid generator in the lineage of ``door_sound_generator``,
``creature_foley_generator``, ``unarmed_foley_generator`` and
``fire_sound_generator``: NumPy and SciPy trim, retune, decay, layer, and
window bundled recordings rather than synthesizing a voice from oscillators.

Why this is recordings and not synthesis
----------------------------------------
A human vocalization is a chaotic glottal source driven through a fleshy,
moving resonator: a pain grunt is a burst of turbulent breath, a hard glottal
onset, and formant transients that shift as the throat closes on it. That is
exactly the territory procedural DSP loses in -- the same reason
``creature_foley_generator`` is recordings for growls, and the reason its
grunt/hurt/breath cues could not simply be reused here: those are a rubberduck
*creature* library, not a human. Fantasy Adventure's player/hurt prompt is explicit --
"a single very brief nonverbal HUMAN pain grunt ... No monster voice" -- and
there was no human voice source anywhere in the repo. So this generator exists,
and everything voiced in it is a real human recording, retuned and shaped.

The size / character axis is varispeed
--------------------------------------
Body size is resampling, borrowed from ``creature_foley_generator``: a big
chest is not a small one turned down, it is slower and lower, because the vocal
folds are longer and the vocal tract is longer. Resampling a grunt moves its
pitch *and* its formants together, which is what actually reads as a bigger
body -- pitch-shifting alone leaves the formants behind and just sounds like a
slowed-down small person. That is also why size is applied to the recording,
not to a filter afterwards.

The only synthesized element
----------------------------
``_chest`` -- and only for large bodies. It is filtered noise through a low
resonance, enveloped to follow the vocal, never a sine (a sine under a voice
reads as a synth bass, not a chest). It adds the low body weight a giant's
grunt has and a recorded human throat does not. Small and adult voices never
touch it: they are pure recording.

What this generator does NOT own
--------------------------------
Fantasy Adventure's player/death is a human gasp *plus* a soft armored body fall *plus* a
magical life-energy dissolve. This generator owns only the human gasp layer.
The body fall is ``combat_generators``' body_fall; the dissolve is
``magic_sound_generator``'s. A PackEntry is one cue per file, so the ``death``
cue here is the gasp, cleanly -- the layering into the finished player/death is
the game's job (or a future composite), and it is documented as such rather
than approximated with a synthesized thud.

All renderers return unmastered mono floating-point PCM plus the source files
they consumed. The cue registry owns mastering levels, so every public render
follows the same contract, and the consumed filenames reach the manifest.
"""

from __future__ import annotations

from ..._audio.assets import asset_names

from ..._audio.assets import resolve_asset

from ..._audio.foley_numpy import read_recording

from ..._audio.foley_numpy import write_wav_scipy as write_wav  # noqa: F401

from ..._audio.foley_numpy import trim_activity as trim_recording_activity

from ..._audio.foley_numpy import fade_edges, mix_at_frames, varispeed_configured


from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt

SR = 44_100
SOURCE_DIRECTORY = Path(__file__).resolve().parent / "sources"


# --------------------------------------------------------------------------
# bundled banks
#
# Each name is a real CC0 recording; see sources/provenance.json for the
# licence read off each source's own page and the download SHA-256.
# --------------------------------------------------------------------------

# thebardofblasphemy, CC0: 50 s of one man's voice "attempting to sound injured
# or dying". One file, ~19 grunts separated by silence, cut into individual
# vocalizations at render time by _segments -- the hurt and death bank.
PAIN_BANK = "thebardofblasphemy_death_pain_grunts.wav"

# HaelDB, offered dual OGA-BY 3.0 / CC0 and used here under CC0: four male
# vocalists recorded on a Neumann. The short grunts and the longer yells were
# split out of the archive as individual clips.
MALE_GRUNTS: tuple[str, ...] = (
    "haeldb_grunt_01.wav",
    "haeldb_grunt_02.wav",
    "haeldb_grunt_03.wav",
    "haeldb_grunt_04.wav",
    "haeldb_grunt_05.wav",
    "haeldb_grunt_06.wav",
)
MALE_YELLS: tuple[str, ...] = (
    "haeldb_yell_01.wav",
    "haeldb_yell_02.wav",
    "haeldb_yell_03.wav",
)

# cicifyre, CC0: the female "damaged" and "jump" reactions from the RPG Voice
# Starter Pack. Only the wordless reactions were taken -- the spell names
# ("fire", "heal", "aqua") are speech and have no place in a non-verbal
# generator. At ~0.3 s these are far too short to be the spoken word "damaged";
# they are grunts.
FEMALE_HURT: tuple[str, ...] = (
    "cicifyre_damaged_01.wav",
    "cicifyre_damaged_02.wav",
    "cicifyre_damaged_03.wav",
)
FEMALE_EFFORT: tuple[str, ...] = (
    "cicifyre_jump_01.wav",
    "cicifyre_jump_02.wav",
    "cicifyre_jump_03.wav",
)


# --------------------------------------------------------------------------
# axes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Body:
    """How big the body is, and what that does to the recording.

    `speed` is varispeed and does the real work: a smaller body is faster and
    higher because the folds and the tract are shorter. `weight` is the only
    thing that reaches the synthesizer -- how much of the ``_chest`` sub a big
    body gets. Small bodies get none and stay pure recording.
    """

    name: str
    label: str
    description: str
    speed: float
    weight: float


BODIES: dict[str, Body] = {
    "child": Body("child", "Child", "a small child or halfling", 1.34, 0.0),
    "youth": Body("youth", "Youth", "a boy, girl, or slight adult", 1.14, 0.0),
    "adult": Body("adult", "Adult", "a grown human, the neutral body", 1.00, 0.10),
    "large": Body("large", "Large", "a big warrior or an ogre-kin", 0.84, 0.55),
    "giant": Body("giant", "Giant", "a giant, a troll, a huge body", 0.70, 1.00),
}


# Which recordings each cue may draw from, per voice. A cue with no pool for a
# requested voice is an honest gap, not a fallback to the wrong sound: render
# raises rather than handing back a male grunt for a female death.
@dataclass(frozen=True)
class Voice:
    name: str
    label: str


VOICES: dict[str, Voice] = {
    "male": Voice("male", "Male"),
    "female": Voice("female", "Female"),
}




def _frames(seconds: float) -> int:
    return max(1, round(seconds * SR))


def _trim_activity(signal: np.ndarray, lead_only: bool=False) -> np.ndarray:
    return trim_recording_activity(
            signal, sample_rate=SR, window_seconds=0.012,
            threshold_ratio=0.02, threshold_floor_db=-60.0,
            pre_seconds=0.010,
            post_seconds=None if lead_only else 0.035,
        )


@lru_cache(maxsize=64)
def _load_source(filename: str) -> np.ndarray:
    path = resolve_asset(SOURCE_DIRECTORY / filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing bundled voice source {path}; see sources/provenance.json"
        )
    _, signal = read_recording(
        path,
        expected_rate=SR,
        trim=lambda data, rate: _trim_activity(
            data, lead_only=(filename == PAIN_BANK)
        ),
        label="voice source",
    )
    return signal


@lru_cache(maxsize=4)
def _segments(filename: str) -> tuple[tuple[int, int], ...]:
    """Split a multi-grunt bank into individual vocalizations.

    The bardofblasphemy recording is one long take of many grunts and groans
    separated by silence; a single grunt has to be cut out of it before it can
    be retuned. The gaps between grunts are wide (>0.2 s) and the grunts vary
    enormously in level, so the threshold is deliberately low and merging is
    generous: better one clean grunt with a little room around it than two
    halves of one.
    """
    signal = _load_source(filename)
    window = max(3, round(0.020 * SR))
    envelope = np.sqrt(uniform_filter1d(signal * signal, window))
    threshold = max(float(np.max(envelope)) * 0.05, 10.0 ** (-46.0 / 20.0))
    active = envelope >= threshold

    runs: list[list[int]] = []
    start: int | None = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append([start, index])
            start = None
    if start is not None:
        runs.append([start, len(signal)])

    gap = round(0.180 * SR)
    merged: list[list[int]] = []
    for first, last in runs:
        if merged and first - merged[-1][1] < gap:
            merged[-1][1] = last
        else:
            merged.append([first, last])

    pad = round(0.012 * SR)
    floor = round(0.150 * SR)
    return tuple(
        (max(0, a - pad), min(len(signal), b + pad))
        for a, b in merged
        if (b - a) >= floor
    )


def _varispeed(signal: np.ndarray, speed: float) -> np.ndarray:
    return varispeed_configured(
        signal, speed, minimum=0.25, maximum=4.0, denominator=512
    )


def _fades(signal: np.ndarray, seconds: float=0.005) -> np.ndarray:
    return fade_edges(signal, seconds, sample_rate=SR)


def _decay(signal: np.ndarray, seconds: float) -> np.ndarray:
    """An exponential settle laid over a vocal's tail.

    Not a gate: it tucks the natural decay so a retuned grunt does not ring
    longer than a grunt should, and so a layered chest sub dies with the voice
    rather than humming under the silence after it.
    """
    if seconds <= 0.0 or len(signal) == 0:
        return np.asarray(signal, dtype=float)
    t = np.arange(len(signal)) / SR
    return np.asarray(signal, dtype=float) * np.exp(-t / seconds)


def _choose(rng: np.random.Generator, choices: Sequence[str]) -> str:
    return choices[int(rng.integers(0, len(choices)))]


def _mix_at(destination: np.ndarray, source: np.ndarray, start_frame: int, gain: float) -> None:
    mix_at_frames(destination, source, start_frame, gain)


def _normalize(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    return signal / peak if peak > 1e-9 else signal


# --------------------------------------------------------------------------
# the one synthesized element
# --------------------------------------------------------------------------

def _chest(vocal: np.ndarray, weight: float,
           rng: np.random.Generator) -> np.ndarray:
    """Low body weight for a big frame: noise through a resonance, not a sine.

    A giant's grunt has a chest behind it that a recorded human throat does
    not. This is the only synthesized sound in the generator and it is built
    the way the fireball rule says a body sub must be: broadband noise rung
    through a low resonant band and then made to *follow the voice* -- its
    amplitude is the vocal's own low-frequency envelope -- so it reads as the
    body under the grunt and not as a synth bass note struck alongside it. A
    sine here would sound exactly like the synth the recordings are here to
    avoid, which is why it is noise.
    """
    if weight <= 0.0 or len(vocal) == 0:
        return np.zeros(len(vocal))
    low = sosfilt(butter(2, 200.0, btype="lowpass", fs=SR, output="sos"), vocal)
    envelope = np.sqrt(uniform_filter1d(low * low, max(3, round(0.020 * SR))))
    envelope = envelope / (float(np.max(envelope)) + 1e-9)
    noise = rng.standard_normal(len(vocal))
    rung = sosfilt(
        butter(2, [70.0, 150.0], btype="bandpass", fs=SR, output="sos"), noise
    )
    sub = _decay(rung * envelope, 0.5)
    return weight * 0.6 * _normalize(sub)


# --------------------------------------------------------------------------
# picking and shaping a vocalization
# --------------------------------------------------------------------------

def _bank_clip(rng: np.random.Generator,
               shorter_than: float | None = None,
               longer_than: float | None = None) -> np.ndarray:
    """One grunt cut out of the pain bank, optionally by length.

    hurt wants the short punchy grunts; death wants the long groans and gasps.
    The filter is a soft preference: if nothing matches the window it falls
    back to the whole set rather than failing, so a body/voice combination can
    never starve the cue.
    """
    signal = _load_source(PAIN_BANK)
    segments = _segments(PAIN_BANK)
    candidates = []
    for a, b in segments:
        seconds = (b - a) / SR
        if shorter_than is not None and seconds > shorter_than:
            continue
        if longer_than is not None and seconds < longer_than:
            continue
        candidates.append((a, b))
    if not candidates:
        candidates = list(segments)
    a, b = candidates[int(rng.integers(0, len(candidates)))]
    return signal[a:b].copy()


def _shape(clip: np.ndarray, body: Body, rng: np.random.Generator,
           intensity: float) -> np.ndarray:
    """Retune a raw clip to a body and an intensity, and settle its tail.

    A harder vocalization is a touch faster and higher -- the throat tightens
    under effort -- so intensity nudges the varispeed up. The jitter on top is
    what keeps two variants off the same seed from being the identical file at
    the identical pitch.
    """
    tension = 0.96 + 0.08 * float(np.clip(intensity, 0.0, 1.0))
    speed = body.speed * tension * float(rng.uniform(0.975, 1.028))
    vocal = _varispeed(clip, speed)
    vocal = _trim_activity(vocal, lead_only=False)
    # settle anything longer than a short bark so a retuned groan does not ring
    if len(vocal) > _frames(0.6):
        vocal = _decay(vocal, 0.9)
    return _fades(_normalize(vocal), 0.006)


def _vocal(pool: Sequence[str], body: Body, rng: np.random.Generator,
           intensity: float, *, bank_shorter_than: float | None = None,
           bank_longer_than: float | None = None) -> tuple[np.ndarray, str]:
    """Choose a source from a pool, load and shape it. Returns (signal, name).

    ``PAIN_BANK`` in a pool means "a segment of the bank"; any other name means
    the whole clip. Layers the chest sub under it for big bodies -- the only
    place the recording is joined by something synthesized.
    """
    name = _choose(rng, pool)
    if name == PAIN_BANK:
        clip = _bank_clip(rng, shorter_than=bank_shorter_than,
                          longer_than=bank_longer_than)
    else:
        clip = _load_source(name)
    vocal = _shape(clip, body, rng, intensity)
    chest_weight = body.weight * (0.6 + 0.4 * float(np.clip(intensity, 0.0, 1.0)))
    if chest_weight > 0.0:
        sub = _chest(vocal, chest_weight, rng)
        out = np.zeros(max(len(vocal), len(sub)))
        _mix_at(out, vocal, 0, 1.0)
        _mix_at(out, sub, 0, 1.0)
        vocal = out
    return _fades(vocal, 0.006), name


# --------------------------------------------------------------------------
# cues
#
# Each returns (unmastered mono float, [source filenames]).
# --------------------------------------------------------------------------

def hurt(body: Body, voice: Voice, rng: np.random.Generator,
         intensity: float = 0.85) -> tuple[np.ndarray, list[str]]:
    """A brief nonverbal pain grunt -- Fantasy Adventure's player/hurt.

    The priority cue and the reason the generator exists. Male draws from both
    banks -- the bardofblasphemy pain grunts and HaelDB's short grunts -- so
    five variants are five different throats, not one file five ways. Female is
    cicifyre's wordless "damaged" reactions.
    """
    if voice.name == "female":
        vocal, name = _vocal(FEMALE_HURT, body, rng, intensity)
        return vocal, [name]
    pool = (PAIN_BANK,) + MALE_GRUNTS
    vocal, name = _vocal(pool, body, rng, intensity, bank_shorter_than=1.2)
    return vocal, [name]


def death(body: Body, voice: Voice, rng: np.random.Generator,
          intensity: float = 0.7) -> tuple[np.ndarray, list[str]]:
    """A death gasp or groan -- the human layer of Fantasy Adventure's player/death.

    Only the gasp: player/death is this over a soft armored body fall
    (combat_generators) and a magical dissolve (magic_sound_generator), and
    those layers are the game's to add. Draws the longer segments of the pain
    bank, which are the groans and the breathy gasps rather than the barks.
    """
    if voice.name != "male":
        raise ValueError(
            "death is bundled male-only: no CC0 female death gasp was found "
            "(cicifyre's pack has no death take). Render death with "
            "voice=male, or bundle a CC0 female death gasp to lift this."
        )
    vocal, name = _vocal((PAIN_BANK,), body, rng, intensity,
                         bank_longer_than=0.9)
    return vocal, [name]


def effort(body: Body, voice: Voice, rng: np.random.Generator,
           intensity: float = 0.85) -> tuple[np.ndarray, list[str]]:
    """A swing or lift grunt -- exertion, not pain.

    Male is HaelDB's short grunts; female is cicifyre's "jump" reactions. Both
    are the sound of a body putting effort out, which is what a sword swing or
    a shove wants under it.
    """
    if voice.name == "female":
        vocal, name = _vocal(FEMALE_EFFORT, body, rng, intensity)
        return vocal, [name]
    vocal, name = _vocal(MALE_GRUNTS, body, rng, intensity)
    return vocal, [name]


def shout(body: Body, voice: Voice, rng: np.random.Generator,
          intensity: float = 0.95) -> tuple[np.ndarray, list[str]]:
    """A short vocal call or yell -- a rally, a challenge, a battle cry.

    HaelDB's yells, one per vocalist. Male only: cicifyre's pack has no yell
    and the pain bank has no call in it, so a female shout is an honest gap
    rather than a retuned grunt pretending to be one.
    """
    if voice.name != "male":
        raise ValueError(
            "shout is bundled male-only: no CC0 female yell was found. Render "
            "shout with voice=male, or bundle a CC0 female call to lift this."
        )
    vocal, name = _vocal(MALE_YELLS, body, rng, intensity)
    return vocal, [name]


# Cues with no bundled recording: registered so the gap is visible in `list`
# and honest at render time, per the fireball rule -- refuse, and say what
# recording would fix it, rather than approximate with the wrong sound.
ABSENT: dict[str, str] = {
    "breath": (
        "No CC0 isolated inhale/exhale was found downloadable. Freesound has "
        "clean CC0 breaths (e.g. SpliceSound 218309) but gates the master "
        "behind a login and only serves a lossy preview, which must not be "
        "bundled as a master. Bundle a CC0 breath from OpenGameArt, or a "
        "downloaded Freesound CC0 master, to lift this."
    ),
}


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def master(signal: np.ndarray, peak_db: float) -> list[float]:
    """Final chain shared by every cue: band-limit, soft-limit, fade, normalize.

    The high-pass takes out any DC and sub-rumble the varispeed or the chest
    left; the low-pass takes the very top off so a sped-up child voice does not
    get fizzy. A gentle tanh keeps a hard glottal onset from clipping before
    the normalize sets the peak.
    """
    data = np.asarray(signal, dtype=float)
    if len(data) == 0:
        return []
    data = sosfilt(butter(2, 55.0, btype="highpass", fs=SR, output="sos"), data)
    data = sosfilt(butter(2, 15_000.0, btype="lowpass", fs=SR, output="sos"), data)
    drive = 1.4
    data = np.tanh(data * drive) / np.tanh(drive)
    data = _fades(data, 0.005)
    peak = float(np.max(np.abs(data)))
    if peak > 1e-12:
        data = data * (10.0 ** (peak_db / 20.0) / peak)
    return [float(x) for x in np.clip(data, -0.98, 0.98)]


def body_named(name: str) -> Body:
    try:
        return BODIES[name]
    except KeyError:
        raise ValueError(
            f"unknown body {name!r}; expected one of {sorted(BODIES)}"
        ) from None


def voice_named(name: str) -> Voice:
    try:
        return VOICES[name]
    except KeyError:
        raise ValueError(
            f"unknown voice {name!r}; expected one of {sorted(VOICES)}"
        ) from None


def bundled_sources() -> tuple[str, ...]:
    return asset_names(SOURCE_DIRECTORY)


def referenced_sources() -> tuple[str, ...]:
    """Every recording a cue can reach, for the orphan check."""

    names = {PAIN_BANK, *MALE_GRUNTS, *MALE_YELLS, *FEMALE_HURT, *FEMALE_EFFORT}
    return tuple(sorted(names))


RENDERERS: dict[str, Callable[..., tuple[np.ndarray, list[str]]]] = {
    "hurt": hurt,
    "death": death,
    "effort": effort,
    "shout": shout,
}
