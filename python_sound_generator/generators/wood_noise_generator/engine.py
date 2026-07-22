"""Everything wood does, built from real CC0 recordings.

A hybrid generator in the lineage of ``door_sound_generator``,
``creature_foley_generator``, ``unarmed_foley_generator`` and
``fire_sound_generator``: NumPy and SciPy trim, retune, filter and layer
bundled recordings rather than synthesizing wood from oscillators.

Why this is not synthesized
---------------------------
``weapons_destruction_generator`` had a wood impact and wood debris. Both were
auditioned by ear on 2026-07-15 and both were rejected; the retired comparison
was preserved outside the runtime package. The reason is structural rather than
a tuning miss:
``dsp.modal_ring`` sums pure decaying sinusoids, which is the textbook
construction of a *bell*, so a struck plank came out as a chime with a click in
front of it.

Two finer theories were formed during that audit and both were falsified -- that
the pure-sine primitives were to blame (the sine-free dirt path failed anyway),
and that mode gain predicted the verdict (wood sits between stone and metal at
0.58 and failed regardless). The rule that survives is the coarse one, and it
has resisted refinement six times: procedural DSP wins on anything periodic and
loses on chaotic transients.

Wood is the chaotic half twice over. A struck plank is a contact, a body
resonance and a rattle arriving within milliseconds. A breaking plank is
cellulose fibres failing in sequence -- hundreds of tiny independent fractures,
which is precisely what no oscillator bank reproduces and what a recording gets
for free.

What this generator owns
------------------------
The provider boundary is explicit: a **material** generator owns
what a material *is* -- its contact, its ring, its fracture, its debris -- and an
**event** generator owns the *composition*: which materials, at
what scale, in what order. So there is no ``bullet_hits_plank`` here. There is a
plank being struck, and the consuming event decides what struck it.

Two adjacencies are deliberately not crossed. ``door_sound_generator`` owns
door creaks, which are *hinges* -- metal friction, not timber -- and it owns
wooden doors closing, which is why the archives' ``wood_close_*`` recordings are
not bundled. ``footstep_generator`` owns a foot on a wooden floor.

Four lessons are inherited from item 6's crash and are not rediscovered here:

- **Mass is varispeed.** Resampling moves pitch and decay together, which is what
  makes a beam a beam; pitch-shifting alone leaves the decay behind and yields a
  slowed-down twig.
- **Weight is not mass.** How much of a thing arrives on one contact patch is a
  separate axis from how much it masses. A crate is light and lands on a whole
  face.
- **Lead with the contact, not the clatter.** A break that opens with its
  splinters reads as someone dropping a bag of kindling.
- **Recordings have nothing below ~80 Hz.** These are library recordings of small
  objects. Weight is synthesised underneath -- noise through a resonance, never a
  sine, because a sine held for a few cycles at 60 Hz is a tuned drum and that is
  the exact failure this generator exists to avoid.

The recordings and their licensing live in ``sources/provenance.json``.

Every cue returns unmastered mono floating-point PCM plus the source files it
consumed. The cue registry owns mastering, so every public render follows one
contract and the consumed filenames reach the manifest.
"""

from __future__ import annotations

from ..._audio.assets import asset_names

from ..._audio.assets import resolve_asset

from ..._audio.foley_numpy import read_recording

from ..._audio.foley_numpy import write_wav_scipy as write_wav  # noqa: F401

from ..._audio.foley_numpy import trim_activity as trim_recording_activity

from ..._audio.foley_numpy import fade_edges, mix_at_frames, varispeed_configured


import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

SR = 44_100
SOURCE_DIRECTORY = Path(__file__).resolve().parent / "sources"


# --------------------------------------------------------------------------
# the recordings, by what the wood is doing
#
# Grouped by verb rather than by archive. Which collection a file came from is
# provenance's business; what it *is* is this module's. The names match
# provenance.json exactly, and a test fails the build if the two drift apart.
# --------------------------------------------------------------------------

STRIKE_SOURCES: tuple[str, ...] = (
    "rubberduck_wood_hit_01.wav",
    "rubberduck_wood_hit_02.wav",
    "rubberduck_wood_hit_03.wav",
    "rubberduck_wood_hit_04.wav",
    "rubberduck_wood_hit_05.wav",
    "rubberduck_wood_hit_06.wav",
    "rubberduck_wood_hit_07.wav",
    "rubberduck_wood_hit_08.wav",
    "rubberduck_wood_hit_09.wav",
    "rubberduck_break_wood_hit_01.wav",
    "rubberduck_break_wood_hit_02.wav",
    "rubberduck_break_wood_hit_03.wav",
)

# Kept apart from the hits rather than pooled with them, because each is a
# narrower thing and the cues want them by name. Measured centroids: hammer
# ~1.6 kHz, hit ~2.7 kHz, slam ~3.1 kHz -- so a hammer is not a bright small
# striker, it is a heavy timber being hit, and a slam is not a heavy event, it
# is a large thin face. `strike` and `drop` use them accordingly.
HAMMER_SOURCES: tuple[str, ...] = (
    "rubberduck_wood_hammer_01.wav",
    "rubberduck_wood_hammer_02.wav",
)

SLAM_SOURCES: tuple[str, ...] = (
    "rubberduck_wood_slam_01.wav",
    "rubberduck_wood_slam_02.wav",
    "rubberduck_wood_slam_03.wav",
    "rubberduck_wood_slam_04.wav",
)

BREAK_SOURCES: tuple[str, ...] = (
    "rubberduck_wood_breaking_01.wav",
    "rubberduck_wood_breaking_02.wav",
    "rubberduck_break_wood_breaking_01.wav",
    "rubberduck_break_wood_breaking_02.wav",
    "rubberduck_break_wood_breaking_03.wav",
    "rubberduck_break_wood_breaking_04.wav",
)

# Cracking is fracture without collapse: the bright fibre-tearing top end, with
# almost nothing below 500 Hz. That is why `splinter` is its own cue and why
# `break` layers these *over* a breaking recording rather than instead of one.
SPLINTER_SOURCES: tuple[str, ...] = (
    "rubberduck_wood_cracking_01.wav",
    "rubberduck_wood_cracking_02.wav",
    "rubberduck_wood_cracking_03.wav",
    "rubberduck_wood_cracking_04.wav",
)

FALL_SOURCES: tuple[str, ...] = (
    "rubberduck_wood_falling_01.wav",
    "rubberduck_wood_falling_02.wav",
    "rubberduck_wood_falling_03.wav",
    "rubberduck_wood_falling_04.wav",
    "rubberduck_wood_falling_05.wav",
    "rubberduck_wood_falling_06.wav",
    "rubberduck_break_wood_falling_01.wav",
    "rubberduck_break_wood_falling_02.wav",
)

# Wood under load, sustained. The long one carries the cue and the two short
# squeaks are stress events punctuating it -- see `creak`.
CREAK_SOURCES: tuple[str, ...] = ("antumdeluge_tree_creak.wav",)

STRESS_SOURCES: tuple[str, ...] = (
    "rubberduck_wood_squeak_01.wav",
    "rubberduck_wood_squeak_02.wav",
)

# Approved by ear on 2026-07-15 from Freesound's public HQ previews, which are
# lossy and must never be bundled. Freesound gates the masters behind a login,
# so these two arrive by hand -- the same route egomassive_tire took into
# aircraft_spacecraft_generator. Until they land, `scrape` refuses to render and
# says why; it is absent rather than approximated, which is the choice item 8
# made with `fireball` and item 2 made with these very impacts.
SCRAPE_SOURCES: tuple[str, ...] = (
    "rutgermuller_wood_plank_scraping.wav",
    "lampeight_wood_panel_scrape.wav",
)


@dataclass(frozen=True)
class WoodForm:
    """What piece of wood this is, and what that does to the recordings.

    `speed` is varispeed and does the real work: resampling moves pitch and
    decay together, which is what separates a twig from a beam. Pitch-shifting
    alone would leave the decay behind and turn a beam into a slowed-down twig.

    `damping_db` rotates the recording's spectrum about a pivot inside its own
    band: mass absorbs high end, and that is separate from being pitched lower.
    Without it a beam is a slowed-down twig, which is the failure item 6 names.
    It redistributes recorded energy and adds nothing. See `_damping` for why it
    is a tilt and not the band-mixing the sibling generators use.

    `body` is the mass scalar and does no filtering: it sets how long a cue
    runs, how fast a bounce settles, and how long the weight under it decays.

    `sub_hz` and `sub_gain` are the one synthesized layer, and they exist
    because the recordings are library captures of small objects. That claim was
    checked against this material rather than inherited: across the 36 bundled
    impact recordings the median energy below 80 Hz is **0.2 %**, and only two
    exceed 3 %. A falling beam has a great deal. So it is noise through a
    resonance, never a sine.

    `pieces` scales debris count: a beam does not break into the same number of
    fragments as a twig, and that count is as much of the cue as the timbre.
    """

    name: str
    label: str
    description: str
    speed: float
    body: float
    damping_db: float
    sub_hz: float
    sub_gain: float
    pieces: int


FORMS: dict[str, WoodForm] = {
    "twig": WoodForm(
        "twig", "Twig", "a dry stick or branch: fast, bright, no body",
        1.42, 0.10, 5.0, 0.0, 0.0, 3,
    ),
    "plank": WoodForm(
        "plank", "Plank", "a board, a fence rail, a shelf",
        1.00, 0.50, 0.0, 96.0, 0.30, 6,
    ),
    "crate": WoodForm(
        "crate", "Crate", "a hollow box: light, but lands on a whole face",
        0.88, 0.78, -3.0, 78.0, 0.55, 9,
    ),
    "beam": WoodForm(
        "beam", "Beam", "a structural timber, a mast, a rafter",
        0.66, 1.15, -8.0, 58.0, 1.00, 12,
    ),
}

# The form a form-flat cue renders at. `plank` is speed 1.0 and damping 0, so it
# is the neutral point: rendering `splinter` here applies no varispeed and no
# tilt, which is the only version of that cue that passed by ear.
NEUTRAL_FORM = "plank"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------



def _trim_to_onset(signal: np.ndarray) -> np.ndarray:
    return trim_recording_activity(
            signal, sample_rate=SR, window_seconds=0.004,
            threshold_ratio=0.06, pre_seconds=0.004, post_seconds=0.010,
        )


@lru_cache(maxsize=64)
def _load_source(filename: str) -> np.ndarray:
    path = resolve_asset(SOURCE_DIRECTORY / filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing bundled wood source {path}; see sources/provenance.json"
        )
    _, signal = read_recording(
        path, expected_rate=SR,
        trim=lambda data, rate: _trim_to_onset(data),
        label="wood source",
    )
    return signal


def bundled_sources() -> tuple[str, ...]:
    if not SOURCE_DIRECTORY.is_dir():
        return ()
    return asset_names(SOURCE_DIRECTORY)


def missing_sources(required: Sequence[str]) -> tuple[str, ...]:
    present = set(bundled_sources())
    return tuple(name for name in required if name not in present)


# --------------------------------------------------------------------------
# shaping
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


# Where a wood recording's energy actually sits, near enough: the bundled hits
# centroid around 2.7 kHz. The pivot has to be *inside* that band, and it moves
# with varispeed because varispeed moves the band.
PIVOT_HZ = 2_000.0


def _damping(signal: np.ndarray, gain_db: float, pivot: float) -> np.ndarray:
    """Damping: rotate the recording's spectrum about `pivot`, in decibels.

    Mass damps. A beam absorbs its own high end where a twig radiates it, and
    that is a different thing from being pitched lower -- varispeed alone gives
    a slowed-down twig, which is the failure item 6 names. This is the layer
    that makes a beam a beam rather than a big twig.

    A *tilt* -- high shelf up by half, low shelf down by half -- and not the
    band-mixing that `unarmed_foley_generator` and `fire_sound_generator` use.
    That was tried here first and measured, and it does not work on this
    material:

    - `signal + brightness * highpassed(signal)` moved a strike's centroid by
      **2.9 %** at the twig's setting and **-1.8 %** at the beam's: a 4.7 % span
      across the entire form range, which is not an axis but a rounding error.
    - Its body term is worse than useless. It lowpasses at 220 Hz and adds the
      result back, but these recordings have a median **0.2 %** of their energy
      below 80 Hz and little below 200 Hz, so it amplifies almost nothing.
      Measured, `body=1.15` moved the centroid 0.7 %.

    `stone_noise_generator` reached this independently and its notes say to start
    here rather than rediscover it; the numbers above are this generator paying
    the toll anyway before reading them. A tilt works because it always cuts one
    side while lifting the other, so there is no net gain for `_prepare`'s
    normalisation to eat, and the pivot sits inside the band that carries the
    sound rather than under it.

    Nothing synthetic is added: this is the recording's own energy, redistributed.
    Weight that is genuinely *not* in the recording is `_sub`'s job.
    """
    if abs(gain_db) < 0.05 or len(signal) == 0:
        return signal
    lifted = _shelf(signal, gain_db * 0.5, pivot, high=True)
    return _shelf(lifted, -gain_db * 0.5, pivot, high=False)


def _fades(signal: np.ndarray, seconds: float=0.003) -> np.ndarray:
    return fade_edges(signal, seconds, sample_rate=SR)


def _decay(signal: np.ndarray, seconds: float, hold: float = 0.0) -> np.ndarray:
    """Impose an exponential decay, so a library tail cannot outstay its cue."""

    if seconds <= 0.0 or len(signal) == 0:
        return signal
    t = np.arange(len(signal)) / SR
    return signal * np.exp(-np.maximum(0.0, t - hold) / seconds)


def _choose(rng: np.random.Generator, choices: Sequence[str]) -> str:
    return choices[int(rng.integers(0, len(choices)))]


def _draw(rng: np.random.Generator, choices: Sequence[str],
          count: int) -> list[str]:
    """Pick `count` recordings, exhausting the pool before repeating any.

    Independent draws are wrong for a scatter. Picking three pieces at random
    from twelve returns the same file twice about a quarter of the time, and
    three pieces built from one recording read as one piece stuttering rather
    than as three -- the same "a replayed sample is recognisably the same
    sample" problem `strike` solves by layering two. Varispeed hides some of it
    and not enough of it. Drawing without replacement makes a small scatter
    always sound like distinct objects; a large one still repeats, but by then
    there is enough going on that it does not read.
    """
    out: list[str] = []
    while len(out) < count:
        for index in rng.permutation(len(choices)):
            out.append(choices[int(index)])
            if len(out) == count:
                break
    return out


def _mix_at(destination: np.ndarray, source: np.ndarray, start_frame: int, gain: float) -> None:
    mix_at_frames(destination, source, start_frame, gain)


def _sub(form: WoodForm, rng: np.random.Generator, seconds: float,
         decay: float) -> np.ndarray:
    """The weight under the foley, and the one synthesized layer here.

    Noise through a resonance, never a sine. The distinction is the whole
    reason this generator exists: a decaying sine at 60 Hz is a tuned drum, and
    that pitch is what got item 2's impacts rejected. Filtered noise has the
    same weight and no note.
    """
    if form.sub_gain <= 0.0 or form.sub_hz <= 0.0:
        return np.zeros(0)
    n = _frames(seconds)
    noise = rng.standard_normal(n)
    low = max(24.0, form.sub_hz * 0.55)
    high = min(form.sub_hz * 2.1, 380.0)
    body = sosfilt(
        butter(2, [low, high], btype="bandpass", fs=SR, output="sos"), noise
    )
    t = np.arange(n) / SR
    body = body * np.exp(-t / max(decay, 1e-4))
    peak = float(np.max(np.abs(body))) if len(body) else 0.0
    if peak > 1e-9:
        body = body / peak
    return _fades(body, 0.004)


def _prepare(filename: str, form: WoodForm, rng: np.random.Generator,
             speed_multiplier: float = 1.0,
             jitter: tuple[float, float] = (0.955, 1.048)) -> np.ndarray:
    """Load one recording, retuned and tilted for this piece of wood."""

    cached = _load_source(filename)
    speed = form.speed * speed_multiplier * float(rng.uniform(*jitter))
    signal = _varispeed(cached, speed)
    # After varispeed, not before: resampling moves the band the pivot has to
    # sit inside, so the pivot moves with it.
    signal = _damping(signal, form.damping_db, PIVOT_HZ * form.speed)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal = signal / peak
    return _fades(signal)


# --------------------------------------------------------------------------
# cues
# --------------------------------------------------------------------------

def strike(form: WoodForm, rng: np.random.Generator,
           intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Wood struck, without breaking.

    Two recordings, not one: the second sits a few milliseconds behind the
    first, quieter and retuned away from it. A single sample replayed is
    recognisably the same strike every time; two staggered ones read as one
    thicker contact while giving the rotation somewhere to vary. The offset
    stays under ~12 ms because past that the ear stops fusing them and hears a
    double tap -- the same bound `unarmed_foley_generator` found.

    A beam leads with `hammer` and everything else with `hit`. The reason is
    measured rather than assumed: the hammer recordings are the darkest material
    bundled here, ~1.6 kHz against the hits' ~2.7 kHz, because a hammer is
    swung at something massive and the recording captures *both ends* of that.
    The heavy timber on the far end is the cue; the hammer is incidental.

    That is worth stating because it was wired backwards first. Twig led with
    the hammer, on the tidy reasoning that a small thing is struck by something
    small and hard -- and it made a twig measurably *darker* than a plank, which
    is the opposite of a twig. A recording is the striker and the struck thing
    together, and only one of them is what the cue is about.

    Everything else about the form rides varispeed and the tilt, and it is left
    to. `slam` was tried as the crate's lead here on the "weight is not mass"
    reasoning -- a crate is a whole face arriving -- and it was wrong twice: it
    is the *brightest* and thinnest family bundled, so a struck crate came out
    with less low end than a struck plank (14.1% against 14.6% below 160 Hz)
    when a crate's entire identity is that it is hollower. The tilt cannot fix
    that, because it can only amplify low end the recording already has, and
    these recordings have almost none. Dropped, the crate reads correctly
    (19.9%). `slam` earns its place in `drop`, where a large face really is what
    arrives.
    """
    lead_family = HAMMER_SOURCES if form.name == "beam" else STRIKE_SOURCES

    primary = _choose(rng, lead_family)
    secondary = _choose(rng, tuple(s for s in STRIKE_SOURCES if s != primary))

    lead = _prepare(primary, form, rng)
    back = _prepare(secondary, form, rng, speed_multiplier=0.90)

    total = max(len(lead), len(back) + _frames(0.012)) + _frames(0.02)
    out = np.zeros(total)
    _mix_at(out, lead, 0, 1.0)
    _mix_at(out, back, _frames(float(rng.uniform(0.004, 0.012))),
            0.40 * intensity)

    weight = _sub(form, rng, min(total / SR, 0.42), 0.05 + 0.05 * form.body)
    if len(weight):
        _mix_at(out, weight, _frames(0.003), 0.5 * form.sub_gain * intensity)
    return _fades(out, 0.004), [primary, secondary]


def fracture(form: WoodForm, rng: np.random.Generator,
             intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Wood breaking: the failure, the fibres tearing, and the pieces after.

    Lead with the contact and let the splinters follow. Opening with the
    splinters is item 6's dropped-bag-of-gravel mistake in another material:
    the break is the event, and the kindling is a consequence of it.

    The tear is what makes this wood rather than stone. Stone shatters -- it is
    over at the contact. Wood is fibrous, so the failure *runs*: it keeps
    tearing along the grain after the thing that broke it has stopped, which is
    why the cracking layer is held with a long decay rather than cut short.
    """
    primary = _choose(rng, BREAK_SOURCES)
    seconds = 0.9 + 0.7 * min(form.body, 1.2) + 0.4 * intensity
    out = np.zeros(_frames(seconds))
    sources = [primary]

    lead = _prepare(primary, form, rng)
    _mix_at(out, lead, 0, 1.0)

    # Fibres letting go along the grain, a beat after the failure and running on.
    for index in range(1 + round(2 * intensity)):
        name = _choose(rng, SPLINTER_SOURCES)
        sources.append(name)
        tear = _prepare(name, form, rng,
                        speed_multiplier=float(rng.uniform(0.82, 1.10)))
        tear = _decay(tear, 0.30 + 0.25 * intensity, hold=0.05)
        _mix_at(out, tear, _frames(0.02 + 0.09 * index +
                                   float(rng.uniform(0.0, 0.03))),
                (0.55 - 0.12 * index) * intensity)

    # The pieces arriving. Fewer and later than `debris` proper: this is the
    # break's own tail, not a scatter cue.
    for name in _draw(rng, FALL_SOURCES, max(2, form.pieces // 2)):
        sources.append(name)
        piece = _prepare(name, form, rng,
                         speed_multiplier=float(rng.uniform(0.95, 1.45)))
        piece = _decay(piece, float(rng.uniform(0.08, 0.22)))
        start = 0.14 + float(rng.random()) ** 0.7 * (seconds - 0.3)
        _mix_at(out, piece, _frames(start),
                float(rng.uniform(0.10, 0.30)) * (1.0 - 0.4 * start / seconds))

    weight = _sub(form, rng, min(seconds, 0.55), 0.07 + 0.06 * form.body)
    if len(weight):
        _mix_at(out, weight, _frames(0.004), 0.7 * form.sub_gain * intensity)
    return _fades(out, 0.004), sources


def splinter(form: WoodForm, rng: np.random.Generator,
             intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Fracture without collapse: wood cracking under strain, still in one piece.

    The cracking recordings carry almost nothing below 500 Hz, and that is the
    cue rather than a deficiency -- a crack is the grain letting go, not the
    body of the thing moving. So this deliberately does not get the sub layer
    that `strike` and `fracture` do. Adding weight here would make it a break.

    **This cue is form-flat, and that was decided by ear.** On 2026-07-16 it was
    auditioned at all four forms and only `plank` -- the form with no varispeed
    and no damping -- passed; beam, crate and twig were all rejected. A crack is a
    precise-template sound, the same class as item 3's tire squeal: the ear holds
    an exact model of it, so any retuning to suit a form reads as fake, the way
    stone's ring and a punch's low end do not. The cue registry enforces this by
    always calling with the neutral form, so `form` arrives as `plank` whatever
    the caller asked for. The parameter stays in the signature only so every cue
    shares one shape; the varispeed and damping it drives are, here, both zero.
    """
    primary = _choose(rng, SPLINTER_SOURCES)
    signal = _prepare(primary, form, rng)
    signal = sosfilt(butter(2, 260.0, btype="highpass", fs=SR, output="sos"),
                     signal)
    signal = _damping(signal, 4.0, PIVOT_HZ * form.speed)
    signal = _decay(signal, 0.18 + 0.14 * intensity, hold=0.03)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal = signal / peak
    return _fades(signal * intensity, 0.004), [primary]


def drop(form: WoodForm, rng: np.random.Generator,
         intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """A piece of wood landing, and settling.

    Wood bounces. It is light and stiff, so it does not arrive and stop the way
    stone does -- it lands, lifts, and lands again, closer and quieter each
    time. The decreasing gap is the whole identity of the cue, and it is why
    this is not just `strike` with a different source.

    A crate or a beam leads with `slam`, which is where that family belongs:
    a slam *is* a large wooden face arriving, which is what these two forms do
    and what the loose `falling` recordings -- small pieces tumbling -- do not.
    It is worth the swap. It roughly doubles the low end on both forms (crate
    6.9% to 13.9% below 160 Hz, beam 13.0% to 23.0%), because the cue was
    previously built from recordings of objects far smaller than the thing it
    claimed to be.
    """
    lead_family = SLAM_SOURCES if form.name in ("crate", "beam") else FALL_SOURCES
    primary = _choose(rng, lead_family)
    seconds = 0.6 + 0.5 * min(form.body, 1.2)
    out = np.zeros(_frames(seconds))
    sources = [primary]

    lead = _prepare(primary, form, rng)
    _mix_at(out, lead, 0, 1.0)

    # Settling: each contact sooner after the last, and quieter.
    onset = 0.0
    gap = 0.10 + 0.07 * form.body
    for index in range(1 + round(2 * intensity)):
        onset += gap * float(rng.uniform(0.8, 1.2))
        gap *= 0.62
        name = _choose(rng, FALL_SOURCES)
        sources.append(name)
        bounce = _prepare(name, form, rng,
                          speed_multiplier=float(rng.uniform(1.0, 1.3)))
        bounce = _decay(bounce, 0.10)
        _mix_at(out, bounce, _frames(onset), 0.42 * (0.55 ** index) * intensity)

    weight = _sub(form, rng, min(seconds, 0.45), 0.05 + 0.05 * form.body)
    if len(weight):
        _mix_at(out, weight, _frames(0.003), 0.6 * form.sub_gain * intensity)
    return _fades(out, 0.004), sources


def debris(form: WoodForm, rng: np.random.Generator,
           intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """The pieces, scattering and coming to rest.

    Count is by form and it is as much of the cue as the timbre: a beam does not
    break into as many fragments as a twig does, and no amount of retuning makes
    twelve pieces sound like three. The pieces thin out rather than stopping --
    ``random ** 0.7`` clusters them early, because everything is thrown at once
    and then arrives over time.
    """
    seconds = 1.1 + 0.9 * intensity + 0.3 * min(form.body, 1.2)
    out = np.zeros(_frames(seconds))
    sources: list[str] = []

    count = max(3, round(form.pieces * (0.5 + 0.7 * intensity)))
    names = _draw(rng, FALL_SOURCES + SPLINTER_SOURCES, count)

    # The first piece lands on the trigger. Drawing every start independently
    # left a twig's three pieces beginning up to 650 ms late, because with only
    # a handful of draws nothing guarantees one of them is early -- so a cue a
    # game fires on contact opened with two thirds of a second of silence. The
    # rest scatter behind it; `** 0.7` clusters them early, because everything
    # is thrown at once and then arrives over time.
    starts = [0.0] + sorted(float(rng.random()) ** 0.7 * seconds * 0.82
                            for _ in range(count - 1))

    for name, start in zip(names, starts):
        sources.append(name)
        piece = _prepare(name, form, rng,
                         speed_multiplier=float(rng.uniform(0.9, 1.9)))
        piece = _decay(piece, float(rng.uniform(0.05, 0.20)))
        _mix_at(out, piece, _frames(start),
                float(rng.uniform(0.12, 0.36)) * (1.0 - 0.5 * start / seconds))
    return _fades(out, 0.004), sources


def creak(form: WoodForm, rng: np.random.Generator,
          intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Wood under load: a mast, a rafter, a deck taking weight.

    The bed is a tree creaking in wind, which is wood loaded and complaining
    exactly as a rafter is. A window of it is taken rather than the whole
    recording, so the cue varies per seed instead of being the same bed every
    time. The window lands on real material because `_load_source` has already
    cut the recording's dead tail: the file runs 6.58 s but stops creaking at
    4.64 s, and untrimmed roughly a third of the seeds drew silence.

    Note what this is *not*. `door_sound_generator` owns door creaks, and those
    are **hinges** -- metal on metal, a different sound with a different
    physics. This is timber. The two are adjacent and must not be confused: a
    game wanting a creaking door wants that generator, and a game wanting a
    groaning ship's hull wants this one.

    The short squeaks punctuate rather than carry. Wood under load does not
    complain steadily; it holds, slips, and holds again.
    """
    bed_name = _choose(rng, CREAK_SOURCES)
    bed = _load_source(bed_name)
    sources = [bed_name]

    seconds = 1.4 + 1.6 * intensity
    want = _frames(seconds / max(form.speed, 0.05))
    if want < len(bed):
        start = int(rng.integers(0, len(bed) - want))
        bed = bed[start:start + want]
    signal = _varispeed(np.asarray(bed, dtype=float), form.speed)
    signal = _damping(signal, form.damping_db, PIVOT_HZ * form.speed)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal = signal / peak

    out = np.zeros(len(signal) + _frames(0.05))
    _mix_at(out, signal, 0, 0.85)

    # Stress events: it holds, slips, holds again.
    for _ in range(1 + round(2 * intensity)):
        name = _choose(rng, STRESS_SOURCES)
        sources.append(name)
        stress = _prepare(name, form, rng,
                          speed_multiplier=float(rng.uniform(0.7, 1.1)))
        at = float(rng.uniform(0.1, 0.75)) * len(signal) / SR
        _mix_at(out, stress, _frames(at), 0.30 * intensity)
    return _fades(out, 0.02), sources


def scrape(form: WoodForm, rng: np.random.Generator,
           intensity: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """Wood dragged: a crate pushed, a plank hauled across a floor.

    Sustained friction, so a window of a recording carries it the way `creak`
    works rather than the way `strike` does.

    The sources for this were approved by ear on 2026-07-15 from Freesound's
    public HQ previews and are fetched by hand, because Freesound gates masters
    behind a login. If they are not present this raises rather than substituting
    something else: OpenGameArt's obvious candidate turned out to be an object
    dragged across a **cinder block**, which is item 13's material and would have
    put concrete in a wood library. See provenance.json's `excluded`.
    """
    missing = missing_sources(SCRAPE_SOURCES)
    if missing:
        raise FileNotFoundError(
            "scrape needs recordings that are not bundled yet: "
            + ", ".join(missing)
            + ". They were approved by ear from Freesound previews but the "
              "masters need a login to download; see sources/provenance.json "
              "`pending` for the pages and the expected filenames."
        )

    name = _choose(rng, SCRAPE_SOURCES)
    bed = _load_source(name)
    seconds = 0.5 + 1.3 * intensity
    want = _frames(seconds / max(form.speed, 0.05))
    if want < len(bed):
        start = int(rng.integers(0, len(bed) - want))
        bed = bed[start:start + want]
    signal = _varispeed(np.asarray(bed, dtype=float), form.speed)
    signal = _damping(signal, form.damping_db, PIVOT_HZ * form.speed)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal = signal / peak

    out = np.zeros(len(signal) + _frames(0.04))
    _mix_at(out, signal, 0, 1.0)
    weight = _sub(form, rng, min(len(signal) / SR, 0.5), 0.20)
    if len(weight):
        _mix_at(out, weight, 0, 0.35 * form.sub_gain * intensity)
    return _fades(out, 0.012), [name]


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def master(signal: np.ndarray, peak_db: float) -> list[float]:
    """Final chain shared by every cue: soft limit, fade, normalize."""

    data = np.asarray(signal, dtype=float)
    if len(data) == 0:
        return []
    drive = 1.25
    data = np.tanh(data * drive) / np.tanh(drive)
    data = _fades(data, 0.003)
    peak = float(np.max(np.abs(data)))
    if peak > 1e-12:
        data = data * (10.0 ** (peak_db / 20.0) / peak)
    return [float(x) for x in np.clip(data, -0.98, 0.98)]


def form_named(name: str) -> WoodForm:
    if name not in FORMS:
        raise ValueError(
            f"unknown form {name!r}; expected one of {sorted(FORMS)}"
        )
    return FORMS[name]
