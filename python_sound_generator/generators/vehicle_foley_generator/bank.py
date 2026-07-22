"""The recorded half of this generator: loading, pitching, and looping Foley.

Tire-on-pavement is the one part of a driving soundscape that is not
synthesised here.  It was, and it was auditioned and rejected: the synthesis had
the mechanism right — stick-slip, a tread-block series, the correct formants —
and still landed as a whoosh, then as a grinder.  A tire squeal is a car-sized
resonator being bowed by a road, and that is cheaper to record than to model.

So ``roll``, ``skid``, ``drift`` and ``spray`` play CC0 recordings out of
``sources/`` (see ``sources/provenance.json``).  Everything else in this package
— chassis, driveline, jumps, impacts — stays procedural, because those are
impacts and resonances, which synthesis is good at.

Nothing here needs numpy: the bank is WAV files through the standard library,
and the shaping is the same ``combat_generators`` DSP the rest of the package
uses.  Recordings replaced the tire *model*, not the toolchain.

Only four surfaces have recordings — asphalt, dirt, gravel, wet — because no CC0
material exists for the rest.  The other five are derived rather than faked:
each :class:`SurfaceProfile` already carries the band and damping that define
it, so snow is the dirt bed heard through snow's own numbers.  See :func:`bed`.
"""

from __future__ import annotations

from ..._audio.assets import resolve_asset

import array
import math
import random
import wave
from functools import lru_cache
from pathlib import Path

from ..._audio import dsp


SOURCES = Path(__file__).resolve().parent / "sources"

# Recorded beds, and which surface borrows which. A surface with no recording of
# its own takes the nearest one it could plausibly be a version of: sand, grass
# and snow are all dirt with progressively more damping, mud is the Landcruiser
# bed (whose source drives through mud), and concrete is asphalt with a harder,
# brighter band. The borrowing is not the derivation — the profile is.
BEDS: dict[str, str] = {
    "asphalt": "squareal_roll_asphalt",
    "concrete": "squareal_roll_asphalt",
    "wet": "klau78_roll_wet",
    "dirt": "felixblume_roll_dirt",
    "gravel": "felixblume_roll_gravel",
    "sand": "felixblume_roll_dirt",
    "mud": "felixblume_roll_gravel",
    "grass": "felixblume_roll_dirt",
    "snow": "felixblume_roll_dirt",
}

# The squeal bank. One car, one session, four takes: they sit together because
# they are the same tires. The distant pair are a different perspective and are
# kept apart from the close ones rather than mixed into them.
SQUEALS: tuple[str, ...] = (
    "audibleedge_squeal_01",
    "audibleedge_squeal_02",
    "audibleedge_squeal_03",
    "audibleedge_squeal_04",
)

DISTANT_SQUEALS: tuple[str, ...] = (
    "lukacafuka_screech_distant",
    "aaronsawyer_squeal_chevy",
)

# Loose material being thrown, as opposed to rolled over. Mud is the only one of
# these recorded doing exactly that -- a car stuck with its wheels spinning -- and
# it is the only such recording in CC0 at all: `car mud spray` and `tire mud`
# both return nothing. The rest still borrow a driving or turning take.
SPRAYS: dict[str, str] = {
    "gravel": "erwil_gravel_turn",
    "mud": "craigsmith_mud_spin",
    "dirt": "felixblume_roll_dirt",
    "sand": "felixblume_roll_dirt",
    "snow": "felixblume_roll_dirt",
    "grass": "felixblume_roll_dirt",
    "asphalt": "erwil_gravel_turn",
    "concrete": "erwil_gravel_turn",
    "wet": "klau78_roll_wet",
}

# The squeals were recorded from a car whose tires squealed around here. Pitch
# ratios are taken against this, so `squeal_frequency` keeps meaning what it
# means everywhere else in the package.
SOURCE_SQUEAL_HZ = 950.0

CROSSFADE = 0.25


class MissingSource(RuntimeError):
    """A bundled recording is absent. Never a fallback: it is a broken install."""


@lru_cache(maxsize=None)
def load(stem: str) -> tuple[float, ...]:
    """Read one bundled recording as mono floating-point PCM.

    Cached: a pack render asks for the same handful of beds hundreds of times,
    and these are seconds-long files.
    """

    path = resolve_asset(SOURCES / f"{stem}.wav")
    if not path.is_file():
        raise MissingSource(
            f"bundled recording {stem!r} is missing from {SOURCES}; "
            "see sources/provenance.json for what belongs there"
        )
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise MissingSource(f"{stem!r} is not mono 16-bit PCM")
        if handle.getframerate() != dsp.SR:
            raise MissingSource(f"{stem!r} is not {dsp.SR} Hz")
        frames = array.array("h")
        frames.frombytes(handle.readframes(handle.getnframes()))
    return tuple(sample / 32768.0 for sample in frames)


def resample(signal, ratio: float) -> list[float]:
    """Resample by linear interpolation; pitch and duration move together.

    Deliberately varispeed rather than a pitch shift that preserves length.  A
    tire turning faster *is* pitched up and its texture *does* come at you
    quicker — a formant-preserving shift would keep the wrong thing constant.
    """

    if ratio <= 0.0:
        raise ValueError("resample ratio must be positive")
    if abs(ratio - 1.0) < 1e-9:
        return list(signal)
    count = int(len(signal) / ratio)
    if count < 2:
        raise ValueError("resample ratio leaves nothing behind")
    last = len(signal) - 1
    out = []
    for index in range(count):
        position = index * ratio
        left = int(position)
        frac = position - left
        right = min(left + 1, last)
        out.append(signal[left] * (1.0 - frac) + signal[right] * frac)
    return out


def loop(signal, duration: float, rng: random.Random) -> list[float]:
    """Cut a seamless loop of `duration` from a longer recording.

    This is the crossfade the synthesised loops in :mod:`.engine` do not need
    and must not use.  There, ``loopdsp`` makes a loop periodic by construction:
    it primes the filters and quantises the oscillators, because it owns them.
    A recording owns none of that.  It has no period to land on and no filter
    state to prime, so the only way its end meets its beginning is to fold one
    over the other.

    Equal power, because the two regions are uncorrelated — which is exactly the
    condition that makes it the right curve, and exactly what is not true of a
    quantised oscillator.
    """

    want = dsp.samples(duration)
    source = list(signal)
    if len(source) < 16:
        raise ValueError("recording is too short to loop")

    # The fade cannot outrun the recording. A ten-second drift is a real ask and
    # the shortest squeal take is under four, so this path is not hypothetical:
    # with a fade longer than the source, every fold consumed exactly what it
    # added and the build never grew.
    fade = max(2, min(dsp.samples(CROSSFADE), len(source) // 3))

    built = source
    while len(built) < want + fade:
        built = _fold(built, source, fade)

    start = rng.randrange(0, len(built) - want - fade + 1)
    body = list(built[start:start + want])
    tail = built[start + want:start + want + fade]
    for index in range(min(fade, len(tail), len(body))):
        position = index / max(1, fade - 1)
        head = math.sin(position * math.pi * 0.5)
        over = math.cos(position * math.pi * 0.5)
        body[index] = body[index] * head + tail[index] * over
    return body


def _fold(first: list[float], second: list[float], fade: int) -> list[float]:
    """Butt two takes together under an equal-power crossfade."""

    fade = min(fade, len(first), len(second))
    out = list(first[:len(first) - fade])
    for index in range(fade):
        position = index / max(1, fade - 1)
        out.append(
            first[len(first) - fade + index] * math.cos(position * math.pi * 0.5)
            + second[index] * math.sin(position * math.pi * 0.5)
        )
    return out + list(second[fade:])


def bed(vehicle, surface, speed: float) -> list[float]:
    """The recorded roll bed for a vehicle on a surface, derived if need be.

    Returns the whole shaped recording rather than a loop.  Cutting the loop is
    the caller's last step, because every filter here would otherwise start cold
    at the loop point — which is the one place in a loop nothing may.

    The derivation is the surface's own profile.  Every surface already declares
    the band it rolls in and how much it damps, and those numbers were tuned by
    ear; running the nearest recording through them is what turns dirt into snow.
    It is not a disguise — snow really is a duller, quieter, lower-banded version
    of the same contact — but it is a derivation, and the README says which five
    surfaces are derived rather than recorded.
    """

    source = load(BEDS[surface.name])
    # Faster tires are pitched up and their texture arrives quicker; heavier
    # vehicles ride lower and slower. Both are varispeed, because both are.
    ratio = (0.78 + 0.55 * speed) * (max(0.25, vehicle.mass) ** -0.12)
    voiced = resample(source, ratio)
    low = surface.roll_low * (0.72 + 0.38 * speed)
    high = surface.roll_high * (0.45 + 0.72 * speed)
    voiced = dsp.bandpass(voiced, low, high, 2)
    if surface.damping > 0.0:
        voiced = dsp.lowpass(
            voiced, max(400.0, high * (1.0 - 0.55 * surface.damping)), 2
        )
    return voiced


def take(signal, duration: float, rng: random.Random, tries: int = 6) -> list[float]:
    """Cut `duration` out of a recording, favouring a part with something in it.

    Random starts alone land in the gaps — these takes are a car doing skids,
    which means they are mostly a car *not* skidding.  Sampling a few candidates
    and keeping the loudest picks the event rather than the pause before it, and
    stays deterministic because the rng chooses the candidates.
    """

    want = dsp.samples(duration)
    if len(signal) <= want:
        return list(signal) + [0.0] * (want - len(signal))
    best = None
    for _ in range(tries):
        start = rng.randrange(0, len(signal) - want)
        chunk = signal[start:start + want]
        level = sum(sample * sample for sample in chunk)
        if best is None or level > best[0]:
            best = (level, chunk)
    return list(best[1])


def squeal(vehicle, rng: random.Random, distant: bool = False) -> list[float]:
    """One recorded tire squeal, pitched to the vehicle.

    ``squeal_frequency`` is a real number in the profile table — a kart's tires
    ring higher than a monster truck's — so the take is varispeeded by the ratio
    against the car that was actually recorded.
    """

    names = DISTANT_SQUEALS if distant else SQUEALS
    take = load(names[rng.randrange(len(names))])
    ratio = vehicle.squeal_frequency / SOURCE_SQUEAL_HZ
    # Keep it a car. Past about an octave either way a varispeeded squeal stops
    # sounding like rubber and starts sounding like a cartoon.
    ratio = max(0.55, min(1.9, ratio))
    return resample(take, ratio)


def spray_bed(surface, speed: float, rng: random.Random) -> list[float]:
    """The recorded bed of material being thrown up, per surface."""

    source = load(SPRAYS[surface.name])
    voiced = resample(source, 0.8 + 0.5 * speed)
    return dsp.bandpass(
        voiced,
        surface.grain_low * 0.8,
        surface.grain_high * (0.5 + 0.5 * speed),
        2,
    )
