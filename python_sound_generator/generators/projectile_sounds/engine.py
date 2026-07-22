"""Projectile impacts built from real CC0 recordings.

A hybrid generator in the lineage of ``door_sound_generator``,
``creature_foley_generator``, ``unarmed_foley_generator`` and
``aircraft_spacecraft_generator``: NumPy and SciPy trim, retune, filter and
layer bundled recordings rather than synthesizing an impact from oscillators.

Why this exists
---------------
``weapons_destruction_generator`` models ranged weapons properly, and its
reports, bursts, launches and flybys are excellent -- a report is a pressure
event with an envelope, a flyby is a Doppler rate, and modelling a rate is what
synthesis is genuinely good at.  Its *impacts* were not, and the reason is
structural rather than a tuning miss.  On 2026-07-15 eleven of its twelve
impact and debris renders were rejected by ear.  The construction was
``dsp.modal_ring`` -- a sum of pure decaying sinusoids, which is precisely how a
bell is synthesised -- over a ``sine_track`` sweep, which is precisely how a
beep is synthesised.  The listener's words were "too fake".

Two finer theories were formed during that audit and both were wrong, which is
worth recording because they are the obvious ones to have again:

* *the pure-sine primitives are the problem, so a sine-free path will survive.*
  The dirt path is sine-free.  It failed anyway.
* *mode gain predicts it: stone survives at 0.42, metal fails at 0.86.*  Wood
  sits between them at 0.58.  It failed too.

The primitive does not predict the verdict and neither does the mode gain.  The
whole class fails, and the rule that has held across six generators refuses to
be refined any finer than: **procedural DSP wins on anything periodic and loses
on chaotic transients.**  An impact is the archetypal chaotic transient --
contact, deformation and debris arriving milliseconds apart, each with its own
decay -- and the ear knows the pattern from a lifetime of dropping things.

So this generator owns the impacts and nothing else.
``weapons_destruction_generator`` stays stdlib-only and keeps everything it is
already better at; the pack registry here imports it for the reports and
flybys, so one command still fills a game's bank.

The model
---------
Two registries cross: a **projectile**, imported wholesale from
``weapons_destruction_generator`` so the names and the physics stay in sync, and
a **material**, defined here because a material's foley identity is which
recordings it is made of, which is not something that generator needed to know.

Weight is varispeed.  Resampling an impact moves its pitch and its decay
together, which is what makes a heavy projectile heavy; pitch-shifting alone
leaves the decay behind and yields a slowed-down light one.  Borrowed from
``creature_foley_generator``, where the same trick separates a small throat from
a large one.

Every renderer returns unmastered mono float PCM **plus the source files it
consumed**, so the manifest can name the recordings behind every artifact.  The
cue registry owns mastering.

The recordings and their licensing live in ``sources/provenance.json``. The
general rubberduck wood library is owned by ``wood_noise_generator`` and read
from its public source bank rather than duplicated here.
"""

from __future__ import annotations

from ..._audio.assets import asset_names, resolve_asset

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ..._audio.foley_numpy import (
    SR,
    darken as _darken,
    decay as _decay,
    load_wav,
    master,
    place as _place,
    resonant_sub,
    varispeed as _varispeed,
    write_wav,
)

from ..weapons_destruction_generator.engine import (
    PROJECTILES,
    ProjectileProfile,
    projectile_named,
)


SOURCE_DIRECTORY = Path(__file__).resolve().parent / "sources"
WOOD_SOURCE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "wood_noise_generator" / "sources"
)

# Item 12 owns these general-purpose wood recordings. Item 11 consumes them by
# name while keeping its fifteen Kenney impactWood masters local: those are
# projectile-contact recordings selected by impact mass, not the general wood
# material bank. Keeping this list explicit prevents unrelated wood sources
# from silently widening deterministic selection pools.
SHARED_WOOD_SOURCES: tuple[str, ...] = (
    *(f"rubberduck_wood_hit_{index:02d}.wav" for index in range(1, 10)),
    *(f"rubberduck_wood_cracking_{index:02d}.wav" for index in range(1, 5)),
    *(f"rubberduck_wood_falling_{index:02d}.wav" for index in range(1, 7)),
    *(f"rubberduck_wood_slam_{index:02d}.wav" for index in range(1, 5)),
    *(f"rubberduck_break_wood_breaking_{index:02d}.wav"
      for index in range(1, 5)),
    *(f"rubberduck_break_wood_falling_{index:02d}.wav"
      for index in range(1, 3)),
    *(f"rubberduck_break_wood_hit_{index:02d}.wav"
      for index in range(1, 4)),
)


@dataclass(frozen=True)
class MaterialProfile:
    """What a material is, expressed as the recordings it is made of.

    ``contact`` is the layer that leads and it is not negotiable that it leads:
    a projectile arriving is first and foremost a contact, and an impact that
    opens with its debris reads as someone dropping a bag of gravel.  That was
    learned the hard way on a lander touchdown in
    ``aircraft_spacecraft_generator``.

    ``sub_frequency`` is the one procedural layer left.  These are library
    recordings of small objects and there is nothing below about 80 Hz in any of
    them; a shell arriving has a great deal.  It is synthesised as noise through
    a resonance -- never a sine, for the reason in the module docstring.
    """

    name: str
    label: str
    description: str
    contact: tuple[str, ...]
    debris: tuple[str, ...]
    wet: tuple[str, ...] = ()
    speed: float = 1.0
    darken: float = 16_000.0
    sub_frequency: float = 70.0
    sub_gain: float = 1.0
    debris_count: int = 6
    debris_spread: float = 1.0


MATERIALS: dict[str, MaterialProfile] = {
    "metal": MaterialProfile(
        name="metal", label="Metal",
        description="plate and structure; rings, and keeps ringing",
        contact=("kenney_impact_metal_", "kenney_impact_plate_",
                 "rubberduck_metal_hit_"),
        debris=("rubberduck_break_metal_falling_", "rubberduck_break_metal_hit_",
                "rubberduck_sfx2_metal_"),
        speed=1.0, darken=16_000.0, sub_frequency=95.0, sub_gain=0.75,
        debris_count=7, debris_spread=1.0,
    ),
    "wood": MaterialProfile(
        name="wood", label="Wood",
        description="plank and beam; cracks rather than rings",
        contact=("kenney_impact_wood_", "kenney_impact_plank_",
                 "rubberduck_wood_hit_"),
        debris=("rubberduck_wood_cracking_", "rubberduck_break_wood_breaking_",
                "rubberduck_break_wood_falling_", "rubberduck_wood_falling_"),
        speed=1.0, darken=13_000.0, sub_frequency=80.0, sub_gain=0.7,
        debris_count=6, debris_spread=1.1,
    ),
    "stone": MaterialProfile(
        name="stone", label="Stone",
        description="masonry and rock; a dull crack and a shower of grit",
        contact=("rubberduck_sfx2_stones_", "kenney_impact_mining_",
                 "rubberduck_break_rock_breaking_"),
        debris=("rubberduck_break_rock_falling_",
                "rubberduck_break_rock_breaking_"),
        speed=0.95, darken=11_000.0, sub_frequency=75.0, sub_gain=0.85,
        debris_count=9, debris_spread=1.2,
    ),
    "dirt": MaterialProfile(
        name="dirt", label="Dirt",
        description="earth and sand; absorbs the hit and throws the rest",
        contact=("kenney_impact_soft_", "kenney_impact_mining_"),
        debris=("rubberduck_break_rock_falling_",),
        speed=0.9, darken=3_800.0, sub_frequency=60.0, sub_gain=1.0,
        debris_count=8, debris_spread=1.3,
    ),
    "water": MaterialProfile(
        name="water", label="Water",
        description="a hole punched in a surface that closes again",
        contact=("rubberduck_splash_",),
        debris=("rubberduck_splash_",),
        wet=("rubberduck_bubble_",),
        speed=1.0, darken=14_000.0, sub_frequency=55.0, sub_gain=0.5,
        debris_count=4, debris_spread=1.4,
    ),
    "flesh": MaterialProfile(
        name="flesh", label="Flesh",
        description="a body; blunt, wet, and entirely without a ring",
        contact=("independent_nu_hit",),
        debris=(),
        wet=("rubberduck_slime_",),
        speed=1.0, darken=6_500.0, sub_frequency=65.0, sub_gain=0.9,
        debris_count=0, debris_spread=1.0,
    ),
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def bundled_sources() -> tuple[str, ...]:
    local = set(asset_names(SOURCE_DIRECTORY))
    return tuple(sorted(local | set(SHARED_WOOD_SOURCES)))


def source_path(filename: str) -> Path:
    """Resolve a provenance name through the owning package's source bank."""

    if filename in SHARED_WOOD_SOURCES:
        return resolve_asset(WOOD_SOURCE_DIRECTORY / filename)
    return resolve_asset(SOURCE_DIRECTORY / filename)


@lru_cache(maxsize=None)
def _load(filename: str) -> np.ndarray:
    """One bundled recording as mono float in [-1, 1], trimmed to its onset.

    The lead-in matters more than it looks.  These are library recordings and
    several open with a few milliseconds of room tone; layered at a precise
    onset that silence becomes a timing error, and an impact whose layers are a
    few milliseconds late reads as mushy rather than as violent.
    """

    path = source_path(filename)
    if not path.is_file():
        raise ValueError(
            f"missing bundled source {path}; see sources/provenance.json"
        )
    return load_wav(path)


def _family(prefix: str) -> tuple[str, ...]:
    return tuple(n for n in bundled_sources() if n.startswith(prefix))


def _pick(rng: random.Random, prefixes: Sequence[str]) -> str:
    names: list[str] = []
    for prefix in prefixes:
        names.extend(_family(prefix))
    if not names:
        raise ValueError(f"no bundled source matching any of {list(prefixes)}")
    return rng.choice(sorted(names))


def _weight_speed(projectile: ProjectileProfile) -> float:
    """Playback rate for a projectile: heavier is slower, lower and longer."""

    return 1.0 / (0.62 + 0.38 * min(projectile.weight, 4.0))


def _sub(rng: random.Random, freq: float, seconds: float,
         decay: float) -> np.ndarray:
    """The low end, and the only thing in here that is synthesised.

    Noise through a resonance.  A sine here would be the exact mistake this
    generator exists to undo -- it is what made the old flesh impact land on a
    note and the old metal impact land on a bell.
    """

    return resonant_sub(rng, freq, seconds, decay)


def _energy(projectile: ProjectileProfile, intensity: float) -> float:
    """How much event there is, from the projectile and the caller together."""

    return (0.45 + 0.55 * min(projectile.weight, 4.0) / 4.0) * (
        0.55 + 0.5 * intensity)


# --------------------------------------------------------------------------
# public material layers
# --------------------------------------------------------------------------

def material_contact(material: MaterialProfile, rng: random.Random,
                     weight: float = 0.5, intensity: float = 0.85
                     ) -> tuple[np.ndarray, list[str]]:
    """One recorded contact, without projectile or debris composition.

    Event generators need the material library below :func:`impact`: a firearm
    bolt is a small metal contact and a casing is smaller still, but neither is
    a projectile impact with a sub-pressure layer and a debris shower.  Exposing
    this layer preserves the provider boundary: material generators supply
    contacts, while this event generator owns composition. ``weight`` is
    varispeed, just as projectile mass is in ``impact``.
    """

    intensity = _check_unit(intensity, "intensity")
    if not 0.05 <= weight <= 4.0:
        raise ValueError("weight must be between 0.05 and 4")
    name = _pick(rng, material.contact)
    speed = (1.0 / (0.62 + 0.38 * weight)) * material.speed
    contact = _varispeed(_load(name), speed)
    contact = _darken(contact, material.darken)
    contact = _decay(contact, 0.055 + 0.12 * min(weight, 1.5), hold=0.004)
    peak = float(np.max(np.abs(contact))) if contact.size else 0.0
    if peak > 1e-9:
        contact = contact / peak
    return contact * intensity, [name]


# --------------------------------------------------------------------------
# event renderers
# --------------------------------------------------------------------------

def impact(projectile: ProjectileProfile, material: MaterialProfile,
           rng: random.Random, intensity: float = 0.85
           ) -> tuple[np.ndarray, list[str]]:
    """Render a projectile striking a material.

    Three layers, in the order the ear needs them: the contact, whatever the
    material sheds because of it, and the weight underneath.  The contact leads
    always.  Energy projectiles get a brighter, faster contact and no debris
    worth the name -- a laser does not throw splinters, it flashes a hole.
    """

    intensity = _check_unit(intensity, "intensity")
    energy = _energy(projectile, intensity)
    speed = _weight_speed(projectile) * material.speed
    coherent = projectile.kind == "energy"
    seconds = 0.30 + 0.55 * energy + (0.0 if coherent else 0.25 * material.debris_spread)
    out = np.zeros(int(seconds * SR))
    used: list[str] = []

    # Contact.
    name = _pick(rng, material.contact)
    used.append(name)
    contact = _varispeed(_load(name), speed * (1.25 if coherent else 1.0))
    contact = _darken(contact, material.darken * (0.7 + 0.5 * projectile.brightness))
    _place(out, 0.0, _decay(contact, 0.10 + 0.16 * energy), 1.15 + 0.3 * energy)

    # The wet layer, where a material has one: water closing, flesh being flesh.
    if material.wet:
        name = _pick(rng, material.wet)
        used.append(name)
        wet = _varispeed(_load(name), speed * rng.uniform(0.9, 1.15))
        _place(out, 0.012, _decay(_darken(wet, material.darken), 0.18),
               0.55 * energy)

    # Debris. Not for energy projectiles, and not for flesh.
    if material.debris and not coherent:
        count = max(1, round(material.debris_count * (0.4 + 0.75 * energy)))
        for index in range(count):
            name = _pick(rng, material.debris)
            used.append(name)
            piece = _varispeed(_load(name),
                               speed * rng.uniform(1.0, 1.9))
            start = 0.02 + rng.random() ** 0.7 * (seconds - 0.1) * 0.7
            _place(out, start, _decay(piece, rng.uniform(0.05, 0.16)),
                   rng.uniform(0.10, 0.30) * energy
                   * (1.0 - 0.4 * start / seconds))

    # The weight. Recordings of small objects have none of it.
    _place(out, 0.0,
           _sub(rng, material.sub_frequency / max(0.75, speed),
                min(seconds, 0.45), 0.04 + 0.05 * energy),
           material.sub_gain * energy * (0.35 if coherent else 1.0))
    return out, used


def debris(material: MaterialProfile, rng: random.Random,
           intensity: float = 0.85) -> tuple[np.ndarray, list[str]]:
    """Render a decelerating shower of material fragments.

    Unlike the debris inside :func:`impact` this is the whole event, so it
    decelerates twice: pieces land, then settle.  Flesh has no debris field and
    says so rather than approximating one.
    """

    intensity = _check_unit(intensity, "intensity")
    if not material.debris:
        raise ValueError(
            f"{material.name} throws no debris; there is nothing to render"
        )
    seconds = 1.1 + 0.7 * intensity
    out = np.zeros(int(seconds * SR))
    used: list[str] = []

    count = max(2, round(material.debris_count * 2.0 * (0.5 + 0.8 * intensity)))
    for index in range(count):
        name = _pick(rng, material.debris)
        used.append(name)
        piece = _varispeed(_load(name), rng.uniform(0.9, 2.0) * material.speed)
        piece = _darken(piece, material.darken)
        start = rng.random() ** 0.65 * seconds * 0.8
        _place(out, start, _decay(piece, rng.uniform(0.05, 0.2)),
               rng.uniform(0.12, 0.38) * (1.0 - 0.5 * start / seconds)
               * (0.6 + 0.5 * intensity))
    return out, used


def _check_unit(value: float, name: str) -> float:
    if not 0.05 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.05 and 1")
    return value


# --------------------------------------------------------------------------
# lookups and io
# --------------------------------------------------------------------------

def material_named(name: str) -> MaterialProfile:
    try:
        return MATERIALS[name]
    except KeyError:
        raise ValueError(
            f"unknown material {name!r}; expected one of {sorted(MATERIALS)}"
        ) from None


__all__ = [
    "MATERIALS",
    "PROJECTILES",
    "SR",
    "MaterialProfile",
    "ProjectileProfile",
    "SHARED_WOOD_SOURCES",
    "SOURCE_DIRECTORY",
    "WOOD_SOURCE_DIRECTORY",
    "bundled_sources",
    "debris",
    "impact",
    "material_contact",
    "master",
    "material_named",
    "projectile_named",
    "source_path",
    "write_wav",
]
