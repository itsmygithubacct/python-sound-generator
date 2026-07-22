"""Recording-driven collisions, touchdowns, debris fields, and crashes.

This is an event composer.  It does not define what wood or stone sounds like:
those layers come from :mod:`wood_noise_generator` and
:mod:`stone_noise_generator`.  Metal and loose-surface layers come from the
auditioned material bank in :mod:`projectile_sounds`.  The job here is order,
scale, and consequence: contact leads, failure follows, debris outlives both,
and low weight stays underneath the recordings.

No recordings are copied into this package.  Every result returns the exact
repository-relative source paths it consumed so the CLI can preserve the
licensing chain in its manifest.
"""

from __future__ import annotations

from ..._audio.assets import asset_paths, resolve_asset

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..._audio.foley_numpy import (
    SR,
    darken,
    decay,
    load_wav,
    master,
    place,
    resonant_sub,
    varispeed,
    write_wav,
)
from ..projectile_sounds import engine as projectile
from ..stone_noise_generator import engine as stone
from ..wood_noise_generator import engine as wood


ROOT = Path(__file__).resolve().parent.parent
AIRCRAFT_SOURCES = ROOT / "aircraft_spacecraft_generator" / "sources"

# A road-vehicle collision is body-to-body contact.  Its surface still damps
# the body response, but is not itself struck like a rock.  Keeping this set
# explicit also leaves aircraft, landers, and the audition-approved mixed
# structural crash on the general material-composition path below.
ROAD_VEHICLE_STRUCTURES = frozenset({
    "kart", "compact", "sedan", "muscle", "buggy", "truck", "monster",
})


@dataclass(frozen=True)
class Component:
    """One material in a structure, with its material-specific scale."""

    material: str
    variant: str
    scale: str
    gain: float = 1.0


@dataclass(frozen=True)
class StructureProfile:
    """What fails in an event; composition, not a parallel material model."""

    name: str
    label: str
    description: str
    mass: float
    components: tuple[Component, ...]
    contact: str = "body"
    tear: float = 0.0


@dataclass(frozen=True)
class SurfaceProfile:
    """What receives the event and how much of its tail survives."""

    name: str
    label: str
    description: str
    material: str
    variant: str
    scale: str
    damping: float


def _metal(name: str, label: str, description: str, mass: float, *,
           contact: str = "body", tear: float = 0.35) -> StructureProfile:
    return StructureProfile(
        name, label, description, mass,
        (Component("metal", "metal", "body"),),
        contact=contact, tear=tear,
    )


# Vehicle and aircraft names intentionally match the registries they replace.
# Their old packages can therefore delegate without translating acoustic data.
STRUCTURES: dict[str, StructureProfile] = {
    "kart": _metal("kart", "Kart", "light exposed-frame kart", 0.30,
                   tear=0.15),
    "compact": _metal("compact", "Compact", "small hatchback body", 0.75),
    "sedan": _metal("sedan", "Sedan", "midsize road-car body", 1.00),
    "muscle": _metal("muscle", "Muscle car", "heavy coupe body", 1.30),
    "buggy": _metal("buggy", "Buggy", "tube-frame off-roader", 0.85,
                    tear=0.20),
    "truck": _metal("truck", "Truck", "body-on-frame pickup", 2.10),
    "monster": _metal("monster", "Monster truck", "oversized off-road truck",
                      3.20),
    "trainer": _metal("trainer", "Trainer", "thin-skinned light aircraft",
                      0.60, contact="tire", tear=0.80),
    "warbird": _metal("warbird", "Warbird", "stressed-skin piston fighter",
                      1.40, contact="tire", tear=0.90),
    "airliner": _metal("airliner", "Airliner", "large transport airframe",
                       3.20, contact="tire", tear=1.00),
    "fighter": _metal("fighter", "Fighter", "tight fast-jet airframe", 1.60,
                      contact="tire", tear=0.85),
    "lander": _metal("lander", "Lander", "footpad spacecraft", 0.90,
                     contact="pad", tear=0.65),
    "freighter": _metal("freighter", "Freighter", "heavy spacecraft hull",
                        4.00, contact="pad", tear=1.00),
    "timber": StructureProfile(
        "timber", "Timber frame", "beams and planks failing together", 1.40,
        (Component("wood", "beam", "body", 1.0),),
    ),
    "masonry": StructureProfile(
        "masonry", "Masonry", "concrete or dressed-stone structure", 2.80,
        (Component("stone", "concrete", "slab", 1.0),),
    ),
    "mixed": StructureProfile(
        "mixed", "Mixed structure", "masonry, timber, and metal in one event",
        3.20,
        (
            Component("stone", "concrete", "slab", 1.0),
            Component("wood", "beam", "body", 0.65),
            Component("metal", "metal", "body", 0.45),
        ),
        tear=0.35,
    ),
}


SURFACES: dict[str, SurfaceProfile] = {
    "asphalt": SurfaceProfile(
        "asphalt", "Asphalt", "sealed aggregate road", "stone", "concrete",
        "boulder", 0.15,
    ),
    "concrete": SurfaceProfile(
        "concrete", "Concrete", "hard concrete apron", "stone", "concrete",
        "boulder", 0.08,
    ),
    "wet": SurfaceProfile(
        "wet", "Wet road", "hard road under standing water", "water", "water",
        "body", 0.28,
    ),
    "dirt": SurfaceProfile(
        "dirt", "Dirt", "packed soil and loose topsoil", "dirt", "dirt",
        "body", 0.48,
    ),
    "gravel": SurfaceProfile(
        "gravel", "Gravel", "loose stone chippings", "stone", "granite",
        "cobble", 0.30,
    ),
    "sand": SurfaceProfile(
        "sand", "Sand", "deep dry sand", "dirt", "dirt", "body", 0.65,
    ),
    "mud": SurfaceProfile(
        "mud", "Mud", "wet churned soil", "dirt", "dirt", "body", 0.82,
    ),
    "grass": SurfaceProfile(
        "grass", "Grass", "turf over soil", "dirt", "dirt", "body", 0.68,
    ),
    "snow": SurfaceProfile(
        "snow", "Snow", "packed snow over ground", "dirt", "dirt", "body",
        0.88,
    ),
    "stone": SurfaceProfile(
        "stone", "Stone", "bare rock or masonry", "stone", "limestone",
        "boulder", 0.12,
    ),
    "wood": SurfaceProfile(
        "wood", "Wood", "timber deck or floor", "wood", "plank", "body",
        0.25,
    ),
    "water": SurfaceProfile(
        "water", "Water", "open water", "water", "water", "body", 0.55,
    ),
}


def _check_unit(value: float, name: str) -> float:
    if not 0.05 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.05 and 1")
    return value


def _python_rng(rng: np.random.Generator) -> random.Random:
    return random.Random(int(rng.integers(0, 2**63 - 1)))


def rng_from_random(rng: random.Random) -> np.random.Generator:
    """Bridge legacy registries without making them import NumPy themselves."""

    return np.random.default_rng(rng.getrandbits(63))


def _prefix(package: str, names: Iterable[str]) -> list[str]:
    return [f"{package}/sources/{name}" for name in names]


def _merge_sources(*groups: Iterable[str]) -> list[str]:
    result: list[str] = []
    for group in groups:
        for name in group:
            if name not in result:
                result.append(name)
    return result


def _place_layer(out: np.ndarray, signal: np.ndarray, start: float,
                 gain: float) -> None:
    place(out, start, np.asarray(signal, dtype=np.float64), gain)


def _material_contact(component: Component, mass: float,
                      rng: np.random.Generator,
                      intensity: float) -> tuple[np.ndarray, list[str]]:
    if component.material == "metal":
        signal, used = projectile.material_contact(
            projectile.MATERIALS["metal"], _python_rng(rng),
            weight=min(4.0, max(0.05, mass)), intensity=intensity,
        )
        return signal, _prefix("projectile_sounds", used)
    if component.material == "wood":
        signal, used = wood.strike(
            wood.FORMS[component.variant], rng, intensity
        )
        return signal, _prefix("wood_noise_generator", used)
    if component.material == "stone":
        rock = stone.STONES[component.variant]
        scale = stone.SCALES[component.scale]
        signal, used = stone.contact(rock, scale, rng, intensity)
        weight = stone.sub(
            scale, rng, min(0.55, max(0.15, len(signal) / SR)),
            decay=0.05 + 0.05 * scale.weight,
        )
        out = np.zeros(max(len(signal), len(weight)))
        _place_layer(out, signal, 0.0, 1.0)
        _place_layer(out, weight, 0.003, 0.20 + 0.12 * intensity)
        return out, _prefix("stone_noise_generator", used)
    raise ValueError(f"unsupported component material {component.material!r}")


def _material_fracture(component: Component, mass: float,
                       rng: np.random.Generator,
                       intensity: float) -> tuple[np.ndarray, list[str]]:
    if component.material == "metal":
        names = asset_paths(AIRCRAFT_SOURCES, "rubberduck_metal_sheet_*.wav")
        if not names:
            return np.zeros(0), []
        source = names[int(rng.integers(0, len(names)))]
        speed = 1.0 / (0.65 + 0.35 * min(mass, 4.0))
        signal = varispeed(load_wav(source), speed * 0.72)
        signal = decay(signal, 0.38 + 0.55 * intensity, hold=0.12)
        return signal, [
            f"aircraft_spacecraft_generator/sources/{source.name}"
        ]
    if component.material == "wood":
        signal, used = wood.fracture(
            wood.FORMS[component.variant], rng, intensity
        )
        return signal, _prefix("wood_noise_generator", used)
    if component.material == "stone":
        signal, used = stone.fracture(
            stone.STONES[component.variant], stone.SCALES[component.scale],
            rng, intensity, completeness=1.0,
        )
        return signal, _prefix("stone_noise_generator", used)
    raise ValueError(f"unsupported component material {component.material!r}")


def _material_debris(component: Component, rng: np.random.Generator,
                     intensity: float) -> tuple[np.ndarray, list[str]]:
    if component.material == "metal":
        signal, used = projectile.debris(
            projectile.MATERIALS["metal"], _python_rng(rng), intensity
        )
        return signal, _prefix("projectile_sounds", used)
    if component.material == "wood":
        signal, used = wood.debris(
            wood.FORMS[component.variant], rng, intensity
        )
        return signal, _prefix("wood_noise_generator", used)
    if component.material == "stone":
        seconds = 1.0 + 0.9 * intensity
        signal, used = stone.debris(
            stone.STONES[component.variant], stone.SCALES[component.scale],
            rng, seconds=seconds, intensity=intensity,
        )
        return signal, _prefix("stone_noise_generator", used)
    raise ValueError(f"unsupported component material {component.material!r}")


def _surface_component(surface: SurfaceProfile) -> Component:
    return Component(surface.material, surface.variant, surface.scale)


def _surface_contact(surface: SurfaceProfile, rng: np.random.Generator,
                     intensity: float) -> tuple[np.ndarray, list[str]]:
    if surface.material in ("stone", "wood"):
        return _material_contact(
            _surface_component(surface), 0.8, rng, intensity
        )
    material = projectile.MATERIALS[surface.material]
    signal, used = projectile.material_contact(
        material, _python_rng(rng), weight=0.8, intensity=intensity
    )
    if surface.damping > 0.6:
        signal = darken(signal, 2_400.0 - 1_200.0 * surface.damping)
    return signal, _prefix("projectile_sounds", used)


def _vehicle_collision(structure: StructureProfile, surface: SurfaceProfile,
                       rng: np.random.Generator,
                       intensity: float) -> tuple[np.ndarray, list[str]]:
    """Recorded body slam and panel deformation for road vehicles.

    The previous general path treated asphalt as a stone object hit by the
    car.  That layer is useful for structures landing on concrete, but made a
    standalone traffic impact read as metal plus loose rocks.  Here the road
    affects damping only; the audible event remains two vehicle-body layers
    with noise-derived low weight underneath.
    """

    seconds = 0.85 + 0.32 * min(structure.mass, 4.0) + 0.35 * intensity
    out = np.zeros(round(seconds * SR))

    slam_path = resolve_asset(AIRCRAFT_SOURCES / "rubberduck_metal_slam_01.wav")
    mass = min(structure.mass, 3.2)
    speed = 1.0 / (0.72 + 0.28 * mass)
    slam = varispeed(load_wav(slam_path), speed)
    slam = darken(slam, 14_000.0 - 6_000.0 * surface.damping)
    slam = decay(
        slam,
        (0.34 + 0.28 * intensity) * (1.0 - 0.35 * surface.damping),
        hold=0.045,
    )
    _place_layer(out, slam, 0.0, 0.96 + 0.26 * intensity)

    sheet_names = asset_paths(AIRCRAFT_SOURCES, "rubberduck_metal_sheet_*.wav")
    sources = [
        "aircraft_spacecraft_generator/sources/rubberduck_metal_slam_01.wav"
    ]
    if sheet_names:
        sheet_path = sheet_names[int(rng.integers(0, len(sheet_names)))]
        sheet_speed = 0.92 / (0.82 + 0.18 * mass)
        sheet = varispeed(load_wav(sheet_path), sheet_speed)
        sheet = darken(sheet, 11_500.0 - 4_500.0 * surface.damping)
        sheet = decay(sheet, 0.18 + 0.24 * intensity, hold=0.025)
        _place_layer(out, sheet, 0.009, 0.25 + 0.25 * intensity)
        sources.append(
            f"aircraft_spacecraft_generator/sources/{sheet_path.name}"
        )

    sub = resonant_sub(
        _python_rng(rng),
        62.0 / max(0.72, mass ** 0.34),
        min(seconds, 0.52),
        0.07 + 0.04 * mass,
        q=3.0,
        cutoff=280.0,
    )
    _place_layer(
        out, sub, 0.002,
        (0.55 + 0.35 * intensity) * min(1.7, 0.75 + 0.45 * mass),
    )
    return out, sources


def collision(structure: StructureProfile, surface: SurfaceProfile,
              rng: np.random.Generator,
              intensity: float = 0.85) -> tuple[np.ndarray, list[str]]:
    """A structure taking one hit: contacts first, failure just behind."""

    intensity = _check_unit(intensity, "intensity")
    if structure.name in ROAD_VEHICLE_STRUCTURES:
        return _vehicle_collision(structure, surface, rng, intensity)

    seconds = 0.85 + 0.32 * min(structure.mass, 4.0) + 0.35 * intensity
    out = np.zeros(round(seconds * SR))
    sources: list[str] = []

    for index, component in enumerate(structure.components):
        signal, used = _material_contact(
            component, structure.mass, rng, intensity
        )
        _place_layer(out, signal, 0.004 * index, component.gain)
        sources = _merge_sources(sources, used)

        if intensity >= 0.48:
            failure, used = _material_fracture(
                component, structure.mass, rng, intensity
            )
            _place_layer(
                out, failure, 0.018 + 0.012 * index,
                component.gain * (0.22 + 0.58 * intensity)
                * (structure.tear if component.material == "metal" else 1.0),
            )
            sources = _merge_sources(sources, used)

    ground, used = _surface_contact(surface, rng, max(0.05, intensity * 0.65))
    _place_layer(out, ground, 0.012, 0.34 * (1.0 - 0.55 * surface.damping))
    sources = _merge_sources(sources, used)
    return out, sources


def touchdown(structure: StructureProfile, surface: SurfaceProfile,
              rng: np.random.Generator, speed: float = 0.5,
              intensity: float = 0.85) -> tuple[np.ndarray, list[str]]:
    """A controlled arrival: contact, gear load, ground, and settling."""

    speed = _check_unit(speed, "speed")
    intensity = _check_unit(intensity, "intensity")
    seconds = 1.2 + 0.48 * min(structure.mass, 3.0)
    out = np.zeros(round(seconds * SR))
    sources: list[str] = []

    if structure.contact == "tire":
        tire_path = resolve_asset(AIRCRAFT_SOURCES / "egomassive_tire.wav")
        tire = varispeed(load_wav(tire_path), 0.9 + 0.4 * speed)
        tire = decay(tire, 0.13 + 0.10 * speed, hold=0.02)
        _place_layer(out, tire, 0.0, 0.55 + 0.40 * speed)
        sources.append(
            "aircraft_spacecraft_generator/sources/egomassive_tire.wav"
        )

    component = structure.components[0]
    gear, used = _material_contact(
        component, structure.mass * (1.45 if structure.contact == "pad" else 0.7),
        rng, intensity,
    )
    _place_layer(out, gear, 0.002, 0.90 + 0.30 * intensity)
    sources = _merge_sources(sources, used)

    ground, used = _surface_contact(surface, rng, intensity)
    ground_gain = 0.34 if structure.contact == "tire" else 0.62
    _place_layer(out, ground, 0.006, ground_gain)
    sources = _merge_sources(sources, used)

    py_rng = _python_rng(rng)
    sub = resonant_sub(
        py_rng,
        72.0 / max(0.65, structure.mass ** 0.35),
        min(seconds, 0.62),
        0.055 + 0.055 * min(structure.mass, 3.0),
    )
    _place_layer(
        out, sub, 0.004,
        (0.38 + 0.26 * intensity) * min(1.5, structure.mass)
        * (1.35 if structure.contact == "pad" else 1.0),
    )
    return out, sources


def debris(structure: StructureProfile, surface: SurfaceProfile,
           rng: np.random.Generator,
           intensity: float = 0.85) -> tuple[np.ndarray, list[str]]:
    """Fragments from every component, then the surface they settle on."""

    intensity = _check_unit(intensity, "intensity")
    seconds = 1.65 + 0.65 * intensity
    out = np.zeros(round(seconds * SR))
    sources: list[str] = []
    for index, component in enumerate(structure.components):
        field, used = _material_debris(component, rng, intensity)
        _place_layer(out, field, 0.035 * index, component.gain)
        sources = _merge_sources(sources, used)

    # Loose and brittle ground contributes to the aftermath; hard road does not
    # receive a second invented debris family.
    if surface.material in ("dirt", "stone") and surface.damping < 0.75:
        component = _surface_component(surface)
        if component.material == "dirt":
            field, used = projectile.debris(
                projectile.MATERIALS["dirt"], _python_rng(rng),
                max(0.05, intensity * 0.7)
            )
            used = _prefix("projectile_sounds", used)
        else:
            field, used = _material_debris(
                component, rng, max(0.05, intensity * 0.6)
            )
        _place_layer(out, field, 0.05, 0.38 * (1.0 - surface.damping))
        sources = _merge_sources(sources, used)
    return out, sources


def crash(structure: StructureProfile, surface: SurfaceProfile,
          rng: np.random.Generator,
          intensity: float = 0.85) -> tuple[np.ndarray, list[str]]:
    """A full crash: primary hit, decelerating secondaries, then debris."""

    intensity = _check_unit(intensity, "intensity")
    seconds = 2.85 + 0.85 * intensity
    out = np.zeros(round(seconds * SR))
    sources: list[str] = []

    first, used = collision(structure, surface, rng, intensity)
    _place_layer(out, first, 0.0, 1.0)
    sources = _merge_sources(sources, used)

    onset = 0.0
    for index in range(1 + round(2 * intensity)):
        onset += float(rng.uniform(0.18, 0.43)) + 0.05 * index
        level = max(0.05, intensity * (0.62 ** (index + 1)))
        secondary, used = collision(structure, surface, rng, level)
        _place_layer(out, secondary, onset, float(rng.uniform(0.50, 0.78)))
        sources = _merge_sources(sources, used)

    aftermath, used = debris(structure, surface, rng, intensity)
    _place_layer(out, aftermath, onset + 0.08, 0.88)
    sources = _merge_sources(sources, used)
    return out, sources


def structure_named(name: str) -> StructureProfile:
    try:
        return STRUCTURES[name]
    except KeyError:
        raise ValueError(
            f"unknown structure {name!r}; expected one of {sorted(STRUCTURES)}"
        ) from None


def surface_named(name: str) -> SurfaceProfile:
    try:
        return SURFACES[name]
    except KeyError:
        raise ValueError(
            f"unknown surface {name!r}; expected one of {sorted(SURFACES)}"
        ) from None


__all__ = [
    "AIRCRAFT_SOURCES",
    "ROOT",
    "SR",
    "STRUCTURES",
    "SURFACES",
    "Component",
    "StructureProfile",
    "SurfaceProfile",
    "collision",
    "crash",
    "debris",
    "master",
    "rng_from_random",
    "structure_named",
    "surface_named",
    "touchdown",
    "write_wav",
]
