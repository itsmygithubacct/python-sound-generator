"""Cue and pack registries for the vehicle-foley generator.

One thing here is not shared with the other generators in this repository: a
cue can be a **loop**.  Continuous foley — tire roll, a held drift, spray,
nitrous — is bed material a game starts and stops, not a one-shot it triggers,
and the difference reaches all the way down to mastering.  See :func:`render`.
"""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from ..._audio.models import VariationFilenames

from ..._audio.models import RenderResult as Render

import random
from dataclasses import dataclass
from typing import Callable

from ..._audio import dsp
from ..._audio import loops as loopdsp

from . import engine


@dataclass(frozen=True)
class Options:
    """Every supported render axis; cues ignore fields they do not honour."""

    vehicle: engine.VehicleProfile = engine.VEHICLES["sedan"]
    surface: engine.SurfaceProfile = engine.SURFACES["asphalt"]
    speed: float = 0.6
    duration: float = 2.5
    intensity: float = 0.85


@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Callable[[random.Random, Options], list[float]]
    peak_db: float
    honours: Honours | str | tuple[str, ...] = ""
    loop: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )




CUES: dict[str, Cue] = {
    "roll": Cue(
        "roll", "tires", "seamless loop of tires rolling over a surface",
        lambda rng, opts: engine.tire_roll(
            opts.vehicle, opts.surface, rng, opts.speed, opts.duration,
            opts.intensity,
        ),
        -8.0, "vehicle, surface, speed, duration, intensity", loop=True,
    ),
    "skid": Cue(
        "skid", "tires", "one-shot skid: grab, squeal, and release",
        lambda rng, opts: engine.tire_skid(
            opts.vehicle, opts.surface, rng, opts.speed, opts.intensity
        ),
        -3.0, "vehicle, surface, speed, intensity",
    ),
    "drift": Cue(
        "drift", "tires", "seamless loop of a sustained held slide",
        lambda rng, opts: engine.tire_drift(
            opts.vehicle, opts.surface, rng, opts.speed, opts.duration,
            opts.intensity,
        ),
        -6.0, "vehicle, surface, speed, duration, intensity", loop=True,
    ),
    "spray": Cue(
        "spray", "tires", "seamless loop of mud, gravel, or grit in the arches",
        lambda rng, opts: engine.surface_spray(
            opts.vehicle, opts.surface, rng, opts.speed, opts.duration,
            opts.intensity,
        ),
        -9.0, "vehicle, surface, speed, duration, intensity", loop=True,
    ),
    "suspension": Cue(
        "suspension", "chassis", "one suspension compression and rebound",
        lambda rng, opts: engine.suspension_bump(
            opts.vehicle, rng, opts.intensity
        ),
        -4.0, "vehicle, intensity",
    ),
    "gear": Cue(
        "gear", "chassis", "driveline lash closing, then the shove",
        lambda rng, opts: engine.gear_thump(opts.vehicle, rng, opts.intensity),
        -4.0, "vehicle, intensity",
    ),
    "nitro": Cue(
        "nitro", "chassis", "seamless loop of nitrous under pressure",
        lambda rng, opts: engine.nitro_loop(
            opts.vehicle, rng, opts.duration, opts.intensity
        ),
        -5.0, "vehicle, duration, intensity", loop=True,
    ),
    "boost": Cue(
        "boost", "chassis", "one-shot boost attack: valve, swell, and roar",
        lambda rng, opts: engine.boost_burst(
            opts.vehicle, rng, opts.intensity
        ),
        -3.0, "vehicle, intensity",
    ),
    "takeoff": Cue(
        "takeoff", "air", "leaving a ramp: chirp, unload, and air",
        lambda rng, opts: engine.ramp_takeoff(
            opts.vehicle, opts.surface, rng, opts.speed, opts.intensity
        ),
        -3.0, "vehicle, surface, speed, intensity",
    ),
    "landing": Cue(
        "landing", "air", "touchdown: contact, suspension travel, and boom",
        lambda rng, opts: engine.ramp_landing(
            opts.vehicle, opts.surface, rng, opts.speed, opts.intensity
        ),
        -1.5, "vehicle, surface, speed, intensity",
    ),
    "collision": Cue(
        "collision", "impacts", "a body panel taking a hit",
        lambda rng, opts: engine.body_collision(
            opts.vehicle, rng, opts.intensity
        ),
        -1.0, "vehicle, intensity",
    ),
    "crash": Cue(
        "crash", "impacts", "full crash: impact, secondaries, and aftermath",
        lambda rng, opts: engine.vehicle_crash(
            opts.vehicle, opts.surface, rng, opts.intensity
        ),
        -0.5, "vehicle, surface, intensity",
    ),
    "debris": Cue(
        "debris", "impacts", "parts leaving the vehicle and finding the ground",
        lambda rng, opts: engine.crash_debris(
            opts.vehicle, opts.surface, rng, opts.intensity
        ),
        -5.0, "vehicle, surface, intensity",
    ),
}


GROUPS = ("tires", "chassis", "air", "impacts")

LOOPS = tuple(name for name, cue in CUES.items() if cue.loop)


def render(cue_name: str, options: Options, rng: random.Random) -> list[float]:
    """Render and master a named cue.

    One-shots take the shared chain from ``combat_generators``, which fades both
    edges to keep a trigger from clicking.  A loop must not: its last sample is
    built to flow into its first, and fading them to zero would punch a hole in
    the loop point every time it came round.  ``loopdsp.master`` is the same
    chain without the fades — limiting and normalising are memoryless, so a
    periodic signal stays periodic through both.
    """

    return render_with_sources(cue_name, options, rng).samples


def render_with_sources(cue_name: str, options: Options,
                        rng: random.Random) -> Render:
    """Render while retaining source paths from the central crash composer."""

    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    sources: tuple[str, ...] = ()
    if cue_name in ("collision", "crash", "debris"):
        try:
            from ..crash_generator import engine as crashes
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ValueError(
                "collision, crash, and debris are owned by crash_generator; "
                "install its dependencies with `pip install -r "
                "crash_generator/requirements.txt`"
            ) from exc

        structure = crashes.STRUCTURES[options.vehicle.name]
        surface_name = (
            "asphalt" if cue_name == "collision" else options.surface.name
        )
        surface = crashes.SURFACES[surface_name]
        crash_rng = crashes.rng_from_random(rng)
        if cue_name == "collision":
            signal, used = crashes.collision(
                structure, surface, crash_rng, options.intensity
            )
        elif cue_name == "crash":
            signal, used = crashes.crash(
                structure, surface, crash_rng, options.intensity
            )
        else:
            signal, used = crashes.debris(
                structure, surface, crash_rng, options.intensity
            )
        sources = tuple(used)
    else:
        signal = cue.render(rng, options)
    if cue.loop:
        samples = loopdsp.master(signal, cue.peak_db)
    else:
        samples = dsp.master(signal, cue.peak_db)
    return Render(samples, sources)


def stem_for(cue_name: str, options: Options) -> str:
    """Build a descriptive default artifact name from honoured options."""

    vehicle = options.vehicle.name
    surface = options.surface.name
    if cue_name in ("roll", "skid", "drift", "landing", "crash", "debris"):
        return f"{cue_name}_{vehicle}_{surface}"
    if cue_name == "spray":
        return f"spray_{surface}"
    if cue_name in ("suspension", "gear", "nitro", "boost", "takeoff"):
        return f"{cue_name}_{vehicle}"
    return cue_name


@dataclass(frozen=True)
class PackEntry:
    filename: str
    cue: str
    count: int
    options: Options


@dataclass(frozen=True)
class Pack(VariationFilenames):
    name: str
    description: str
    entries: tuple[PackEntry, ...]



DEFAULTS = Options()

# Packs render shorter loops than an interactive `render` defaults to. A bank is
# dozens of files of pure-Python DSP, and 2 s of tire roll loops exactly as well
# as 2.5 s does.
PACK_LOOP = 2.0


def _at(surface: str = "asphalt", vehicle: str = "sedan", **kwargs) -> Options:
    """Options for a pack entry, named rather than positional."""

    return Options(
        vehicle=engine.VEHICLES[vehicle],
        surface=engine.SURFACES[surface],
        duration=kwargs.pop("duration", PACK_LOOP),
        **kwargs,
    )


SHOWCASE = Pack(
    "showcase",
    "one representative render of every cue",
    (
        PackEntry("roll_sedan_asphalt", "roll", 1, _at()),
        PackEntry("skid_sedan_asphalt", "skid", 1, _at(speed=0.75)),
        PackEntry("drift_muscle_asphalt", "drift", 1,
                  _at(vehicle="muscle", speed=0.75)),
        PackEntry("spray_gravel", "spray", 1, _at("gravel", speed=0.7)),
        PackEntry("suspension_sedan", "suspension", 1, _at()),
        PackEntry("gear_muscle", "gear", 1, _at(vehicle="muscle")),
        PackEntry("nitro_muscle", "nitro", 1, _at(vehicle="muscle")),
        PackEntry("boost_muscle", "boost", 1, _at(vehicle="muscle")),
        PackEntry("takeoff_buggy", "takeoff", 1,
                  _at("dirt", "buggy", speed=0.85)),
        PackEntry("landing_buggy_dirt", "landing", 1,
                  _at("dirt", "buggy", speed=0.85)),
        PackEntry("collision_sedan", "collision", 1, _at()),
        PackEntry("crash_sedan_asphalt", "crash", 1, _at()),
        PackEntry("debris_sedan_asphalt", "debris", 1, _at()),
    ),
)


SURFACE_MATRIX = Pack(
    "surfaces",
    "roll, spray, skid, and landing across every surface",
    tuple(
        PackEntry(f"roll_{name}", "roll", 2, _at(name, speed=0.6))
        for name in engine.SURFACES
    )
    + tuple(
        PackEntry(f"spray_{name}", "spray", 1, _at(name, speed=0.7))
        for name in engine.SURFACES
    )
    + tuple(
        PackEntry(f"skid_{name}", "skid", 2, _at(name, speed=0.75))
        for name in engine.SURFACES
    )
    + tuple(
        PackEntry(f"landing_{name}", "landing", 2, _at(name, speed=0.8))
        for name in engine.SURFACES
    ),
)


ARCADE_DRIVER = Pack(
    "arcade_driver",
    "road-car bank for Arcade Driver; pair with car_engine_generator",
    (
        PackEntry("tire_roll_slow", "roll", 1, _at(speed=0.25)),
        PackEntry("tire_roll_cruise", "roll", 1, _at(speed=0.6)),
        PackEntry("tire_roll_fast", "roll", 1, _at(speed=1.0)),
        PackEntry("tire_roll_wet", "roll", 1, _at("wet", speed=0.6)),
        PackEntry("tire_roll_concrete", "roll", 1, _at("concrete", speed=0.6)),
        PackEntry("tire_skid", "skid", 3, _at(speed=0.8)),
        PackEntry("tire_drift", "drift", 1, _at(speed=0.8)),
        PackEntry("suspension_bump", "suspension", 3, _at(intensity=0.6)),
        PackEntry("suspension_hard", "suspension", 2, _at(intensity=1.0)),
        PackEntry("gear_thump", "gear", 3, _at()),
        PackEntry("boost", "boost", 2, _at()),
        PackEntry("nitro_loop", "nitro", 1, _at()),
        PackEntry("jump_takeoff", "takeoff", 2, _at(speed=0.9)),
        PackEntry("jump_landing", "landing", 2, _at(speed=0.9)),
        PackEntry("traffic_impact", "collision", 3, _at(intensity=0.7)),
        PackEntry("crash", "crash", 2, _at(intensity=1.0)),
        PackEntry("crash_debris", "debris", 2, _at()),
    ),
)


# Minimal runtime bank for the first Arcade Driver audio integration.  The larger
# `arcade_driver` library pack remains useful for auditioning alternates; this one
# contains exactly the stems its C mixer opens.
ARCADE_DRIVER_RUNTIME = Pack(
    "arcade-driver-runtime",
    "the seven foley stems loaded by Arcade Driver's runtime mixer",
    (
        PackEntry("tire_roll", "roll", 1, _at(speed=0.6)),
        PackEntry("nitro_loop", "nitro", 1, _at()),
        PackEntry("boost", "boost", 1, _at()),
        PackEntry("jump_takeoff", "takeoff", 1, _at(speed=0.9)),
        PackEntry("jump_landing", "landing", 1, _at(speed=0.9)),
        PackEntry("traffic_impact", "collision", 1, _at(intensity=0.7)),
        PackEntry("crash", "crash", 1, _at(intensity=1.0)),
    ),
)


FLIGHT_SIM_TIRES = Pack(
    "flight-sim-tires",
    "the one seamless tire_roll.wav Flight Simulator loads on the ground",
    (
        PackEntry("tire_roll", "roll", 1, _at("asphalt", "sedan", speed=0.6)),
    ),
)


OFFROAD = Pack(
    "offroad",
    "loose-surface bank for Offroad Racer; pair with its procedural engine",
    (
        PackEntry("tire_roll_dirt", "roll", 1, _at("dirt", "buggy", speed=0.6)),
        PackEntry("tire_roll_gravel", "roll", 1,
                  _at("gravel", "buggy", speed=0.6)),
        PackEntry("tire_roll_mud", "roll", 1, _at("mud", "buggy", speed=0.6)),
        PackEntry("tire_roll_sand", "roll", 1, _at("sand", "buggy", speed=0.6)),
        PackEntry("spray_dirt", "spray", 2, _at("dirt", "buggy", speed=0.7)),
        PackEntry("spray_gravel", "spray", 2, _at("gravel", "buggy", speed=0.7)),
        PackEntry("spray_mud", "spray", 2, _at("mud", "buggy", speed=0.7)),
        PackEntry("skid_dirt", "skid", 3, _at("dirt", "buggy", speed=0.8)),
        PackEntry("drift_dirt", "drift", 1, _at("dirt", "buggy", speed=0.8)),
        PackEntry("suspension_bump", "suspension", 3,
                  _at(vehicle="buggy", intensity=0.7)),
        PackEntry("gear_thump", "gear", 2, _at(vehicle="buggy")),
        PackEntry("nitro_loop", "nitro", 1, _at(vehicle="buggy")),
        PackEntry("boost", "boost", 2, _at(vehicle="buggy")),
        PackEntry("jump_takeoff", "takeoff", 2,
                  _at("dirt", "buggy", speed=0.9)),
        PackEntry("jump_landing_dirt", "landing", 2,
                  _at("dirt", "buggy", speed=0.9)),
        PackEntry("jump_landing_mud", "landing", 2,
                  _at("mud", "buggy", speed=0.9)),
        PackEntry("crash", "crash", 2,
                  _at("dirt", "buggy", intensity=1.0)),
        PackEntry("crash_debris", "debris", 2, _at("dirt", "buggy")),
        PackEntry("monster_roll", "roll", 1,
                  _at("dirt", "monster", speed=0.5)),
        PackEntry("monster_landing", "landing", 2,
                  _at("dirt", "monster", speed=0.8)),
    ),
)


# Offroad Racer loads a fixed set of stems out of `sfx/`, probing name.wav,
# name_v02.wav, ... until one is missing.  Its filenames say where a sound is
# used, not what it is, so the entries below are named for the game while
# `offroad` above stays the descriptive bank.
OFFROAD_RACER_RUNTIME = Pack(
    "offroad-racer-runtime",
    "drop-in replacements for Offroad Racer's baked sfx/ one-shots",
    (
        PackEntry("skid", "skid", 4, _at("dirt", "buggy", speed=0.8)),
        # Both bump sites are collisions -- a barrier wall and truck-on-truck --
        # rather than the suspension travel the name suggests.
        PackEntry("bump", "collision", 4, _at(vehicle="buggy", intensity=0.7)),
        # Fired once per boost, so this wants the one-shot attack and not the
        # `nitro` loop the filename shares a name with.
        PackEntry("nitro", "boost", 4, _at(vehicle="buggy")),
        PackEntry("jump", "takeoff", 2, _at("dirt", "buggy", speed=0.9)),
        PackEntry("land", "landing", 4, _at("dirt", "buggy", speed=0.9)),
    ),
)


PACKS = {
    pack.name: pack
    for pack in (SHOWCASE, SURFACE_MATRIX, ARCADE_DRIVER, ARCADE_DRIVER_RUNTIME,
                 FLIGHT_SIM_TIRES, OFFROAD, OFFROAD_RACER_RUNTIME)
}
