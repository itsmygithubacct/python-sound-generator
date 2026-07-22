"""Cue and pack registries for the aircraft/spacecraft generator.

Running is a verb and a powerplant is a noun, so there is one ``engine`` cue and
eight profiles for it rather than eight cues -- the same split
``weapons_destruction_generator`` makes between ``fire`` and ``--weapon``.  The
four propulsion loops the roadmap asks for are ``engine`` crossed with the four
powerplant *kinds*; ``list --powerplants`` is where they are enumerated.

Two flags on a cue decide what mastering and validation it gets.  ``loop`` picks
the fade-free master, because fading a bed's edges to zero puts a hole at the
loop point every time it comes round.  ``stationary`` says whether the bed also
holds a steady level, which is what :func:`verify` can fairly hold it to; see
:class:`Cue`.
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

def foley_available() -> bool:
    """Whether the recording-driven impact cues can render."""

    module = _load_hybrid()
    return module is not None and module.available()


def _load_hybrid():
    """Load optional recording dependencies only when an impact needs them."""

    try:
        from . import hybrid
    except ImportError:  # pragma: no cover - optional dependencies absent
        return None
    return hybrid


def _foley():
    """The recording-driven renderers, or a clear error explaining their absence.

    There is deliberately no synthesised fallback for impacts.  One was written
    and it produced a chime over a beep, because a sum of decaying sinusoids is
    a bell and a sine sweep is a beep no matter what they are labelled -- so
    falling back to it would not be a degraded aeroplane, it would be a
    different and worse sound wearing the same cue name.  Failing loudly is the
    honest option: the other nine cues need nothing but the standard library and
    still work.
    """

    hybrid = _load_hybrid()
    if hybrid is None or not hybrid.available():
        raise ValueError(
            "the touchdown and crash cues are rendered from recordings and "
            "are owned by crash_generator and need its NumPy/SciPy "
            "dependencies: "
            "pip install -r crash_generator/requirements.txt"
        )
    return hybrid


def renderer_for(cue_name: str) -> str:
    """Which path a cue takes: ``foley`` for impacts, ``procedural`` otherwise."""

    cue = CUES.get(cue_name)
    return "foley" if cue is not None and cue.foley else "procedural"


@dataclass(frozen=True)
class Options:
    """Every supported render axis; cues ignore fields they do not honour.

    ``power`` and ``speed`` are separate axes because an aeroplane can hold any
    combination of the two, and the difference between them is most of what
    flying sounds like: full power at low speed is a climb, idle at high speed
    is a descent, and only ``cockpit`` hears both at once.
    """

    craft: engine.CraftProfile = engine.CRAFT["trainer"]
    powerplant: engine.PowerplantProfile = engine.POWERPLANTS["flat_four"]
    power: float = 0.6
    speed: float = 0.5
    duration: float = 2.5
    intensity: float = 0.85


@dataclass(frozen=True)
class Cue:
    """One named sound.

    ``stationary`` only means something when ``loop`` is set, and it exists for
    exactly one reason: ``loopdsp.seam`` measures a loop point two ways, and one
    of the two does not apply to rhythmic content.  Its level check compares the
    energy closing the buffer against the energy opening it, which catches the
    swell of an unprimed filter -- but a klaxon legitimately ends in the gap
    between two beeps, so it fails a check it was never in scope for.  The step
    check still applies to everything, and it is the one that catches clicks.
    """

    name: str
    group: str
    description: str
    render: Callable[[random.Random, Options], list[float]]
    peak_db: float
    honours: Honours | str | tuple[str, ...] = ""
    loop: bool = False
    stationary: bool = True
    foley: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )




CUES: dict[str, Cue] = {
    "engine": Cue(
        "engine", "propulsion",
        "seamless loop of a powerplant running at a power setting",
        lambda rng, opts: engine.run_powerplant(
            opts.powerplant, rng, opts.power, opts.duration, opts.intensity
        ),
        -6.0, "powerplant, power, duration, intensity", loop=True,
    ),
    "prop": Cue(
        "prop", "propulsion",
        "seamless loop of propeller or fan blade-pass on its own",
        lambda rng, opts: engine.propeller(
            opts.powerplant, rng, opts.power, opts.duration, opts.intensity
        ),
        -8.0, "powerplant, power, duration, intensity", loop=True,
    ),
    "start": Cue(
        "start", "propulsion",
        "one-shot start: crank and catch, spool and light, or ignition",
        lambda rng, opts: engine.start_powerplant(
            opts.powerplant, rng, opts.power, opts.intensity
        ),
        -4.0, "powerplant, power, intensity",
    ),
    "shutdown": Cue(
        "shutdown", "propulsion",
        "one-shot wind-down: it coasts, because nothing is driving it now",
        lambda rng, opts: engine.shutdown_powerplant(
            opts.powerplant, rng, opts.power, opts.intensity
        ),
        -6.0, "powerplant, power, intensity",
    ),
    "wind": Cue(
        "wind", "airframe",
        "seamless loop of speed-dependent airflow over the airframe",
        lambda rng, opts: engine.airflow(
            opts.craft, rng, opts.speed, opts.duration, opts.intensity
        ),
        -10.0, "craft, speed, duration, intensity", loop=True,
    ),
    "vibration": Cue(
        "vibration", "airframe",
        "seamless loop of the structure being shaken by the boundary layer",
        lambda rng, opts: engine.airframe_vibration(
            opts.craft, rng, opts.speed, opts.duration, opts.intensity
        ),
        -12.0, "craft, speed, duration, intensity", loop=True,
    ),
    "cockpit": Cue(
        "cockpit", "airframe",
        "seamless loop of the interior: muffled powerplant, cavity, rattle",
        lambda rng, opts: engine.cockpit_interior(
            opts.craft, opts.powerplant, rng, opts.power, opts.speed,
            opts.duration, opts.intensity,
        ),
        -9.0, "craft, powerplant, power, speed, duration, intensity",
        loop=True,
    ),
    "stall": Cue(
        "stall", "alarms",
        "seamless loop of a warbling two-tone stall warning reed",
        lambda rng, opts: engine.stall_horn(
            opts.craft, rng, opts.duration, opts.intensity
        ),
        -3.0, "craft, duration, intensity", loop=True,
    ),
    "alarm": Cue(
        "alarm", "alarms",
        "seamless loop of a two-tone master caution klaxon",
        lambda rng, opts: engine.alarm_klaxon(
            opts.craft, rng, opts.duration, opts.intensity
        ),
        -2.0, "craft, duration, intensity", loop=True, stationary=False,
    ),
    "touchdown": Cue(
        "touchdown", "landing",
        "one-shot arrival from recordings: contact, strut, and settling",
        lambda rng, opts: _foley().touchdown(
            opts.craft, rng, opts.speed, opts.intensity
        ),
        -1.5, "craft, speed, intensity", foley=True,
    ),
    "crash": Cue(
        "crash", "landing",
        "one-shot crash from recordings: impact, tearing, tumbling, debris",
        lambda rng, opts: _foley().craft_crash(
            opts.craft, rng, opts.intensity
        ),
        -0.5, "craft, intensity", foley=True,
    ),
}


GROUPS = ("propulsion", "airframe", "alarms", "landing")

LOOPS = tuple(name for name, cue in CUES.items() if cue.loop)

ONESHOTS = tuple(name for name, cue in CUES.items() if not cue.loop)


def render(cue_name: str, options: Options, rng: random.Random) -> list[float]:
    """Render and master a named cue.

    One-shots take the shared chain from ``combat_generators``, which fades both
    edges so a trigger cannot click.  A loop must not be faded: the whole point
    of building it periodic is that its last sample flows into its first, and
    fading them to zero would punch a hole in the loop point on every repeat.
    ``loopdsp.master`` is the same chain with the fades removed.
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
    if cue.foley:
        from ..crash_generator import engine as crashes

        if not foley_available():
            _foley()
        structure = crashes.STRUCTURES[options.craft.name]
        surface = crashes.SURFACES[
            "dirt" if options.craft.gear == "pad" else "asphalt"
        ]
        crash_rng = crashes.rng_from_random(rng)
        if cue_name == "touchdown":
            signal, used = crashes.touchdown(
                structure, surface, crash_rng, options.speed,
                options.intensity,
            )
        else:
            signal, used = crashes.crash(
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

    craft = options.craft.name
    plant = options.powerplant.name
    if cue_name == "cockpit":
        return f"cockpit_{craft}_{plant}"
    if cue_name in ("engine", "prop", "start", "shutdown"):
        return f"{cue_name}_{plant}"
    return f"{cue_name}_{craft}"


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
# dozens of files of pure-Python DSP, and 2 s of turbofan loops exactly as well
# as 2.5 s does.
PACK_LOOP = 2.0


def _at(craft: str = "trainer", powerplant: str = "flat_four",
        **kwargs) -> Options:
    """Options for a pack entry, named rather than positional."""

    return Options(
        craft=engine.CRAFT[craft],
        powerplant=engine.POWERPLANTS[powerplant],
        duration=kwargs.pop("duration", PACK_LOOP),
        **kwargs,
    )


SHOWCASE = Pack(
    "showcase",
    "one representative render of every cue",
    (
        PackEntry("engine_flat_four", "engine", 1, _at(power=0.7)),
        PackEntry("prop_flat_four", "prop", 1, _at(power=0.7)),
        PackEntry("start_flat_four", "start", 1, _at(power=0.3)),
        PackEntry("shutdown_flat_four", "shutdown", 1, _at(power=0.3)),
        PackEntry("wind_trainer", "wind", 1, _at(speed=0.6)),
        PackEntry("vibration_trainer", "vibration", 1, _at(speed=0.6)),
        PackEntry("cockpit_trainer_flat_four", "cockpit", 1,
                  _at(power=0.7, speed=0.6)),
        PackEntry("stall_trainer", "stall", 1, _at()),
        PackEntry("alarm_lander", "alarm", 1, _at("lander", "rocket")),
        PackEntry("touchdown_trainer", "touchdown", 1, _at(speed=0.6)),
        PackEntry("crash_trainer", "crash", 1, _at()),
    ),
)


# The four "propulsion loops" of the roadmap are the four kinds in here.
POWERPLANT_MATRIX = Pack(
    "powerplants",
    "every powerplant running, starting, and stopping",
    tuple(
        PackEntry(f"engine_{name}", "engine", 1,
                  _at(powerplant=name, power=0.7))
        for name in engine.POWERPLANTS
    )
    + tuple(
        PackEntry(f"engine_{name}_idle", "engine", 1,
                  _at(powerplant=name, power=0.1))
        for name in engine.POWERPLANTS
    )
    + tuple(
        PackEntry(f"start_{name}", "start", 1, _at(powerplant=name, power=0.3))
        for name in engine.POWERPLANTS
    )
    + tuple(
        PackEntry(f"shutdown_{name}", "shutdown", 1,
                  _at(powerplant=name, power=0.3))
        for name in engine.POWERPLANTS
    )
    + tuple(
        PackEntry(f"prop_{name}", "prop", 1, _at(powerplant=name, power=0.7))
        for name, plant in engine.POWERPLANTS.items()
        if plant.blades > 0
    ),
)


FLIGHT_SIM = Pack(
    "flight",
    "piston trainer bank for Flight Simulator; pair tire roll with vehicle_foley",
    (
        PackEntry("engine_idle", "engine", 1, _at(power=0.1)),
        PackEntry("engine_cruise", "engine", 1, _at(power=0.6)),
        PackEntry("engine_full", "engine", 1, _at(power=1.0)),
        PackEntry("engine_start", "start", 1, _at(power=0.15)),
        PackEntry("engine_shutdown", "shutdown", 1, _at(power=0.15)),
        PackEntry("prop_cruise", "prop", 1, _at(power=0.6)),
        PackEntry("wind_slow", "wind", 1, _at(speed=0.25)),
        PackEntry("wind_cruise", "wind", 1, _at(speed=0.6)),
        PackEntry("wind_dive", "wind", 1, _at(speed=1.0)),
        PackEntry("airframe_cruise", "vibration", 1, _at(speed=0.6)),
        PackEntry("airframe_buffet", "vibration", 1,
                  _at(speed=0.8, intensity=1.0)),
        PackEntry("cockpit_cruise", "cockpit", 1, _at(power=0.6, speed=0.6)),
        PackEntry("cockpit_climb", "cockpit", 1, _at(power=1.0, speed=0.35)),
        PackEntry("stall_warning", "stall", 1, _at(intensity=0.7)),
        PackEntry("stall_deep", "stall", 1, _at(intensity=1.0)),
        PackEntry("touchdown", "touchdown", 2, _at(speed=0.5)),
        PackEntry("touchdown_fast", "touchdown", 2, _at(speed=0.9)),
        PackEntry("crash", "crash", 2, _at(intensity=1.0)),
    ),
)


# The legacy `flight` pack above is a descriptive library bank.  The game does
# not probe those extra stems or variations: it opens six fixed filenames.  Keep
# this second pack deliberately small so copying it into assets/sfx cannot leave
# a plausible-looking bank the runtime silently ignores.
FLIGHT_SIM_STEMS = Pack(
    "flight-sim",
    "the six aircraft stems Flight Simulator loads by exact filename",
    (
        PackEntry("engine_idle", "engine", 1, _at(power=0.1)),
        PackEntry("engine_power", "engine", 1, _at(power=1.0)),
        PackEntry("wind_low", "wind", 1, _at(speed=0.25)),
        PackEntry("wind_high", "wind", 1, _at(speed=1.0)),
        PackEntry("stall_warning", "stall", 1, _at(intensity=0.7)),
        PackEntry("touchdown", "touchdown", 1, _at(speed=0.5)),
    ),
)


LUNAR_LANDER = Pack(
    "lander",
    "rocket and thruster bank for Lunar Lander",
    (
        PackEntry("descent_low", "engine", 1,
                  _at("lander", "rocket", power=0.25)),
        PackEntry("descent_mid", "engine", 1,
                  _at("lander", "rocket", power=0.6)),
        PackEntry("descent_full", "engine", 1,
                  _at("lander", "rocket", power=1.0)),
        PackEntry("descent_ignition", "start", 1,
                  _at("lander", "rocket", power=0.6)),
        PackEntry("descent_cutoff", "shutdown", 1,
                  _at("lander", "rocket", power=0.6)),
        PackEntry("rcs_burst", "start", 3, _at("lander", "rcs", power=0.8)),
        PackEntry("rcs_hold", "engine", 1, _at("lander", "rcs", power=0.8)),
        PackEntry("cockpit_descent", "cockpit", 1,
                  _at("lander", "rocket", power=0.6, speed=0.3)),
        PackEntry("hull_vibration", "vibration", 1,
                  _at("lander", "rocket", speed=0.5, intensity=0.7)),
        PackEntry("alarm_caution", "alarm", 1,
                  _at("lander", "rocket", intensity=0.5)),
        PackEntry("alarm_urgent", "alarm", 1,
                  _at("lander", "rocket", intensity=1.0)),
        PackEntry("touchdown_soft", "touchdown", 2,
                  _at("lander", "rocket", speed=0.2, intensity=0.5)),
        PackEntry("touchdown_hard", "touchdown", 2,
                  _at("lander", "rocket", speed=0.6, intensity=1.0)),
        PackEntry("crash", "crash", 2, _at("lander", "rocket", intensity=1.0)),
    ),
)


# The same sounds as `lander` above, cut down to the four the game loads and
# named the way it loads them. Descent burn is the rocket and the attitude
# thrusters are the RCS, so `thrust_main` and `thrust_side` are one cue on two
# powerplants; both run short because the game holds them under a control input.
LUNAR_LANDER_STEMS = Pack(
    "lunar-lander",
    "the thruster and landing stems Lunar Lander loads by name",
    (
        PackEntry("thrust_main", "engine", 3,
                  _at("lander", "rocket", power=1.0, duration=1.0)),
        PackEntry("thrust_side", "engine", 3,
                  _at("lander", "rcs", power=0.8, duration=0.75)),
        PackEntry("landing", "touchdown", 3,
                  _at("lander", "rocket", speed=0.2, intensity=0.5)),
        PackEntry("crash", "crash", 4, _at("lander", "rocket", intensity=1.0)),
    ),
)


PACKS = {
    pack.name: pack
    for pack in (SHOWCASE, POWERPLANT_MATRIX, FLIGHT_SIM,
                 FLIGHT_SIM_STEMS, LUNAR_LANDER,
                 LUNAR_LANDER_STEMS)
}
