"""Procedural aircraft and spacecraft synthesis: propulsion, airframe, alarms.

``car_engine_generator`` models a crankshaft and ``vehicle_foley_generator``
models a contact patch.  Neither is much use once the wheels leave the ground.
An aeroplane's noise is dominated by things a car does not have: a propeller
beating the air a hundred times a second, a compressor spinning fast enough to
sing in the top octave, a plume of burnt gas expanding into open air, and a
whole airframe being shaken by the boundary layer flowing over it.  That is why
this is its own generator rather than another mode of the car one.

This module is the synthesised half and has no dependency beyond the standard
library. Impacts are *not* here: touchdown and crash are owned by
:mod:`crash_generator`, with :mod:`.hybrid` retained as a compatibility facade,
because a crash has no rate to model and synthesising one reliably produces a
bell over a beep. Everything below has a
rate -- a firing order, a blade passing, a shaft turning, air flowing -- which
is exactly why synthesis carries it.

Like ``weapons_destruction_generator`` and ``vehicle_foley_generator`` this
borrows the standard-library DSP core from ``combat_generators`` and adds no
dependency of its own.  It also borrows :mod:`environment_ambience_generator.loopdsp`,
which is the one piece of machinery this generator could not do without: almost
everything an aircraft makes is *continuous*, and a loop that ticks once a cycle
is worse than no loop at all.  Rather than crossfade a loop shut, every bed here
is periodic by construction -- filters are primed circularly, oscillators are
quantised to whole cycles per buffer -- so the last sample genuinely flows into
the first.  ``loopdsp.seam`` measures the result and the ``verify`` command
reports it.

Two registries cross here, and the cross product is the point.  A
:class:`PowerplantProfile` is a thing that makes thrust; a :class:`CraftProfile`
is the airframe it is bolted to and the cockpit you hear it from.  Propulsion
cues are the powerplant's alone, airframe cues are the craft's alone, and
:func:`cockpit_interior` is where the two meet -- which is why a radial in a
warbird and an ion drive in a freighter share every line of code below and no
timbre.

All renderers return unmastered mono floating-point PCM.  The cue registry owns
mastering, and picks the fade-free loop master for beds so their loop point
survives it.
"""

from __future__ import annotations

from ..._audio.dsp import write_wav as write_wav

import math
import random
from dataclasses import dataclass

from ..._audio import dsp
from ..._audio import loops as loopdsp
from ..._audio.rotor import RotorProfile, blade_pass


SR = dsp.SR
TAU = dsp.TAU

MIN_LOOP = 0.5
MAX_LOOP = 10.0

# Above this a partial is close enough to Nyquist to alias rather than sing.
CEILING = SR * 0.45


Modes = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class CraftProfile:
    """An airframe: what the air flows over and what the cockpit sits inside.

    ``wind_coupling`` is how hard the airstream drives the structure, and it is
    the field that lets a spacecraft use the same code as an aeroplane.  A
    lander descending through thin air is not silent, but it is not a fighter
    at sea level either, and the difference is a coefficient rather than a
    branch.

    ``gear`` picks what touches down.  Rubber chirps and skids; a footpad does
    not, and pretending otherwise is the single most obvious way to make a
    lander sound like a Cessna.
    """

    name: str
    label: str
    description: str
    mass: float
    skin_frequency: float
    cabin_frequency: float
    strut_frequency: float
    wind_low: float
    wind_high: float
    wind_coupling: float
    rattle_rate: float
    damping: float
    horn_frequency: float
    gear: str
    panel_modes: Modes


@dataclass(frozen=True)
class PowerplantProfile:
    """A thing that makes thrust, and the rates that follow from how.

    ``kind`` picks the mechanical model -- pistons bang, turbines spin, rockets
    roar, thrusters blow -- and everything else tunes it.  Fields a kind does
    not use are zero, in the same way a hard surface in the vehicle generator
    carries grain settings it never reads.

    The rates are the heart of it.  ``shaft_rate`` is the propeller or fan
    shaft in revolutions per second at full power and ``blades`` is what is
    bolted to it, so blade-pass is their product: a two-blade trainer prop at
    2700 rpm beats at 90 Hz, and a 22-blade fan at 4000 rpm sings at 1450 Hz.
    ``spool_frequency`` is the turbine core, which on a geared engine turns an
    order of magnitude faster than the prop in front of it -- which is exactly
    why a turboprop sounds like neither a piston engine nor a jet.
    """

    name: str
    label: str
    description: str
    kind: str
    cylinders: int
    shaft_rate: float
    blades: int
    open_prop: bool
    exhaust_frequency: float
    spool_frequency: float
    stages: int
    core_low: float
    core_high: float
    chamber_frequency: float
    chug_rate: float
    tone_gain: float
    noise_gain: float
    roughness: float
    idle: float
    spin_up: float


CRAFT: dict[str, CraftProfile] = {
    "trainer": CraftProfile(
        name="trainer", label="Trainer",
        description="light single-engine piston trainer; thin skin, no lining",
        mass=0.6, skin_frequency=190.0, cabin_frequency=240.0,
        strut_frequency=62.0, wind_low=140.0, wind_high=9_000.0,
        wind_coupling=0.85, rattle_rate=9.0, damping=0.25,
        horn_frequency=2_400.0, gear="tire",
        panel_modes=((430.0, 0.045, 0.34), (760.0, 0.031, 0.22),
                     (1290.0, 0.022, 0.14), (2050.0, 0.014, 0.09)),
    ),
    "warbird": CraftProfile(
        name="warbird", label="Warbird",
        description="heavy piston fighter; stressed skin over a big radial",
        mass=1.4, skin_frequency=120.0, cabin_frequency=165.0,
        strut_frequency=48.0, wind_low=110.0, wind_high=7_200.0,
        wind_coupling=0.95, rattle_rate=14.0, damping=0.30,
        horn_frequency=1_900.0, gear="tire",
        panel_modes=((300.0, 0.062, 0.36), (520.0, 0.042, 0.26),
                     (880.0, 0.028, 0.16), (1420.0, 0.020, 0.10)),
    ),
    "airliner": CraftProfile(
        name="airliner", label="Airliner",
        description="large turbofan transport; heavily insulated cabin",
        mass=3.2, skin_frequency=62.0, cabin_frequency=95.0,
        strut_frequency=22.0, wind_low=70.0, wind_high=6_000.0,
        wind_coupling=0.55, rattle_rate=4.0, damping=0.75,
        horn_frequency=1_000.0, gear="tire",
        panel_modes=((160.0, 0.084, 0.40), (275.0, 0.059, 0.29),
                     (460.0, 0.039, 0.18), (760.0, 0.028, 0.11)),
    ),
    "fighter": CraftProfile(
        name="fighter", label="Fighter",
        description="fast jet; tight canopy and a very loud boundary layer",
        mass=1.6, skin_frequency=145.0, cabin_frequency=200.0,
        strut_frequency=34.0, wind_low=160.0, wind_high=12_000.0,
        wind_coupling=1.00, rattle_rate=6.0, damping=0.55,
        horn_frequency=3_100.0, gear="tire",
        panel_modes=((360.0, 0.050, 0.32), (620.0, 0.036, 0.24),
                     (1050.0, 0.025, 0.15), (1700.0, 0.017, 0.09)),
    ),
    "lander": CraftProfile(
        name="lander", label="Lander",
        description="small rocket-powered lander; footpads, and thin air",
        mass=0.9, skin_frequency=210.0, cabin_frequency=260.0,
        strut_frequency=40.0, wind_low=90.0, wind_high=4_000.0,
        wind_coupling=0.25, rattle_rate=12.0, damping=0.45,
        horn_frequency=880.0, gear="pad",
        panel_modes=((480.0, 0.056, 0.34), (830.0, 0.039, 0.24),
                     (1410.0, 0.025, 0.15), (2260.0, 0.017, 0.09)),
    ),
    "freighter": CraftProfile(
        name="freighter", label="Freighter",
        description="heavy spacecraft; a big hull that rings low and slow",
        mass=4.0, skin_frequency=48.0, cabin_frequency=74.0,
        strut_frequency=18.0, wind_low=60.0, wind_high=3_200.0,
        wind_coupling=0.20, rattle_rate=7.0, damping=0.80,
        horn_frequency=620.0, gear="pad",
        panel_modes=((130.0, 0.095, 0.42), (225.0, 0.067, 0.30),
                     (380.0, 0.045, 0.19), (620.0, 0.031, 0.12)),
    ),
}


POWERPLANTS: dict[str, PowerplantProfile] = {
    "flat_four": PowerplantProfile(
        name="flat_four", label="Flat four",
        description="direct-drive light aircraft piston four, two-blade prop",
        kind="piston", cylinders=4, shaft_rate=45.0, blades=2, open_prop=True,
        exhaust_frequency=190.0, spool_frequency=0.0, stages=0,
        core_low=120.0, core_high=5_200.0, chamber_frequency=0.0, chug_rate=0.0,
        tone_gain=0.90, noise_gain=0.55, roughness=0.35, idle=0.26,
        spin_up=1.8,
    ),
    "radial": PowerplantProfile(
        name="radial", label="Radial",
        description="nine-cylinder radial through a collector ring, three blades",
        kind="piston", cylinders=9, shaft_rate=37.0, blades=3, open_prop=True,
        exhaust_frequency=120.0, spool_frequency=0.0, stages=0,
        core_low=90.0, core_high=4_200.0, chamber_frequency=0.0, chug_rate=0.0,
        tone_gain=1.00, noise_gain=0.70, roughness=0.55, idle=0.24,
        spin_up=2.8,
    ),
    "turboprop": PowerplantProfile(
        name="turboprop", label="Turboprop",
        description="turbine core geared down to a slow four-blade propeller",
        kind="turbine", cylinders=0, shaft_rate=32.0, blades=4, open_prop=True,
        exhaust_frequency=0.0, spool_frequency=250.0, stages=3,
        core_low=200.0, core_high=9_000.0, chamber_frequency=0.0, chug_rate=0.0,
        tone_gain=0.85, noise_gain=0.80, roughness=0.20, idle=0.50,
        spin_up=4.0,
    ),
    "turbofan": PowerplantProfile(
        name="turbofan", label="Turbofan",
        description="high-bypass transport fan; the buzz is 22 blades of it",
        kind="turbine", cylinders=0, shaft_rate=66.0, blades=22,
        open_prop=False, exhaust_frequency=0.0, spool_frequency=160.0,
        stages=4, core_low=90.0, core_high=7_000.0, chamber_frequency=0.0,
        chug_rate=0.0, tone_gain=0.80, noise_gain=1.00, roughness=0.12,
        idle=0.22, spin_up=6.0,
    ),
    "turbojet": PowerplantProfile(
        name="turbojet", label="Turbojet",
        description="low-bypass military jet; little fan, enormous plume",
        kind="turbine", cylinders=0, shaft_rate=180.0, blades=17,
        open_prop=False, exhaust_frequency=0.0, spool_frequency=200.0,
        stages=5, core_low=110.0, core_high=12_000.0, chamber_frequency=0.0,
        chug_rate=0.0, tone_gain=0.70, noise_gain=1.15, roughness=0.15,
        idle=0.25, spin_up=5.0,
    ),
    "rocket": PowerplantProfile(
        name="rocket", label="Rocket",
        description="throttleable liquid descent engine; roar, rumble, chug",
        kind="rocket", cylinders=0, shaft_rate=0.0, blades=0, open_prop=False,
        exhaust_frequency=0.0, spool_frequency=0.0, stages=0,
        core_low=30.0, core_high=9_000.0, chamber_frequency=34.0,
        chug_rate=11.0, tone_gain=0.35, noise_gain=1.30, roughness=0.60,
        idle=0.0, spin_up=1.1,
    ),
    "rcs": PowerplantProfile(
        name="rcs", label="RCS thruster",
        description="cold-gas attitude thruster; a valve and a small hole",
        kind="thruster", cylinders=0, shaft_rate=0.0, blades=0,
        open_prop=False, exhaust_frequency=0.0, spool_frequency=0.0, stages=0,
        core_low=900.0, core_high=16_000.0, chamber_frequency=2_100.0,
        chug_rate=0.0, tone_gain=0.30, noise_gain=1.00, roughness=0.30,
        idle=0.0, spin_up=0.06,
    ),
    "ion": PowerplantProfile(
        name="ion", label="Ion drive",
        description="electrostatic drive; almost all tone and almost no gas",
        kind="thruster", cylinders=0, shaft_rate=0.0, blades=0,
        open_prop=False, exhaust_frequency=0.0, spool_frequency=0.0, stages=0,
        core_low=300.0, core_high=12_000.0, chamber_frequency=220.0,
        chug_rate=0.0, tone_gain=1.20, noise_gain=0.35, roughness=0.08,
        idle=0.15, spin_up=2.2,
    ),
}


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _check_unit(value: float, name: str) -> float:
    if not 0.05 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.05 and 1")
    return value


def _check_power(power: float) -> float:
    return _check_unit(power, "power")


def _check_speed(speed: float) -> float:
    return _check_unit(speed, "speed")


def _check_intensity(intensity: float) -> float:
    return _check_unit(intensity, "intensity")


def _check_duration(duration: float) -> float:
    if not MIN_LOOP <= duration <= MAX_LOOP:
        raise ValueError(f"duration must be between {MIN_LOOP} and {MAX_LOOP}")
    return duration


def _rate(plant: PowerplantProfile, base: float, power: float) -> float:
    """A shaft rate at a power setting.

    Nothing here spins down to nothing at low power except the things that
    genuinely stop: an engine at idle still turns its prop at a quarter of
    redline, while a rocket at zero throttle is simply off.  That is what
    ``idle`` carries.
    """

    return base * (plant.idle + (1.0 - plant.idle) * power)


def _vibrato(duration: float, rate: float, depth: float,
             rng: random.Random) -> list[float] | None:
    """A zero-mean frequency multiplier that keeps a tone loopable.

    A tone survives the wrap only if it completes whole cycles in the buffer,
    which means the *average* frequency has to stay exactly where
    ``loopdsp.quantize`` put it.  ``loopdsp.lfo`` cannot be used for this: it is
    unipolar, so it would drag the mean down and quietly detune the loop into a
    click.  A whole-cycle sine sums to zero over the buffer, so this leaves the
    cycle count intact and moves only where inside the loop the pitch sits.
    """

    if depth <= 0.0:
        return None
    wave = loopdsp.sine(rate, duration, rng.uniform(0.0, TAU), minimum=0)
    return [1.0 + depth * value for value in wave]


def _harmonic_stack(freq: float, duration: float, rng: random.Random,
                    count: int, rolloff: float = 1.0,
                    vibrato: list[float] | None = None) -> list[float]:
    """A loopable harmonic series on ``freq``.

    The fundamental is quantised once and the partials are integer multiples of
    the result, so they are exactly harmonic *and* exactly whole-cycle.
    Quantising each partial separately would round each one differently and
    leave the series subtly out of tune with itself.
    """

    base = loopdsp.quantize(freq, duration, 1)
    out = dsp.silence(duration)
    for index in range(1, count + 1):
        partial = base * index
        if partial >= CEILING:
            break
        gain = 1.0 / (index ** rolloff)
        if vibrato is None:
            voice = dsp.sine(partial, duration, rng.uniform(0.0, TAU))
        else:
            voice = dsp.sine_track(
                [partial * value for value in vibrato], rng.uniform(0.0, TAU)
            )
        out = dsp.mix(out, dsp.scale(voice, gain))
    return out


def _cycle_impulses(duration: float, rng: random.Random, rate: float,
                    jitter: float = 0.0,
                    gains: list[float] | None = None) -> list[float]:
    """An impulse train carrying a whole number of events per loop.

    Placing by cycle index rather than by walking a period forward is what
    makes this loopable: the count is exact, so the train's rate is exactly
    ``count / period`` and the gap across the wrap is the same as every other
    gap.  ``gains`` cycles per impulse, which is how a firing order gets its
    pattern.
    """

    count = loopdsp.cycles(rate, duration, 1)
    out = dsp.silence(duration)
    limit = len(out)
    step = limit / count
    for index in range(count):
        position = index * step
        if jitter > 0.0:
            position += rng.uniform(-jitter, jitter) * step
        gain = 1.0 if gains is None else gains[index % len(gains)]
        out[int(position) % limit] += gain * rng.uniform(0.92, 1.0)
    return out


def _events(duration: float, rng: random.Random, count: int, build
            ) -> list[float]:
    """Scatter rendered one-shots across a loop, wrapping their tails.

    An event does not have to be periodic to live in a loop; it only has to
    come round to the start when it runs off the end, which is what a rattle
    beginning just before the loop point actually does.
    """

    out = dsp.silence(duration)
    for onset in loopdsp.scatter(duration, rng, count):
        event, gain = build(rng)
        loopdsp.add_wrapped(out, onset, event, gain)
    return out


def _per_loop(rate: float, duration: float, minimum: int = 1) -> int:
    """Event count for a per-second rate, never rounding a texture to silence."""

    return max(minimum, int(round(rate * loopdsp.period(duration))))


def _even_cycles(rate: float, duration: float) -> int:
    """Whole cycles per loop, rounded up to an even count.

    Anything that alternates between two states needs this rather than
    ``loopdsp.cycles``.  An odd count puts the same state either side of the
    wrap -- two high beeps in a row, once per repeat -- and while the *timing*
    stays even (which is why a step measurement cannot see it), the pattern
    audibly stutters at the loop point.
    """

    count = max(2, loopdsp.cycles(rate, duration, 1))
    return count + (count % 2)


def _snap_to_block(freq: float, block: float) -> float:
    """Nearest frequency completing whole cycles inside one block.

    A tone that steps between pitches is only loopable if each step holds a
    whole number of cycles: then the phase at every block edge is the phase it
    started at, so the steps are continuous and the total over the buffer is a
    whole number too.
    """

    return max(1, round(freq * block / SR)) * SR / block


def _mode_q(freq: float, decay: float) -> float:
    """The Q a mode must have to ring for as long as it does when struck.

    A resonator's Q and a struck mode's decay are the same physical fact seen
    twice: Q = pi * f * decay.  Deriving one from the other is what keeps a
    shaken airframe and a hit airframe sounding like the same aeroplane.
    """

    return max(3.0, min(60.0, math.pi * freq * decay))


def _thud(duration: float, rng: random.Random, freq: float, decay: float,
          q: float = 4.5, cutoff: float = 400.0) -> list[float]:
    """A low mass moving, pitched but not tonal.

    The sine-sweep-plus-envelope this replaces is the textbook way to draw a
    boom, and it is also the textbook way to draw a computer beep -- they are
    the same signal.  What separates a thud from a beep is that a real one is
    noise being shaped by a resonance, so there is no pure tone in it at any
    point.
    """

    drive = dsp.mul(
        dsp.lowpass(dsp.noise(duration, rng), cutoff, 4),
        dsp.exp_env(duration, decay, 0.0015),
    )
    return dsp.resonator(drive, max(18.0, freq), q=q, gain=1.0)


# --------------------------------------------------------------------------
# propulsion
# --------------------------------------------------------------------------

def propeller(plant: PowerplantProfile, rng: random.Random,
              power: float = 0.6, duration: float = 2.5,
              intensity: float = 0.85) -> list[float]:
    """Render a seamless loop of a propeller or fan beating the air.

    Two mechanisms, and a prop is the balance between them.  *Loading* noise is
    tonal: each blade carries a pressure field around with it, and every pass
    of a blade past your ear is one cycle of a tone at blade-pass frequency.
    *Vortex* noise is broadband: the tips shed turbulence into the air behind
    them, and it hisses.  Power moves both -- faster blades beat higher and
    shed harder -- which is why a prop cannot be pitch-shifted convincingly and
    has to be resynthesised per setting.

    A ducted fan is the same maths with the air trapped: more tone, less spill,
    and the duct itself ringing.  That is most of what separates an airliner's
    smooth buzz from a warbird's open snarl.
    """

    return blade_pass(
        RotorProfile(
            name=plant.name,
            shaft_rate=plant.shaft_rate,
            blades=plant.blades,
            open_rotor=plant.open_prop,
            tone_gain=plant.tone_gain,
            noise_gain=plant.noise_gain,
            roughness=plant.roughness,
            idle=plant.idle,
        ),
        rng,
        power=power,
        duration=duration,
        intensity=intensity,
    )


def piston_engine(plant: PowerplantProfile, rng: random.Random,
                  power: float = 0.6, duration: float = 2.5,
                  intensity: float = 0.85) -> list[float]:
    """Render a seamless loop of a piston aero-engine and its propeller.

    A four-stroke fires every other revolution, so the exhaust note sits at
    ``cylinders / 2`` times shaft speed -- 90 Hz for a four at 2700 rpm, 165 Hz
    for a nine at 2200.  Each firing is an impulse; the stack rings in response.
    That is the whole model, and it is why the exhaust pitch and the prop pitch
    are locked together on a direct-drive engine and beat against each other on
    anything else.

    The roughness is deliberately *per cylinder* rather than per firing.  A
    lumpy engine is lumpy the same way every revolution, because it is the same
    weak cylinder coming round again; randomising each bang instead would give
    a smooth engine with a noisy exhaust, which is a different aeroplane.
    """

    power = _check_power(power)
    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)
    if plant.kind != "piston":
        raise ValueError(f"{plant.name} is not a piston engine")

    shaft = _rate(plant, plant.shaft_rate, power)
    fire = shaft * plant.cylinders * 0.5

    cylinders = [
        1.0 - plant.roughness * rng.uniform(0.0, 0.5)
        for _ in range(max(1, plant.cylinders))
    ]
    pulses = _cycle_impulses(duration, rng, fire, 0.03 * plant.roughness,
                             cylinders)

    exhaust = dsp.mix(
        loopdsp.resonator(pulses, plant.exhaust_frequency, q=3.2, gain=1.0),
        loopdsp.resonator(pulses, plant.exhaust_frequency * 2.4, q=5.0,
                          gain=0.45),
    )
    exhaust = loopdsp.lowpass(exhaust, 900.0 + 4_200.0 * power, 4)

    # The half-order lope: one crank imbalance per revolution, under the note.
    lope = _cycle_impulses(duration, rng, shaft)
    lope = loopdsp.resonator(lope, plant.exhaust_frequency * 0.55, q=2.4,
                             gain=0.5)

    intake = loopdsp.bandpass(
        dsp.noise(duration, rng),
        plant.core_low * 2.5,
        min(CEILING, plant.core_high * (0.40 + 0.60 * power)),
        2,
    )
    intake = dsp.mul(intake, loopdsp.lfo(fire, duration, 0.5))

    valve = loopdsp.bandpass(
        _cycle_impulses(duration, rng, shaft * max(1, plant.cylinders), 0.25),
        2_600.0, 11_000.0, 2,
    )

    out = dsp.mix(
        dsp.scale(exhaust, 0.95 * plant.tone_gain),
        dsp.scale(lope, 0.30 * plant.roughness),
        dsp.scale(intake, (0.22 + 0.30 * power) * plant.noise_gain),
        dsp.scale(valve, 0.10 * (1.0 - 0.5 * power)),
    )
    if plant.blades > 0:
        out = dsp.mix(
            out, dsp.scale(propeller(plant, rng, power, duration, intensity),
                           0.85)
        )

    level = (0.30 + 0.70 * power ** 1.1) * (0.55 + 0.52 * intensity)
    return dsp.scale(out, level)


def turbine_engine(plant: PowerplantProfile, rng: random.Random,
                   power: float = 0.6, duration: float = 2.5,
                   intensity: float = 0.85) -> list[float]:
    """Render a seamless loop of a gas turbine.

    A turbine is a stack of tones over a roar.  Every compressor stage is a
    ring of blades passing a fixed vane at ``blades x core speed``, and since
    stages get finer as they go deeper, they sing progressively higher -- 21,
    34, 47 blades and up, so a 160 rev/s core lays whines across 5 kHz and
    beyond.  The roar underneath is the combustor and the plume, and it is the
    part that grows fastest with throttle: a jet at idle is a whistle, and a
    jet at full power is mostly wind.

    The fan or propeller on the front, if there is one, is
    :func:`propeller`'s job -- on a geared engine it turns at a tenth of the
    core's speed, and that ratio is exactly what a turboprop sounds like.
    """

    power = _check_power(power)
    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)
    if plant.kind != "turbine":
        raise ValueError(f"{plant.name} is not a turbine")

    spool = _rate(plant, plant.spool_frequency, power)

    roar = loopdsp.bandpass(
        dsp.noise(duration, rng),
        plant.core_low,
        min(CEILING, plant.core_high * (0.42 + 0.70 * power)),
        4,
    )
    rumble = loopdsp.lowpass(dsp.noise(duration, rng), 90.0 + 140.0 * power, 4)

    hunt = _vibrato(duration, 1.9, 0.0018, rng)
    whine = dsp.silence(duration)
    for stage in range(1, plant.stages + 1):
        # Blade count climbs with depth; the tones climb with it.
        blades = 21 + 13 * stage
        voice = _harmonic_stack(spool * blades, duration, rng, 2, 2.0, hunt)
        whine = dsp.mix(
            dsp.scale(voice, 0.20 / stage ** 0.8 * plant.tone_gain), whine
        )

    jet = loopdsp.bandpass(
        dsp.noise(duration, rng),
        400.0 + 900.0 * power,
        min(CEILING, 14_000.0 * (0.5 + 0.5 * power)),
        2,
    )

    out = dsp.mix(
        dsp.scale(roar, (0.45 + 0.55 * power) * plant.noise_gain),
        dsp.scale(rumble, 0.55 + 0.35 * power),
        whine,
        dsp.scale(jet, (0.10 + 0.55 * power ** 1.6) * plant.noise_gain),
    )
    if plant.blades > 0:
        out = dsp.mix(
            out, dsp.scale(propeller(plant, rng, power, duration, intensity),
                           1.15 if plant.open_prop else 0.75)
        )

    level = (0.18 + 0.82 * power ** 1.3) * (0.55 + 0.52 * intensity)
    return dsp.scale(out, level)


def rocket_motor(plant: PowerplantProfile, rng: random.Random,
                 power: float = 0.6, duration: float = 2.5,
                 intensity: float = 0.85) -> list[float]:
    """Render a seamless loop of a rocket motor.

    A rocket has no moving parts worth hearing, so unlike every other
    powerplant here there is no rate to track and nothing to tune to.  What
    there is instead is an enormous volume of gas tearing itself apart on the
    way out, which is broadband from the sub-bass up.

    Two details do the identifying.  *Chug* is low-frequency combustion
    instability -- the chamber breathing against its own feed system -- and it
    is worst at low throttle, where the flow has least authority; a deep-throttled
    engine gutters and a full one does not.  *Crackle* is the shock cells in
    the plume collapsing, and it is the one part of a rocket that is not smooth:
    sparse, violent, and the reason a rocket recording clips where an equally
    loud waterfall does not.
    """

    power = _check_power(power)
    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)
    if plant.kind != "rocket":
        raise ValueError(f"{plant.name} is not a rocket")

    plume = loopdsp.bandpass(
        dsp.noise(duration, rng),
        plant.core_low,
        min(CEILING, plant.core_high * (0.50 + 0.60 * power)),
        2,
    )
    body = loopdsp.lowpass(dsp.noise(duration, rng), 70.0 + 130.0 * power, 4)

    # The chamber has a pitch, but it is drowned in its own noise: a clean tone
    # here would read as a synth bass rather than a motor.
    hunt = _vibrato(duration, 3.7, 0.02 * plant.roughness, rng)
    chamber = _harmonic_stack(plant.chamber_frequency, duration, rng, 4, 1.4,
                              hunt)
    chamber = dsp.mul(
        chamber, loopdsp.drift(duration, (2.3, 5.9, 13.1), rng, 0.7)
    )

    crackle = loopdsp.bandpass(
        loopdsp.impulses(duration, rng, _per_loop(160.0 * (0.4 + power),
                                                  duration)),
        1_800.0, min(CEILING, 16_000.0), 2,
    )

    out = dsp.mix(
        dsp.scale(plume, (0.55 + 0.45 * power) * plant.noise_gain),
        dsp.scale(body, 0.95 + 0.35 * power),
        dsp.scale(chamber, 0.50 * plant.tone_gain),
        dsp.scale(crackle, 0.28 * plant.noise_gain * (0.4 + 0.6 * power)),
    )

    depth = 0.45 * (1.0 - 0.6 * power) * plant.roughness
    out = dsp.mul(out, loopdsp.lfo(plant.chug_rate, duration, depth))
    out = dsp.mul(out, loopdsp.drift(duration, (1.7, 4.3), rng, 0.18))

    level = (0.25 + 0.75 * power ** 1.25) * (0.55 + 0.52 * intensity)
    return dsp.scale(out, level)


def thruster_jet(plant: PowerplantProfile, rng: random.Random,
                 power: float = 0.6, duration: float = 2.5,
                 intensity: float = 0.85) -> list[float]:
    """Render a seamless loop of a small thruster.

    The same code covers a cold-gas jet and an ion drive, and ``tone_gain``
    against ``noise_gain`` is the entire difference.  Gas through an orifice is
    hiss with a whistle in it.  An electrostatic drive has no meaningful gas
    flow at all, so what is left is the coherent hum of the thing driving it --
    which is why the sci-fi one sounds like a tone with a hiss on top and the
    plausible one sounds like a hiss with a tone in it.
    """

    power = _check_power(power)
    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)
    if plant.kind != "thruster":
        raise ValueError(f"{plant.name} is not a thruster")

    hiss = loopdsp.bandpass(
        dsp.noise(duration, rng),
        plant.core_low,
        min(CEILING, plant.core_high * (0.50 + 0.55 * power)),
        4,
    )
    hunt = _vibrato(duration, 5.3, 0.004 + 0.02 * plant.roughness, rng)
    whistle = _harmonic_stack(plant.chamber_frequency, duration, rng, 4, 1.6,
                              hunt)

    # A shimmer that is not quite a beat: several slow drifts over the tone.
    shimmer = loopdsp.drift(duration, (0.9, 2.7, 6.1), rng, 0.35)

    out = dsp.mix(
        dsp.scale(hiss, (0.55 + 0.45 * power) * plant.noise_gain),
        dsp.scale(dsp.mul(whistle, shimmer), 0.42 * plant.tone_gain),
    )
    out = dsp.mul(out, loopdsp.drift(duration, (1.3, 3.7), rng, 0.14))

    level = (0.30 + 0.70 * power) * (0.55 + 0.52 * intensity)
    return dsp.scale(out, level)


_RUNNERS = {
    "piston": piston_engine,
    "turbine": turbine_engine,
    "rocket": rocket_motor,
    "thruster": thruster_jet,
}


def run_powerplant(plant: PowerplantProfile, rng: random.Random,
                   power: float = 0.6, duration: float = 2.5,
                   intensity: float = 0.85) -> list[float]:
    """Render whichever powerplant this is, running at ``power``.

    Running is one verb, so it is one cue; the profile decides what it means.
    """

    return _RUNNERS[plant.kind](plant, rng, power, duration, intensity)


# --------------------------------------------------------------------------
# transients
# --------------------------------------------------------------------------

def _spin_sweep(plant: PowerplantProfile, rng: random.Random, duration: float,
                start_power: float, end_power: float,
                curve: float = 1.0) -> list[float]:
    """The tonal and broadband layers of a powerplant over changing power.

    The loop renderers cannot do this: every trick that makes them seamless --
    quantised partials, circularly primed filters -- assumes the rates are
    standing still.  A start is the opposite of a loop, so it is built the
    opposite way, with per-sample frequency tracks and a filter that sweeps.
    """

    powers = [max(0.02, value) for value in
              dsp.ramp(duration, start_power, end_power, curve)]
    out = dsp.silence(duration)

    if plant.blades > 0 and plant.shaft_rate > 0.0:
        blade = [_rate(plant, plant.shaft_rate, p) * plant.blades
                 for p in powers]
        for harmonic in (1, 2, 3):
            track = [value * harmonic for value in blade]
            if max(track) >= CEILING:
                continue
            dsp.add_at(out, 0.0, dsp.sine_track(track, rng.uniform(0.0, TAU)),
                       0.55 / harmonic * plant.tone_gain)

    if plant.spool_frequency > 0.0:
        for stage in range(1, plant.stages + 1):
            blades = 21 + 13 * stage
            track = [_rate(plant, plant.spool_frequency, p) * blades
                     for p in powers]
            if max(track) >= CEILING:
                continue
            dsp.add_at(out, 0.0, dsp.sine_track(track, rng.uniform(0.0, TAU)),
                       0.18 / stage ** 0.8 * plant.tone_gain)

    if plant.chamber_frequency > 0.0:
        track = [plant.chamber_frequency * (0.72 + 0.28 * p) for p in powers]
        if max(track) < CEILING:
            dsp.add_at(out, 0.0, dsp.sine_track(track, rng.uniform(0.0, TAU)),
                       0.45 * plant.tone_gain)

    centres = [
        plant.core_low + (plant.core_high - plant.core_low) * 0.35 * p
        for p in powers
    ]
    roar = dsp.svf_bandpass(dsp.noise(duration, rng), centres, q=0.8)
    dsp.add_at(out, 0.0, roar, 0.75 * plant.noise_gain)

    if plant.kind == "piston":
        # The bangs have to accelerate too, or it spins up silently and then
        # starts firing all at once.
        fire = [_rate(plant, plant.shaft_rate, p) * plant.cylinders * 0.5
                for p in powers]
        bangs = dsp.silence(duration)
        position = 0.0
        limit = len(bangs)
        while position < limit:
            index = int(position)
            bangs[index] += rng.uniform(0.6, 1.0)
            position += SR / max(fire[index], 1.0)
        dsp.add_at(
            out, 0.0,
            dsp.lowpass(dsp.resonator(bangs, plant.exhaust_frequency, q=3.0),
                        3_500.0, 4),
            0.9 * plant.tone_gain,
        )

    return dsp.mul(out, [0.12 + 0.88 * p for p in powers])


def start_powerplant(plant: PowerplantProfile, rng: random.Random,
                     power: float = 0.6,
                     intensity: float = 0.85) -> list[float]:
    """Render a powerplant starting and running up to ``power``.

    Every loop in this generator needs one of these, for the same reason
    ``vehicle_foley_generator``'s nitrous loop needs its boost burst: a loop
    cannot contain its own attack and still loop.  What the attack *is* differs
    completely by kind -- a piston engine is cranked until it catches, a
    turbine is spun until it lights, a rocket is simply ignited -- so this is
    the one place the four models genuinely diverge rather than share.
    """

    power = _check_power(power)
    intensity = _check_intensity(intensity)
    duration = plant.spin_up * (0.85 + 0.35 * intensity)

    if plant.kind == "piston":
        crank = duration * 0.55
        out = dsp.silence(duration)
        # Starter motor: a small electric machine through a reduction gear.
        whine = dsp.mul(
            dsp.sine_track(dsp.ramp(crank, 55.0, 150.0, 0.7),
                           rng.uniform(0.0, TAU)),
            dsp.ramp(crank, 0.0, 1.0, 0.25),
        )
        gear = dsp.mul(
            dsp.bandpass(dsp.noise(crank, rng), 900.0, 7_000.0, 2),
            dsp.ramp(crank, 0.0, 1.0, 0.3),
        )
        dsp.add_at(out, 0.0, whine, 0.55)
        dsp.add_at(out, 0.0, gear, 0.28)
        # Compression strokes: the engine is turning, just not firing.
        position = 0.0
        while position < crank:
            thump = dsp.mul(
                dsp.lowpass(dsp.noise(0.06, rng), 700.0, 4),
                dsp.exp_env(0.06, 0.012, 0.002),
            )
            dsp.add_at(out, position, thump, rng.uniform(0.4, 0.7))
            position += 1.0 / max(2.0, plant.shaft_rate * 0.2)
        # Catch: it fires, it stumbles, it settles.
        dsp.add_at(
            out, crank,
            _spin_sweep(plant, rng, duration - crank, 0.10, power, 0.55),
            1.0,
        )
        bark = dsp.mul(
            dsp.lowpass(dsp.noise(0.18, rng), 2_400.0, 4),
            dsp.exp_env(0.18, 0.03, 0.001),
        )
        dsp.add_at(out, crank, bark, 0.7)
        return out

    if plant.kind == "turbine":
        out = _spin_sweep(plant, rng, duration, 0.03, power, 1.7)
        # Air starter, before there is any combustion to hear.
        starter = duration * 0.45
        dsp.add_at(
            out, 0.0,
            dsp.mul(
                dsp.sine_track(dsp.ramp(starter, 260.0, 1_400.0, 1.4),
                               rng.uniform(0.0, TAU)),
                dsp.bell_env(starter, starter * 0.6, starter * 0.5,
                             starter * 0.35),
            ),
            0.30,
        )
        # Light-off: fuel meets a hot core and the roar arrives all at once.
        ignition = duration * 0.35
        whump = dsp.mul(
            dsp.lowpass(dsp.noise(0.5, rng), 320.0, 4),
            dsp.exp_env(0.5, 0.10, 0.008),
        )
        dsp.add_at(out, ignition, whump, 0.85 + 0.3 * intensity)
        return out

    if plant.kind == "rocket":
        out = _spin_sweep(plant, rng, duration, 0.02, power, 0.45)
        sparks = dsp.mul(
            dsp.bandpass(
                loopdsp.impulses(0.22, rng, 90), 2_500.0, 14_000.0, 2
            ),
            dsp.exp_env(0.22, 0.06),
        )
        dsp.add_at(out, 0.0, sparks, 0.5)
        # The chamber filling: a hard start is a thump, not a fade-in.
        kick = dsp.mul(
            dsp.sine_track(
                dsp.ramp(0.4, plant.chamber_frequency * 2.4,
                         plant.chamber_frequency * 0.9, 0.6),
                rng.uniform(0.0, TAU),
            ),
            dsp.exp_env(0.4, 0.09, 0.004),
        )
        dsp.add_at(out, 0.10, kick, 0.8 + 0.3 * intensity)
        return out

    out = _spin_sweep(plant, rng, duration, 0.05, power, 0.35)
    click = dsp.mul(
        dsp.bandpass(dsp.noise(0.03, rng), 1_400.0, 12_000.0, 2),
        dsp.exp_env(0.03, 0.004),
    )
    dsp.add_at(out, 0.0, click, 0.75)
    return out


def shutdown_powerplant(plant: PowerplantProfile, rng: random.Random,
                        power: float = 0.6,
                        intensity: float = 0.85) -> list[float]:
    """Render a powerplant winding down from ``power`` and stopping.

    A shutdown is not a start reversed, because the thing that made it spin up
    is not the thing that slows it down.  Nothing is driving it any more, so it
    coasts: the tones fall away on their own inertia and take far longer to go
    than they took to arrive, and the noise dies first because the flow does.
    """

    power = _check_power(power)
    intensity = _check_intensity(intensity)
    fast = plant.kind in ("rocket", "thruster")
    duration = plant.spin_up * (0.5 if fast else 1.5)

    out = _spin_sweep(plant, rng, duration, power, 0.02, 1.9 if fast else 0.8)

    if plant.kind == "piston":
        # The last compression stroke stops the engine dead.
        clunk = dsp.mix(
            dsp.mul(dsp.lowpass(dsp.noise(0.2, rng), 900.0, 4),
                    dsp.exp_env(0.2, 0.025, 0.001)),
            dsp.modal_ring(0.2, [(plant.exhaust_frequency * 1.4, 0.05, 0.5)],
                           rng),
        )
        dsp.add_at(out, duration * 0.88, clunk, 0.6)
    elif plant.kind == "turbine":
        # A stopped fan is still a fan: the airstream keeps it turning.
        windmill = dsp.mul(
            dsp.bandpass(dsp.noise(duration * 0.4, rng), 300.0, 4_000.0, 2),
            dsp.ramp(duration * 0.4, 1.0, 0.35, 0.6),
        )
        dsp.add_at(out, duration * 0.6, windmill, 0.22)
    else:
        valve = dsp.mul(
            dsp.bandpass(dsp.noise(0.04, rng), 1_200.0, 11_000.0, 2),
            dsp.exp_env(0.04, 0.006),
        )
        dsp.add_at(out, duration * 0.82, valve, 0.55)
    return out


# --------------------------------------------------------------------------
# airframe
# --------------------------------------------------------------------------

def airflow(craft: CraftProfile, rng: random.Random, speed: float = 0.5,
            duration: float = 2.5, intensity: float = 0.85) -> list[float]:
    """Render a seamless loop of air flowing over the airframe.

    This is not weather.  ``environment_ambience_generator`` has a wind cue and
    it is a different sound: that one gusts, because the air is moving and you
    are not.  Here *you* are the thing moving, so the flow is relentless and
    what varies is only the turbulence in it.

    Speed moves the spectrum, not just the level: the boundary layer over a
    fast aeroplane is thinner and much brighter, which is why a dive sounds
    like a hiss and a glide sounds like a rush.  ``wind_coupling`` is what
    keeps the same code honest in thin air -- a lander hears a quarter of what
    a fighter hears at the same number.
    """

    speed = _check_speed(speed)
    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)

    turbulence = loopdsp.drift(duration, (0.7, 1.9, 4.3), rng, 0.5)
    base = craft.wind_low + (craft.wind_high - craft.wind_low) * 0.25 * speed ** 1.1
    centres = [base * (0.6 + 0.8 * value) for value in turbulence]
    flow = loopdsp.svf_bandpass(dsp.noise(duration, rng), centres, q=0.7)

    body = loopdsp.lowpass(dsp.noise(duration, rng), 120.0 + 380.0 * speed, 4)
    hiss = loopdsp.bandpass(
        dsp.noise(duration, rng), 2_000.0,
        min(CEILING, craft.wind_high * (0.5 + 0.6 * speed)), 2,
    )
    # A seal, an aerial, a gap in a fairing: something always sings, and only
    # once there is enough dynamic pressure to drive it.
    whistle = loopdsp.sine(1_400.0 + 2_600.0 * speed, duration,
                           rng.uniform(0.0, TAU))

    out = dsp.mix(
        dsp.scale(flow, 0.85),
        dsp.scale(body, 0.55 + 0.30 * speed),
        dsp.scale(hiss, 0.20 + 0.45 * speed ** 1.5),
        dsp.scale(whistle, 0.09 * speed ** 2),
    )
    out = dsp.mul(out, loopdsp.drift(duration, (1.1, 2.9), rng, 0.22))

    level = (0.12 + 0.88 * speed ** 1.4) * craft.wind_coupling * (
        0.55 + 0.52 * intensity)
    return dsp.scale(out, level)


def airframe_vibration(craft: CraftProfile, rng: random.Random,
                       speed: float = 0.5, duration: float = 2.5,
                       intensity: float = 0.85) -> list[float]:
    """Render a seamless loop of the structure being shaken.

    An airframe is a set of panels and spars with modes, and the boundary layer
    is broadband noise pressing on all of them at once.  Each mode's ringing
    decay sets the Q it resonates at, so ``panel_modes`` describes one structure
    whether it is being shaken or hit.

    Being *shaken* is where that model works, and it is worth saying why: the
    excitation is genuinely broadband noise, so a resonator driven by noise is
    not an approximation of this cue, it is the physics of it.  The same modes
    under an *impact* went to recordings instead -- see :mod:`crash_generator`
    -- because
    there the excitation is a chaotic transient the ear knows too well to be
    fooled by a synthesised one.

    Intensity here is buffet, not volume -- smooth cruise through to the
    shuddering that arrives just before a wing stops flying.
    """

    speed = _check_speed(speed)
    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)

    drive = loopdsp.bandpass(
        dsp.noise(duration, rng), 40.0, 900.0 + 1_800.0 * speed, 2
    )
    structure = dsp.silence(duration)
    for freq, decay, gain in craft.panel_modes:
        structure = dsp.mix(
            structure,
            loopdsp.resonator(drive, freq, q=_mode_q(freq, decay), gain=gain),
        )

    skin = loopdsp.resonator(
        loopdsp.lowpass(dsp.noise(duration, rng), 400.0, 4),
        craft.skin_frequency, q=5.0, gain=1.0,
    )

    out = dsp.mix(
        dsp.scale(structure, 0.85),
        dsp.scale(skin, 0.70 + 0.30 * craft.mass / 4.0),
    )
    # Buffet: the shaking is not steady, and how unsteady is the whole point.
    out = dsp.mul(
        out, loopdsp.drift(duration, (1.3, 3.1, 6.7), rng,
                           0.25 + 0.45 * intensity)
    )

    # Spars work against their fittings, and complain about it.
    creaks = _events(
        duration, rng, _per_loop(1.4 + 3.0 * intensity, duration),
        lambda r: (
            dsp.mul(
                dsp.stick_slip(0.28, r, rate=18.0,
                               freq=craft.panel_modes[0][0] * 1.7, jitter=0.9),
                dsp.exp_env(0.28, 0.09, 0.02),
            ),
            r.uniform(0.10, 0.30) * intensity,
        ),
    )
    out = dsp.mix(out, creaks)

    level = (0.20 + 0.80 * speed) * craft.wind_coupling * (
        0.5 + 0.6 * intensity)
    return dsp.scale(out, level)


def cockpit_interior(craft: CraftProfile, plant: PowerplantProfile,
                     rng: random.Random, power: float = 0.6,
                     speed: float = 0.5, duration: float = 2.5,
                     intensity: float = 0.85) -> list[float]:
    """Render a seamless loop of the view from inside.

    This is the cue where the two registries actually meet, and it is a
    filtering problem rather than a synthesis one: everything outside is
    already modelled, and a cockpit is what a firewall, a window and some
    insulation do to it.  ``damping`` is the whole story -- an open trainer
    hears its engine almost raw, an airliner hears a rumour of one.

    The structure is driven by the mix rather than rendered separately, which
    is both cheaper and more correct: the panels around you are being shaken by
    that engine and that airflow, not by an independent copy of them.
    """

    power = _check_power(power)
    speed = _check_speed(speed)
    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)

    muffled = loopdsp.lowpass(
        run_powerplant(plant, rng, power, duration, intensity),
        max(160.0, 5_200.0 * (1.0 - 0.82 * craft.damping)), 4,
    )
    air = loopdsp.lowpass(
        airflow(craft, rng, speed, duration, intensity),
        max(300.0, 3_000.0 * (1.0 - 0.70 * craft.damping)), 4,
    )
    driven = dsp.mix(dsp.scale(muffled, 0.9), dsp.scale(air, 0.75))

    frame = dsp.silence(duration)
    for freq, decay, gain in craft.panel_modes:
        frame = dsp.mix(
            frame,
            loopdsp.resonator(driven, freq, q=_mode_q(freq, decay),
                              gain=gain * 0.30),
        )
    cavity = loopdsp.resonator(driven, craft.cabin_frequency, q=5.0, gain=0.45)

    # Avionics: a small fan and a supply whine, and they never stop.
    avionics = dsp.mix(
        dsp.scale(loopdsp.sine(400.0, duration, rng.uniform(0.0, TAU)), 0.010),
        dsp.scale(
            loopdsp.bandpass(dsp.noise(duration, rng), 700.0, 4_000.0, 2), 0.02
        ),
    )

    rattle = _events(
        duration, rng, _per_loop(craft.rattle_rate * (0.4 + 0.9 * intensity),
                                 duration),
        lambda r: (
            dsp.mul(
                dsp.bandpass(dsp.noise(0.045, r), 1_100.0, 9_000.0, 2),
                dsp.exp_env(0.045, 0.007),
            ),
            r.uniform(0.05, 0.22) * (0.3 + 0.7 * intensity),
        ),
    )

    out = dsp.mix(
        driven,
        dsp.scale(frame, 0.9),
        dsp.scale(cavity, 0.8),
        avionics,
        dsp.scale(rattle, 0.6 * (0.3 + 0.7 * power)),
    )
    return dsp.scale(out, 0.55 + 0.52 * intensity)


# --------------------------------------------------------------------------
# alarms
# --------------------------------------------------------------------------

# A minor third: far enough apart to read as two notes rather than as a wobble,
# close enough to stay one alarm rather than become a tune.
WARBLE_INTERVAL = 1.19


def stall_horn(craft: CraftProfile, rng: random.Random, duration: float = 2.5,
               intensity: float = 0.85) -> list[float]:
    """Render a seamless loop of a stall warning.

    On a light aircraft this is not electronics: it is a reed in a hole in the
    wing, and as the angle of attack goes up the stagnation point moves until
    air is sucked through it and it screams.  The reed is what it *is* -- rich
    in harmonics, breathy, never quite in tune with itself.

    But a warning that gets the fidelity right and the job wrong is worthless,
    and a steady tone is the one thing an ear reliably stops hearing.  That is
    the argument :func:`alarm_klaxon` already makes for two tones, and it
    applies with more force here: nothing in this generator matters more than
    being unmissable.  So the reed warbles between a pair a minor third apart.

    The interval is a readability choice and not a physical one, and is best
    admitted as such.  What *is* physical is that a reed under fluctuating flow
    hunts between its modes as the stall approaches; this leans on that to buy
    an alarm a pilot -- or a player -- cannot tune out.

    ``intensity`` is how deep into the stall you are, and it drives the reed
    sharper, breathier and faster, so the cue escalates on its own rather than
    by a game turning it up.  ``horn_frequency`` sets the register: the
    trainer's is a shrill mechanical reed and the freighter's is a low
    annunciator that only sounds like one because everything on that ship is
    old.
    """

    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)

    # Blocks per loop, always even so the alternation closes across the wrap.
    rate = 4.0 + 1.2 * intensity
    blocks = _even_cycles(rate, duration)
    total = dsp.samples(duration)
    block = total / blocks

    # A reed under more airflow blows slightly sharp.
    base = craft.horn_frequency * (0.96 + 0.08 * intensity)
    high = _snap_to_block(base * WARBLE_INTERVAL, block)
    low = _snap_to_block(base, block)
    track = [
        high if int(index / block) % 2 == 0 else low for index in range(total)
    ]

    tone = dsp.silence(duration)
    for harmonic in range(1, 6):
        if max(high, low) * harmonic >= CEILING:
            break
        voice = dsp.sine_track(
            [freq * harmonic for freq in track], rng.uniform(0.0, TAU)
        )
        tone = dsp.mix(tone, dsp.scale(voice, 1.0 / harmonic ** 0.85))

    breath = loopdsp.bandpass(
        dsp.noise(duration, rng), low * 0.8, min(CEILING, high * 7.0), 2
    )

    out = dsp.mix(
        dsp.scale(tone, 0.85),
        dsp.scale(breath, 0.12 + 0.09 * intensity),
    )
    return dsp.scale(out, 0.45 + 0.65 * intensity)


def alarm_klaxon(craft: CraftProfile, rng: random.Random,
                 duration: float = 2.5, intensity: float = 0.85
                 ) -> list[float]:
    """Render a seamless loop of a master caution klaxon.

    Two tones alternating, because one tone is a beep and two are an alarm: a
    fixed pitch fades into the background of a cockpit within seconds, and a
    switching one cannot.  Intensity is urgency, and it drives the repetition
    rate rather than the volume -- which is how a real annunciator escalates,
    and how a game can escalate one without simply turning it up.

    The beeps are placed as wrapped events at a whole number per loop, so the
    cadence closes at the loop point instead of stuttering there.
    """

    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)

    rate = 1.6 + 2.8 * intensity
    # Even, so the high/low alternation closes across the wrap rather than
    # landing two highs in a row once per repeat. See :func:`_even_cycles`.
    count = _even_cycles(rate, duration)
    span = loopdsp.period(duration)
    step = span / count
    beep = min(step * 0.5, 0.22)

    out = dsp.silence(duration)
    for index in range(count):
        freq = craft.horn_frequency * (1.0 if index % 2 == 0 else 0.75)
        voice = dsp.mix(
            dsp.sine(freq, beep, rng.uniform(0.0, TAU)),
            dsp.scale(dsp.sine(freq * 2.0, beep, rng.uniform(0.0, TAU)), 0.35),
            dsp.scale(dsp.sine(freq * 3.0, beep, rng.uniform(0.0, TAU)), 0.16),
        )
        # A hard edge on a klaxon is the point; a soft one is a doorbell.
        envelope = dsp.bell_env(beep, beep * 0.5, beep * 0.05, beep * 0.30)
        loopdsp.add_wrapped(out, index * step, dsp.mul(voice, envelope), 0.9)

    cabin = loopdsp.resonator(out, craft.cabin_frequency, q=4.0, gain=0.30)
    return dsp.mix(out, cabin)


# --------------------------------------------------------------------------
# lookups and io
# --------------------------------------------------------------------------

def craft_named(name: str) -> CraftProfile:
    try:
        return CRAFT[name]
    except KeyError:
        raise ValueError(
            f"unknown craft {name!r}; expected one of {sorted(CRAFT)}"
        ) from None


def powerplant_named(name: str) -> PowerplantProfile:
    try:
        return POWERPLANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown powerplant {name!r}; expected one of "
            f"{sorted(POWERPLANTS)}"
        ) from None
