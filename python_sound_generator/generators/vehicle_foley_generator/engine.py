"""Vehicle tires and chassis, with impact compatibility delegates.

This is the foley half of a driving soundscape; ``car_engine_generator`` owns
the engine half.  Nothing here models combustion, and nothing there models the
contact patch, so a game layers one over the other.

The tire, chassis, and air implementation borrows the standard-library DSP core
from ``combat_generators`` and adds no dependencies of its own. The public
``collision``, ``crash``, and ``debris`` functions are compatibility delegates
to :mod:`crash_generator`; their rejected modal-ring/sine-track implementation
was removed under item 10.

Two axes run through everything.  A :class:`VehicleProfile` sets mass and the
frequencies that follow from it, and a :class:`SurfaceProfile` sets what the
tire is touching.  The cross product is the point: a monster truck on mud and a
kart on concrete share every line of code below and no timbre.

Continuous foley — roll, drift, spray, nitrous — is looped rather than
triggered, and those cues are built with :mod:`environment_ambience_generator.
loopdsp`: filters primed circularly, oscillators and modulators quantised to
whole cycles per loop.  That makes a loop seamless by construction, so nothing
here crossfades a render onto itself or renders longer than it returns.  It also
carries a grain's ringing around the loop point for free — a stone landing just
before the end is filtered with the buffer's own tail in front of it, so its ring
decays into the start rather than off the end.

One-shots use the plain ``dsp`` filters, which must *not* be primed: priming
wraps a sound's tail into its own attack, which is exactly right for a loop and
pre-echo for an event.

All renderers return unmastered mono floating-point PCM.  The cue registry owns
mastering levels, so every public render follows the same contract.
"""

from __future__ import annotations

from ..._audio.dsp import write_wav as write_wav

import random
from dataclasses import dataclass

from ..._audio import dsp
from ..._audio import loops as loopdsp

from . import bank


SR = dsp.SR
TAU = dsp.TAU

MIN_LOOP = 0.5
MAX_LOOP = 10.0


Modes = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class VehicleProfile:
    """Mass and the acoustic frequencies that follow from it.

    Tread pattern is absent, and used to be here: the recorded bed carries the
    tread now, so a number describing it would be a number nothing reads.
    """

    name: str
    label: str
    description: str
    mass: float
    cavity_frequency: float
    body_frequency: float
    suspension_frequency: float
    squeal_frequency: float
    grip: float
    gear_gain: float
    panel_modes: Modes


@dataclass(frozen=True)
class SurfaceProfile:
    """What the contact patch is touching.

    ``kind`` picks the mechanical model (hard rolls, loose sheds grains, soft
    absorbs) and ``wetness`` layers water on top of any of them, so wet asphalt
    is asphalt plus a film rather than a separate surface with duplicated
    tuning.
    """

    name: str
    label: str
    description: str
    kind: str
    roll_low: float
    roll_high: float
    damping: float
    grip: float
    wetness: float
    grain_rate: float
    grain_low: float
    grain_high: float
    spray_gain: float


VEHICLES: dict[str, VehicleProfile] = {
    "kart": VehicleProfile(
        "kart", "Kart", "featherweight slick-shod kart with no suspension",
        0.3, 420.0, 260.0, 220.0, 1450.0, 1.0, 0.5,
        ((980.0, 0.1, 0.3), (1720.0, 0.07, 0.2), (2840.0, 0.05, 0.12)),
    ),
    "compact": VehicleProfile(
        "compact", "Compact", "small light front-drive hatchback",
        0.75, 240.0, 150.0, 95.0, 1050.0, 0.8, 0.6,
        ((520.0, 0.16, 0.34), (870.0, 0.11, 0.24), (1490.0, 0.08, 0.15), (2360.0, 0.05, 0.09)),
    ),
    "sedan": VehicleProfile(
        "sedan", "Sedan", "midsize road car; the reference vehicle",
        1.0, 215.0, 120.0, 82.0, 950.0, 0.75, 0.7,
        ((430.0, 0.19, 0.36), (720.0, 0.13, 0.26), (1240.0, 0.09, 0.16), (1980.0, 0.06, 0.1)),
    ),
    "muscle": VehicleProfile(
        "muscle", "Muscle car", "heavy rear-drive coupe on wide soft rubber",
        1.3, 200.0, 105.0, 74.0, 880.0, 0.95, 1.0,
        ((360.0, 0.22, 0.38), (610.0, 0.15, 0.27), (1020.0, 0.1, 0.17), (1670.0, 0.07, 0.1)),
    ),
    "buggy": VehicleProfile(
        "buggy", "Buggy", "tube-frame off-roader on knobbly tires",
        0.85, 180.0, 135.0, 58.0, 760.0, 0.5, 0.9,
        ((640.0, 0.14, 0.3), (1080.0, 0.1, 0.21), (1830.0, 0.07, 0.13)),
    ),
    "truck": VehicleProfile(
        "truck", "Truck", "body-on-frame pickup with long spring travel",
        2.1, 150.0, 78.0, 55.0, 640.0, 0.6, 1.25,
        ((250.0, 0.28, 0.4), (415.0, 0.19, 0.28), (690.0, 0.13, 0.18), (1130.0, 0.09, 0.11)),
    ),
    "monster": VehicleProfile(
        "monster", "Monster truck", "oversized lugs and enormous unsprung mass",
        3.2, 95.0, 62.0, 38.0, 420.0, 0.35, 1.4,
        ((180.0, 0.34, 0.42), (300.0, 0.24, 0.3), (505.0, 0.16, 0.19), (830.0, 0.11, 0.12)),
    ),
}


SURFACES: dict[str, SurfaceProfile] = {
    "asphalt": SurfaceProfile(
        "asphalt", "Asphalt", "dry sealed road",
        "hard", 220.0, 3000.0, 0.1, 1.0, 0.0,
        650.0, 800.0, 9000.0, 0.05,
    ),
    "concrete": SurfaceProfile(
        "concrete", "Concrete", "tined concrete highway; brighter than asphalt",
        "hard", 260.0, 3800.0, 0.06, 0.95, 0.0,
        520.0, 800.0, 9000.0, 0.04,
    ),
    "wet": SurfaceProfile(
        "wet", "Wet road", "sealed road under a film of standing water",
        "hard", 300.0, 6000.0, 0.2, 0.55, 1.0,
        260.0, 700.0, 12000.0, 1.0,
    ),
    "dirt": SurfaceProfile(
        "dirt", "Dirt", "dry packed earth and loose topsoil",
        "loose", 120.0, 2800.0, 0.45, 0.35, 0.0,
        900.0, 300.0, 5200.0, 0.75,
    ),
    "gravel": SurfaceProfile(
        "gravel", "Gravel", "loose stone chippings",
        "loose", 150.0, 4800.0, 0.3, 0.3, 0.0,
        2200.0, 900.0, 11000.0, 1.0,
    ),
    "sand": SurfaceProfile(
        "sand", "Sand", "deep fine sand",
        "loose", 100.0, 3200.0, 0.6, 0.28, 0.0,
        3200.0, 500.0, 9500.0, 0.85,
    ),
    "mud": SurfaceProfile(
        "mud", "Mud", "wet churned mud",
        "soft", 80.0, 2100.0, 0.7, 0.22, 0.9,
        420.0, 90.0, 1600.0, 1.1,
    ),
    "grass": SurfaceProfile(
        "grass", "Grass", "turf and light undergrowth",
        "soft", 90.0, 2400.0, 0.65, 0.4, 0.15,
        700.0, 250.0, 4800.0, 0.5,
    ),
    "snow": SurfaceProfile(
        "snow", "Snow", "packed snow over ice",
        "soft", 70.0, 2200.0, 0.8, 0.2, 0.35,
        1400.0, 200.0, 4200.0, 0.7,
    ),
}


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _check_unit(value: float, name: str) -> float:
    if not 0.05 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.05 and 1")
    return value


def _check_intensity(intensity: float) -> float:
    return _check_unit(intensity, "intensity")


def _check_speed(speed: float) -> float:
    return _check_unit(speed, "speed")


def _check_duration(duration: float) -> float:
    if not MIN_LOOP <= duration <= MAX_LOOP:
        raise ValueError(f"duration must be between {MIN_LOOP} and {MAX_LOOP}")
    return duration


def _noise_envelope(
    duration: float,
    rng: random.Random,
    low: float,
    high: float,
    decay: float,
    *,
    attack: float = 0.0,
    order: int = 4,
) -> list[float]:
    source = dsp.bandpass(dsp.noise(duration, rng), low, high, order)
    return dsp.mul(source, dsp.exp_env(duration, decay, attack))


def _low_noise(
    duration: float,
    rng: random.Random,
    cutoff: float,
    decay: float,
    *,
    attack: float = 0.0,
) -> list[float]:
    source = dsp.lowpass(dsp.noise(duration, rng), cutoff, 4)
    return dsp.mul(source, dsp.exp_env(duration, decay, attack))


def _bandpass(
    signal: list[float], low: float, high: float, order: int, loop: bool
) -> list[float]:
    """Bandpass, primed circularly for a loop and cold for a one-shot."""

    if loop:
        return loopdsp.bandpass(signal, low, high, order)
    return dsp.bandpass(signal, low, high, order)


def _grains(
    duration: float,
    rng: random.Random,
    rate: float,
    low: float,
    high: float,
    *,
    gain: float = 1.0,
    spread: float = 1.0,
    loop: bool = False,
) -> list[float]:
    """A shower of short filtered clicks: stones, grit, thrown debris.

    One shared impulse bed filtered once, rather than N filtered one-shots; at
    thousands of grains per second that is the difference between a loop
    rendering in a moment and rendering in a minute.
    """

    count = int(round(duration * rate))
    if count <= 0 or gain <= 0.0:
        return dsp.silence(duration)
    field = loopdsp.impulses(duration, rng, count, 0.3, 1.0)
    voiced = _bandpass(field, low, high, 2, loop)
    if spread != 1.0:
        heavy = (
            loopdsp.lowpass(voiced, high * 0.35, 2) if loop
            else dsp.lowpass(voiced, high * 0.35, 2)
        )
        voiced = dsp.mix(voiced, dsp.scale(heavy, spread - 1.0))
    return dsp.scale(voiced, gain)


def _water_film(
    duration: float,
    rng: random.Random,
    wetness: float,
    speed: float,
    *,
    loop: bool = False,
) -> list[float]:
    """Tires shedding water: broadband hiss plus a few discrete droplets."""

    if wetness <= 0.0:
        return dsp.silence(duration)
    hiss = _bandpass(
        dsp.noise(duration, rng), 900.0, 6_000.0 + 7_000.0 * speed, 2, loop
    )
    drops = _grains(
        duration, rng, 40.0 * wetness * speed, 1_200.0, 9_000.0, gain=0.55,
        loop=loop,
    )
    return dsp.scale(dsp.mix(dsp.scale(hiss, 0.7), drops), wetness * speed)


# --------------------------------------------------------------------------
# tires
# --------------------------------------------------------------------------

def tire_roll(
    vehicle: VehicleProfile,
    surface: SurfaceProfile,
    rng: random.Random,
    speed: float = 0.6,
    duration: float = 2.5,
    intensity: float = 0.85,
) -> list[float]:
    """Render a seamless loop of tires rolling over a surface, from recording.

    This was four synthesised layers — tread noise, a block-passing hum, the
    cavity, a rumble — and the model was sound.  It was still replaced, because
    a rolling tire is a car-sized resonator being scrubbed by a whole road, and
    the recording of one is free.

    What survives of the model is the part a recording cannot supply: the
    vehicle.  The bed is one car on one surface, so the tire cavity of whichever
    vehicle is asking is imposed on top of it, and the profile's band and
    damping are what carry a dirt recording to snow.

    The loop is cut last, after every filter, or the loop point would be the one
    moment in the render where the filters were still warming up.
    """

    speed = _check_speed(speed)
    _check_intensity(intensity)
    duration = _check_duration(duration)

    voiced = bank.bed(vehicle, surface, speed)
    cavity = dsp.resonator(
        voiced,
        vehicle.cavity_frequency * (0.88 + 0.24 * speed),
        q=6.0,
        gain=0.30 * (1.0 - 0.5 * surface.damping),
    )
    shaped = dsp.mix(voiced, cavity)
    out = bank.loop(shaped, duration, rng)

    # Speed moves level far more than it moves timbre, and a stationary wheel
    # is silent; without this a crawl and a sprint differ only in brightness.
    level = (0.20 + 0.80 * speed ** 1.15) * (0.55 + 0.52 * intensity)
    return dsp.scale(out, level)


def tire_skid(
    vehicle: VehicleProfile,
    surface: SurfaceProfile,
    rng: random.Random,
    speed: float = 0.7,
    intensity: float = 0.85,
) -> list[float]:
    """Render a one-shot skid: a recorded squeal over synthesised scrub.

    Grip decides what a skid even is, and that is why this cue is a blend rather
    than a sample.  On asphalt it is almost entirely the recording: rubber
    grabbing and letting go is the whole event.  On gravel the rubber never
    grabs long enough to ring, so the squeal recedes and what is left — scrub,
    and stones leaving the tire — is synthesised, because loose material is
    granular, and granular is the thing synthesis is actually better at.

    So the recording carries the squeal and the model carries the surface.
    """

    speed = _check_speed(speed)
    intensity = _check_intensity(intensity)
    duration = 0.85 + 0.75 * intensity + 0.35 * speed

    grip = surface.grip * vehicle.grip
    envelope = dsp.bell_env(
        duration, duration * 0.34, duration * 0.11, duration * 0.42
    )
    voice = bank.take(bank.squeal(vehicle, rng), duration, rng)
    squeal = dsp.mul(voice, envelope)

    # A fixed band, not a swept one: sweeping a filter across an envelope is the
    # recipe for a whoosh, which is what this cue sounded like when the scrub
    # was doing the talking.
    scrub = dsp.mul(
        dsp.bandpass(dsp.noise(duration, rng), 700.0, 2_400.0 + 2_000.0 * speed, 2),
        envelope,
    )
    heat = _low_noise(duration, rng, 420.0, duration * 0.4, attack=0.02)

    out = dsp.mix(
        dsp.scale(squeal, 1.15 * intensity * grip),
        dsp.scale(scrub, 0.20 + 0.42 * (1.0 - surface.grip)),
        dsp.scale(heat, 0.22 * min(vehicle.mass, 2.0)),
    )

    if surface.kind in ("loose", "soft"):
        out = dsp.mix(out, dsp.mul(_grains(
            duration, rng, surface.grain_rate * (0.6 + 0.8 * speed),
            surface.grain_low, surface.grain_high,
            gain=0.62 * surface.spray_gain,
        ), envelope))
    if surface.wetness > 0.0:
        out = dsp.mix(out, dsp.mul(
            _water_film(duration, rng, surface.wetness, speed), envelope
        ))
    return out


def tire_drift(
    vehicle: VehicleProfile,
    surface: SurfaceProfile,
    rng: random.Random,
    speed: float = 0.7,
    duration: float = 2.5,
    intensity: float = 0.85,
) -> list[float]:
    """Render a seamless loop of a sustained held slide.

    A drift is a skid the driver is holding, so it is the same recording asked
    to keep going: a squeal take folded into a loop instead of enveloped into an
    event.  The wheels are still turning underneath, so the roll loop rides with
    it at the same length — one loop point for both, not two beating against
    each other.
    """

    speed = _check_speed(speed)
    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)

    grip = surface.grip * vehicle.grip
    voice = bank.loop(bank.squeal(vehicle, rng), duration, rng)
    scrub = bank.loop(
        dsp.bandpass(
            dsp.noise(duration * 2.0 + 1.0, rng),
            700.0, 2_400.0 + 2_000.0 * speed, 2,
        ),
        duration, rng,
    )

    out = dsp.mix(
        dsp.scale(voice, 1.10 * intensity * grip),
        dsp.scale(scrub, 0.18 + 0.38 * (1.0 - surface.grip)),
    )
    if surface.kind in ("loose", "soft"):
        out = dsp.mix(out, _grains(
            duration, rng, surface.grain_rate * speed,
            surface.grain_low, surface.grain_high,
            gain=0.58 * surface.spray_gain, loop=True,
        ))
    roll = tire_roll(vehicle, surface, rng, speed, duration, intensity)
    return dsp.mix(out, dsp.scale(roll, 0.45))


def surface_spray(
    vehicle: VehicleProfile,
    surface: SurfaceProfile,
    rng: random.Random,
    speed: float = 0.7,
    duration: float = 2.5,
    intensity: float = 0.85,
) -> list[float]:
    """Render a seamless loop of material thrown up into the wheel arches.

    Half recorded, half not, and for once the split is not about quality.  The
    recording supplies the bed — the continuous rush of a surface being churned,
    which is what a dry road has none of and mud has most of.  The discrete
    fraction stays synthesised because it is a shower of individual stones, and
    a grain field is one of the few things here a model does better than a
    microphone: it can be dealt out at whatever rate and pitch the surface
    profile asks for, at any speed, without a second recording.
    """

    speed = _check_speed(speed)
    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)

    thrown = _grains(
        duration, rng, surface.grain_rate * (0.35 + 0.85 * speed),
        surface.grain_low, surface.grain_high,
        gain=surface.spray_gain, spread=1.0 + 0.6 * surface.wetness, loop=True,
    )
    recorded = bank.loop(bank.spray_bed(surface, speed, rng), duration, rng)
    out = dsp.mix(
        thrown,
        dsp.scale(recorded, 0.55 + 0.45 * surface.spray_gain),
    )

    if surface.kind == "loose":
        # Stones do not stop at the tire: they ring the arch they land in.
        out = dsp.mix(out, loopdsp.resonator(
            out, vehicle.body_frequency * 3.1, q=4.5, gain=0.40
        ))
    if surface.name == "mud":
        # Mud is the only surface heavy enough to land rather than tick.
        out = dsp.mix(out, dsp.scale(_grains(
            duration, rng, 14.0 * speed, 60.0, 900.0, gain=0.9, spread=1.8,
            loop=True,
        ), 0.7))

    # Spray is not a wash. Material arrives in surges as the tire loads and
    # unloads and the surface changes under it, and without that the loop is a
    # constant hiss that reads as static however good the bed is. Three LFOs at
    # unrelated rates, so the ear cannot latch onto any one of them -- the loop
    # point stays the only repeat, which is loopdsp's job to hide.
    out = dsp.mul(out, loopdsp.drift(duration, (1.7, 2.9, 4.3), rng, 0.38))

    level = (0.25 + 0.75 * speed) * (0.55 + 0.52 * intensity)
    return dsp.scale(out, level)


# --------------------------------------------------------------------------
# chassis and driveline
# --------------------------------------------------------------------------

def suspension_bump(
    vehicle: VehicleProfile,
    rng: random.Random,
    intensity: float = 0.85,
) -> list[float]:
    """Render one suspension compression and rebound.

    Compression is a damped thump at the spring's frequency; a hard enough hit
    also reaches the bump stop, which is the metallic part.  The rebound
    follows a beat later and is always the quieter of the two.
    """

    intensity = _check_intensity(intensity)
    duration = 0.55 + 0.40 * min(vehicle.mass, 2.5)
    base = vehicle.suspension_frequency

    travel = dsp.mul(
        dsp.sine_track(
            dsp.ramp(duration, base * 1.55, base * 0.72, 0.55),
            rng.uniform(0.0, TAU),
        ),
        dsp.exp_env(duration, 0.055 + 0.055 * vehicle.mass, 0.002),
    )
    damper = _noise_envelope(
        min(duration, 0.22), rng, 180.0, 2_600.0, 0.030, order=2
    )
    bushing = dsp.scale(
        dsp.stick_slip(
            min(duration, 0.3), rng, rate=52.0, freq=base * 4.2, jitter=0.7
        ),
        0.16,
    )

    out = dsp.mix(
        dsp.scale(travel, 0.80 + 0.30 * intensity),
        dsp.scale(damper, 0.30),
        dsp.mul(bushing, dsp.exp_env(min(duration, 0.3), 0.09, 0.004)),
    )

    if intensity > 0.6:
        stop = dsp.modal_ring(
            min(duration, 0.35),
            [(freq, decay * 0.45, gain * 0.5)
             for freq, decay, gain in vehicle.panel_modes[:2]],
            rng,
        )
        dsp.add_at(out, 0.012, stop, (intensity - 0.6) * 1.6)

    rebound_duration = duration * 0.5
    rebound = dsp.mul(
        dsp.sine_track(
            dsp.ramp(rebound_duration, base * 1.05, base * 0.66, 0.6),
            rng.uniform(0.0, TAU),
        ),
        dsp.exp_env(rebound_duration, 0.045 + 0.035 * vehicle.mass, 0.004),
    )
    dsp.add_at(out, 0.11 + 0.05 * vehicle.mass, rebound, 0.42)
    return out


def gear_thump(
    vehicle: VehicleProfile,
    rng: random.Random,
    intensity: float = 0.85,
) -> list[float]:
    """Render a driveline shock: lash taken up, then the shove.

    Two events a few milliseconds apart.  The clack is the slack in the
    driveline closing; the thump is the mass on the other side of it moving.
    """

    intensity = _check_intensity(intensity)
    duration = 0.30 + 0.14 * min(vehicle.mass, 2.5)

    clack = _noise_envelope(
        min(duration, 0.09), rng, 900.0, 6_500.0, 0.0055, order=2
    )
    shock = dsp.mul(
        dsp.sine_track(
            dsp.ramp(
                duration,
                vehicle.body_frequency * 1.35,
                max(28.0, vehicle.body_frequency * 0.58),
                0.5,
            ),
            rng.uniform(0.0, TAU),
        ),
        dsp.exp_env(duration, 0.035 + 0.030 * vehicle.mass, 0.0015),
    )
    ring = dsp.modal_ring(
        min(duration, 0.25),
        [(freq * 1.6, decay * 0.30, gain * 0.22)
         for freq, decay, gain in vehicle.panel_modes[:2]],
        rng,
    )

    out = dsp.mix(
        dsp.scale(shock, (0.72 + 0.34 * intensity) * vehicle.gear_gain),
        dsp.scale(ring, 0.5),
    )
    dsp.add_at(out, 0.0, clack, 0.55 * vehicle.gear_gain)
    return out


def nitro_loop(
    vehicle: VehicleProfile,
    rng: random.Random,
    duration: float = 2.5,
    intensity: float = 0.85,
) -> list[float]:
    """Render a seamless loop of nitrous flowing.

    Gas under pressure through a small orifice: hiss on top, a whistle where
    the orifice resonates, and a roar underneath from what it is feeding.
    """

    intensity = _check_intensity(intensity)
    duration = _check_duration(duration)

    hiss = loopdsp.bandpass(dsp.noise(duration, rng), 2_600.0, 15_000.0, 4)

    # The vibrato is a bipolar whole-cycle sine rather than a `loopdsp.lfo`, and
    # that is the whole trick: a unipolar modulator has a mean, so it would drag
    # the whistle's accumulated phase off a whole number of cycles and land the
    # loop point mid-cycle. This one integrates to exactly zero across the loop,
    # so a quantised carrier stays quantised however far the pitch wanders.
    whistle_hz = loopdsp.quantize(
        2_350.0 / max(0.6, vehicle.mass ** 0.3), duration, 1
    )
    vibrato = loopdsp.sine(5.5, duration, 0.0, minimum=1)
    whistle = dsp.sine_track(
        [whistle_hz * (1.0 + 0.02 * value) for value in vibrato],
        rng.uniform(0.0, TAU),
    )
    harmonic = dsp.sine_track(
        [whistle_hz * 2.0 * (1.0 + 0.02 * value) for value in vibrato],
        rng.uniform(0.0, TAU),
    )
    roar = loopdsp.lowpass(
        dsp.noise(duration, rng), 220.0 + 180.0 * intensity, 4
    )
    body = loopdsp.sine(
        vehicle.body_frequency * 0.72, duration, rng.uniform(0.0, TAU)
    )

    out = dsp.mix(
        dsp.scale(hiss, 0.58 + 0.24 * intensity),
        dsp.scale(whistle, 0.16 * intensity),
        dsp.scale(harmonic, 0.07 * intensity),
        dsp.scale(roar, 0.62),
        dsp.scale(body, 0.20),
    )
    return dsp.mul(out, loopdsp.lfo(3.2, duration, 0.20, rng.uniform(0.0, TAU)))


def boost_burst(
    vehicle: VehicleProfile,
    rng: random.Random,
    intensity: float = 0.85,
) -> list[float]:
    """Render the one-shot that starts a boost: valve, swell, roar.

    Pair it with :func:`nitro_loop` for a held boost — this is the attack the
    loop cannot have and still loop.
    """

    intensity = _check_intensity(intensity)
    duration = 1.25 + 0.35 * intensity

    valve = _noise_envelope(
        min(duration, 0.12), rng, 1_800.0, 14_000.0, 0.012, order=2
    )
    swell = dsp.bell_env(duration, duration * 0.30, duration * 0.14, duration * 0.44)
    centres = [700.0 + 6_500.0 * value ** 1.3 for value in swell]
    whoosh = dsp.mul(
        dsp.svf_bandpass(dsp.noise(duration, rng), centres, q=0.8), swell
    )
    surge = dsp.mul(
        dsp.sine_track(
            dsp.ramp(
                duration,
                vehicle.body_frequency * 0.5,
                vehicle.body_frequency * 1.25,
                0.7,
            ),
            rng.uniform(0.0, TAU),
        ),
        swell,
    )
    tail = dsp.mul(
        dsp.bandpass(dsp.noise(duration, rng), 2_400.0, 15_000.0, 2),
        dsp.exp_env(duration, duration * 0.5, 0.05),
    )

    out = dsp.mix(
        dsp.scale(whoosh, 0.78 + 0.22 * intensity),
        dsp.scale(surge, 0.44 * min(vehicle.mass, 2.2)),
        dsp.scale(tail, 0.30 * intensity),
    )
    dsp.add_at(out, 0.0, valve, 0.62)
    return out


# --------------------------------------------------------------------------
# air
# --------------------------------------------------------------------------

def ramp_takeoff(
    vehicle: VehicleProfile,
    surface: SurfaceProfile,
    rng: random.Random,
    speed: float = 0.8,
    intensity: float = 0.85,
) -> list[float]:
    """Render leaving a ramp: chirp, unload, silence, air.

    The sound of a jump is mostly the sound of things stopping.  Tire noise
    cuts, the suspension tops out against its droop stops, and what is left is
    wind and a chassis that is no longer being held down.
    """

    speed = _check_speed(speed)
    intensity = _check_intensity(intensity)
    duration = 1.15 + 0.45 * speed

    lip = tire_skid(vehicle, surface, rng, speed, min(1.0, intensity * 0.7))
    out = dsp.silence(duration)
    dsp.add_at(out, 0.0, lip, 0.55)

    # Roll noise is cut short rather than faded: the contact patch leaves the
    # ground all at once.
    roll_duration = 0.22
    roll = tire_roll(vehicle, surface, rng, speed, MIN_LOOP, intensity)
    roll = roll[:dsp.samples(roll_duration)]
    dsp.add_at(
        out, 0.0,
        dsp.mul(roll, dsp.ramp(roll_duration, 1.0, 0.0, 2.4)),
        0.70,
    )

    topout = dsp.modal_ring(
        0.4,
        [(freq * 0.9, decay * 0.6, gain * 0.4)
         for freq, decay, gain in vehicle.panel_modes[:3]],
        rng,
    )
    dsp.add_at(out, 0.24, topout, 0.5 + 0.3 * intensity)

    flex = dsp.stick_slip(
        0.55, rng, rate=26.0, freq=vehicle.body_frequency * 2.4, jitter=0.8
    )
    dsp.add_at(
        out, 0.26, dsp.mul(flex, dsp.exp_env(0.55, 0.22, 0.02)),
        0.20 * min(vehicle.mass, 2.0),
    )

    air_duration = duration - 0.3
    air_env = dsp.bell_env(
        air_duration, air_duration * 0.6, air_duration * 0.4, air_duration * 0.5
    )
    air = dsp.mul(
        dsp.bandpass(dsp.noise(air_duration, rng), 160.0, 2_200.0, 2), air_env
    )
    dsp.add_at(out, 0.3, air, 0.34 + 0.30 * speed)
    return out


def ramp_landing(
    vehicle: VehicleProfile,
    surface: SurfaceProfile,
    rng: random.Random,
    speed: float = 0.8,
    intensity: float = 0.85,
) -> list[float]:
    """Render a touchdown: contact, travel, boom, and whatever it kicked up.

    The contact layer is the surface's; the travel and boom under it are the
    vehicle's.  Landing a monster truck in mud and a kart on concrete are the
    same three events with nothing in common.
    """

    speed = _check_speed(speed)
    intensity = _check_intensity(intensity)
    duration = 1.30 + 0.50 * min(vehicle.mass, 2.4)
    out = dsp.silence(duration)

    if surface.kind == "hard":
        contact = _noise_envelope(
            0.3, rng, 260.0, 7_000.0, 0.020, order=2
        )
    elif surface.kind == "loose":
        contact = _low_noise(0.4, rng, 1_600.0, 0.055, attack=0.001)
    else:
        contact = _low_noise(0.45, rng, 700.0, 0.075, attack=0.002)
    dsp.add_at(out, 0.0, contact, 0.72 + 0.28 * intensity)

    slap = dsp.mul(
        dsp.sine_track(
            dsp.ramp(0.35, vehicle.cavity_frequency * 1.2,
                     vehicle.cavity_frequency * 0.7, 0.6),
            rng.uniform(0.0, TAU),
        ),
        dsp.exp_env(0.35, 0.045, 0.001),
    )
    dsp.add_at(out, 0.0, slap, 0.40 * (1.0 - 0.6 * surface.damping))

    bump = suspension_bump(vehicle, rng, intensity)
    dsp.add_at(out, 0.008, bump, 0.95 + 0.25 * intensity)

    boom = dsp.mul(
        dsp.sine_track(
            dsp.ramp(
                duration,
                vehicle.body_frequency * 1.15,
                max(26.0, vehicle.body_frequency * 0.55),
                0.6,
            ),
            rng.uniform(0.0, TAU),
        ),
        dsp.exp_env(duration, 0.10 + 0.09 * vehicle.mass, 0.004),
    )
    dsp.add_at(out, 0.006, boom, 0.55 * min(vehicle.mass, 2.4))

    if surface.kind == "hard" and surface.grip * vehicle.grip > 0.2:
        chirp = tire_skid(vehicle, surface, rng, speed, 0.45)
        dsp.add_at(out, 0.01, chirp, 0.34)
    else:
        kicked = _grains(
            0.75, rng, surface.grain_rate * 0.8,
            surface.grain_low, surface.grain_high,
            gain=0.85 * surface.spray_gain,
        )
        dsp.add_at(
            out, 0.02, dsp.mul(kicked, dsp.exp_env(0.75, 0.22, 0.006)), 0.8
        )
    if surface.wetness > 0.0:
        splash = _water_film(0.6, rng, surface.wetness, max(speed, 0.5))
        dsp.add_at(
            out, 0.0, dsp.mul(splash, dsp.exp_env(0.6, 0.18, 0.003)), 1.1
        )
    return out


# --------------------------------------------------------------------------
# impacts
# --------------------------------------------------------------------------

def body_collision(
    vehicle: VehicleProfile,
    rng: random.Random,
    intensity: float = 0.85,
) -> list[float]:
    """Render a body panel taking a hit.

    Crunch, then the panel ringing, then the structure behind it booming.  The
    ring is inharmonic and drops with mass, which is most of what tells a kart
    apart from a truck here.
    """

    # Item 10 centralises this event. Keep the public function so existing
    # vehicle packs and callers do not change API; the implementation and the
    # material decisions live in crash_generator.
    from ..crash_generator import engine as crashes

    signal, _ = crashes.collision(
        crashes.STRUCTURES[vehicle.name], crashes.SURFACES["asphalt"],
        crashes.rng_from_random(rng), intensity,
    )
    return signal.tolist()

def crash_debris(
    vehicle: VehicleProfile,
    surface: SurfaceProfile,
    rng: random.Random,
    intensity: float = 0.85,
) -> list[float]:
    """Render parts leaving the vehicle and finding the ground.

    Unlike a shrapnel shower this decelerates twice: pieces bounce, then skid.
    Hard surfaces let them ring and slide; mud takes the first bounce and keeps
    it.
    """

    from ..crash_generator import engine as crashes

    signal, _ = crashes.debris(
        crashes.STRUCTURES[vehicle.name], crashes.SURFACES[surface.name],
        crashes.rng_from_random(rng), intensity,
    )
    return signal.tolist()

def vehicle_crash(
    vehicle: VehicleProfile,
    surface: SurfaceProfile,
    rng: random.Random,
    intensity: float = 0.85,
) -> list[float]:
    """Render a full crash: the hit, what it hits next, and the aftermath.

    One collision is an accident; a crash is a sequence.  Secondary impacts
    arrive as the vehicle rebounds and settles, each quieter than the last, and
    the debris outlives all of them.
    """

    from ..crash_generator import engine as crashes

    signal, _ = crashes.crash(
        crashes.STRUCTURES[vehicle.name], crashes.SURFACES[surface.name],
        crashes.rng_from_random(rng), intensity,
    )
    return signal.tolist()

# --------------------------------------------------------------------------
# lookups and io
# --------------------------------------------------------------------------

def vehicle_named(name: str) -> VehicleProfile:
    try:
        return VEHICLES[name]
    except KeyError:
        raise ValueError(
            f"unknown vehicle {name!r}; expected one of {sorted(VEHICLES)}"
        ) from None


def surface_named(name: str) -> SurfaceProfile:
    try:
        return SURFACES[name]
    except KeyError:
        raise ValueError(
            f"unknown surface {name!r}; expected one of {sorted(SURFACES)}"
        ) from None
