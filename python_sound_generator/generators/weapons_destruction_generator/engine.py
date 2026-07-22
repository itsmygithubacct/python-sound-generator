"""Procedural ranged-weapon and destruction synthesis.

This module deliberately shares the standard-library DSP core used by
``combat_generators``.  The sounds are layered rather than sampled: a firearm
has a muzzle crack, pressure body, report tail, and mechanism; an explosion has
a shock, blast, sub-pressure wave, rumble, and fragments; an impact is shaped
by both the projectile and the target material.

All renderers return unmastered mono floating-point PCM.  The cue registry owns
mastering levels so that every public render follows the same contract.
"""

from __future__ import annotations

from ..._audio.dsp import write_wav as write_wav

import math
import random
from dataclasses import dataclass

from ..._audio import dsp


SR = dsp.SR


@dataclass(frozen=True)
class FirearmProfile:
    """Acoustic dimensions of one conventional ranged weapon."""

    name: str
    label: str
    description: str
    category: str
    duration: float
    scale: float
    crack_low: float
    crack_high: float
    body_frequency: float
    blast_cutoff: float
    blast_decay: float
    tail_decay: float
    mechanical_gain: float
    rounds_per_second: float


@dataclass(frozen=True)
class ProjectileProfile:
    """Launch, flight, and impact character of a projectile."""

    name: str
    label: str
    description: str
    kind: str
    flyby_duration: float
    launch_duration: float
    pitch: float
    brightness: float
    weight: float
    motor_gain: float = 0.0


@dataclass(frozen=True)
class ExplosiveProfile:
    """Scale and spectral balance of one explosive event."""

    name: str
    label: str
    description: str
    duration: float
    scale: float
    body_frequency: float
    blast_cutoff: float
    blast_decay: float
    rumble_decay: float
    fragment_count: int
    energy: bool = False


Modes = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ImpactMaterial:
    """Layer parameters for a projectile striking a target material."""

    name: str
    label: str
    description: str
    kind: str
    duration: float
    contact_low: float
    contact_high: float
    contact_decay: float
    body_frequency: float
    body_decay: float
    modes: Modes = ()
    particle_gain: float = 0.0


FIREARMS: dict[str, FirearmProfile] = {
    "pistol": FirearmProfile(
        "pistol", "Pistol", "compact semi-automatic handgun", "handgun",
        0.58, 0.52, 1_500.0, 12_000.0, 142.0, 3_400.0, 0.028, 0.16,
        0.22, 4.5,
    ),
    "revolver": FirearmProfile(
        "revolver", "Revolver", "large-frame revolver with a hard report",
        "handgun", 0.72, 0.68, 1_250.0, 11_500.0, 118.0, 3_000.0,
        0.037, 0.23, 0.16, 3.2,
    ),
    "smg": FirearmProfile(
        "smg", "Submachine gun", "short, light automatic report", "automatic",
        0.50, 0.44, 1_750.0, 13_500.0, 156.0, 3_800.0, 0.023, 0.13,
        0.28, 13.0,
    ),
    "rifle": FirearmProfile(
        "rifle", "Rifle", "supersonic rifle crack with a firm pressure body",
        "long gun", 0.86, 0.80, 1_900.0, 15_000.0, 104.0, 3_800.0,
        0.042, 0.31, 0.18, 7.5,
    ),
    "shotgun": FirearmProfile(
        "shotgun", "Shotgun", "broad, low shotgun blast", "long gun",
        1.00, 0.96, 800.0, 10_000.0, 82.0, 2_500.0, 0.058, 0.39,
        0.24, 2.0,
    ),
    "machine_gun": FirearmProfile(
        "machine_gun", "Machine gun", "full-power automatic weapon",
        "automatic", 0.90, 0.88, 1_600.0, 14_500.0, 96.0, 3_500.0,
        0.046, 0.32, 0.25, 10.5,
    ),
    "autocannon": FirearmProfile(
        "autocannon", "Autocannon", "rapid heavy-calibre cannon",
        "ordnance", 1.28, 1.25, 900.0, 12_000.0, 67.0, 2_600.0,
        0.078, 0.58, 0.18, 3.4,
    ),
    "cannon": FirearmProfile(
        "cannon", "Cannon", "large direct-fire cannon", "ordnance",
        1.80, 1.72, 500.0, 10_000.0, 51.0, 1_900.0, 0.12, 0.90,
        0.10, 0.75,
    ),
    "artillery": FirearmProfile(
        "artillery", "Artillery", "heavy indirect-fire artillery piece",
        "ordnance", 2.45, 2.15, 330.0, 8_500.0, 39.0, 1_400.0,
        0.18, 1.30, 0.08, 0.32,
    ),
}


PROJECTILES: dict[str, ProjectileProfile] = {
    "bullet": ProjectileProfile(
        "bullet", "Bullet", "supersonic small-arms projectile", "ballistic",
        0.34, 0.28, 2_250.0, 1.25, 0.42,
    ),
    "shell": ProjectileProfile(
        "shell", "Shell", "heavy cannon or artillery shell", "ballistic",
        0.72, 0.70, 920.0, 0.82, 1.45,
    ),
    "rocket": ProjectileProfile(
        "rocket", "Rocket", "unguided rocket with a rough motor", "motor",
        1.08, 0.92, 255.0, 0.72, 1.42, 1.0,
    ),
    "missile": ProjectileProfile(
        "missile", "Missile", "guided missile with a smoother motor", "motor",
        1.22, 1.05, 390.0, 0.90, 1.55, 0.78,
    ),
    "grenade": ProjectileProfile(
        "grenade", "Grenade", "hand-thrown fragmentation grenade", "thrown",
        0.62, 0.48, 760.0, 0.55, 0.74,
    ),
    "bomb": ProjectileProfile(
        "bomb", "Bomb", "heavy free-falling bomb", "falling",
        1.15, 0.72, 430.0, 0.62, 2.25,
    ),
    "laser": ProjectileProfile(
        "laser", "Laser bolt", "fast coherent-energy bolt", "energy",
        0.38, 0.48, 4_200.0, 1.45, 0.38,
    ),
    "plasma": ProjectileProfile(
        "plasma", "Plasma bolt", "slower unstable-energy projectile", "energy",
        0.82, 0.88, 720.0, 1.05, 1.16,
    ),
}


EXPLOSIVES: dict[str, ExplosiveProfile] = {
    "grenade": ExplosiveProfile(
        "grenade", "Grenade", "compact fragmentation explosion",
        1.05, 0.68, 72.0, 2_700.0, 0.075, 0.52, 10,
    ),
    "rocket": ExplosiveProfile(
        "rocket", "Rocket warhead", "medium shaped-charge explosion",
        1.48, 0.98, 58.0, 2_150.0, 0.11, 0.78, 14,
    ),
    "bomb": ExplosiveProfile(
        "bomb", "Bomb", "large aerial bomb and long pressure tail",
        2.75, 1.75, 36.0, 1_350.0, 0.19, 1.48, 25,
    ),
    "artillery": ExplosiveProfile(
        "artillery", "Artillery impact", "heavy shell detonation",
        2.25, 1.48, 42.0, 1_550.0, 0.16, 1.12, 21,
    ),
    "plasma": ExplosiveProfile(
        "plasma", "Plasma burst", "unstable energy discharge",
        1.68, 1.08, 84.0, 3_600.0, 0.12, 0.72, 8, True,
    ),
}


MATERIALS: dict[str, ImpactMaterial] = {
    "metal": ImpactMaterial(
        "metal", "Metal", "plate, machinery, or vehicle body", "hard",
        0.72, 1_900.0, 14_000.0, 0.006, 116.0, 0.075,
        ((1_180.0, 0.16, 0.34), (2_030.0, 0.12, 0.25),
         (3_470.0, 0.085, 0.17), (5_180.0, 0.050, 0.10)), 0.10,
    ),
    "wood": ImpactMaterial(
        "wood", "Wood", "timber, crates, or wooden structures", "hard",
        0.62, 480.0, 7_500.0, 0.012, 92.0, 0.090,
        ((265.0, 0.10, 0.30), (438.0, 0.070, 0.18),
         (730.0, 0.045, 0.10)), 0.20,
    ),
    "stone": ImpactMaterial(
        "stone", "Stone", "masonry, concrete, or rock", "brittle",
        0.66, 900.0, 10_500.0, 0.008, 76.0, 0.070,
        ((610.0, 0.055, 0.20), (1_370.0, 0.038, 0.14),
         (2_480.0, 0.025, 0.08)), 0.34,
    ),
    "dirt": ImpactMaterial(
        "dirt", "Dirt", "soil, sand, or loose ground", "loose",
        0.58, 120.0, 3_800.0, 0.024, 58.0, 0.105, (), 0.48,
    ),
    "water": ImpactMaterial(
        "water", "Water", "open water or a deep puddle", "liquid",
        0.88, 350.0, 12_000.0, 0.040, 72.0, 0.16, (), 0.56,
    ),
    "flesh": ImpactMaterial(
        "flesh", "Flesh", "unarmoured organic target", "soft",
        0.52, 180.0, 4_600.0, 0.014, 64.0, 0.085, (), 0.18,
    ),
}


def _check_intensity(intensity: float) -> float:
    if not 0.05 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    return intensity


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


def fire_weapon(
    profile: FirearmProfile,
    rng: random.Random,
    intensity: float = 0.85,
) -> list[float]:
    """Render one conventional muzzle event."""

    intensity = _check_intensity(intensity)
    duration = profile.duration
    scale = profile.scale

    crack_decay = 0.0022 + 0.0018 * min(scale, 2.0)
    crack = _noise_envelope(
        duration, rng, profile.crack_low, profile.crack_high,
        crack_decay, order=4,
    )
    blast = _low_noise(
        duration, rng, profile.blast_cutoff, profile.blast_decay,
        attack=0.0005,
    )
    pressure_freqs = dsp.ramp(
        duration,
        profile.body_frequency * (1.18 + 0.04 * intensity),
        max(24.0, profile.body_frequency * 0.62),
        0.68,
    )
    pressure = dsp.mul(
        dsp.sine_track(pressure_freqs),
        dsp.exp_env(duration, profile.blast_decay * 1.7, 0.0008),
    )
    tail = _noise_envelope(
        duration, rng, 32.0, max(420.0, profile.blast_cutoff * 0.62),
        profile.tail_decay, attack=0.003, order=2,
    )

    out = dsp.mix(
        dsp.scale(crack, 0.48 + 0.34 * profile.crack_high / 15_000.0),
        dsp.scale(blast, 0.78 + 0.28 * scale),
        dsp.scale(pressure, 0.46 + 0.31 * scale),
        dsp.scale(tail, 0.24 + 0.34 * min(scale, 2.0)),
    )

    # A shotgun report is many near-simultaneous pressure fronts.  The few-ms
    # spread makes it broad without turning it into a burst of separate shots.
    if profile.name == "shotgun":
        for delay in (0.0016, 0.0034, 0.0051):
            pellet = _noise_envelope(
                0.055, rng, 650.0, 8_000.0, 0.009, order=2,
            )
            dsp.add_at(out, delay, pellet, rng.uniform(0.13, 0.22))

    mechanism_duration = min(0.13, duration)
    mechanism = _noise_envelope(
        mechanism_duration, rng, 1_100.0, 9_000.0, 0.010, order=2,
    )
    mechanism_delay = 0.030 + 0.018 * min(1.0, scale)
    dsp.add_at(out, mechanism_delay, mechanism, profile.mechanical_gain)

    # Intensity changes more than level: quiet reports lose crack and retain
    # proportionally more body, while mastering later preserves headroom.
    body_mix = 0.48 + 0.52 * intensity
    return [sample * body_mix for sample in out]


def fire_burst(
    profile: FirearmProfile,
    rng: random.Random,
    shots: int = 4,
    intensity: float = 0.85,
) -> list[float]:
    """Render a deterministic multi-shot sequence at the weapon's cadence."""

    if not 2 <= shots <= 12:
        raise ValueError("shots must be between 2 and 12")
    _check_intensity(intensity)
    interval = 1.0 / profile.rounds_per_second
    total = interval * (shots - 1) + profile.duration
    out = dsp.silence(total)
    for index in range(shots):
        timing = 0.0 if index == 0 else rng.uniform(-0.012, 0.012) * interval
        shot = fire_weapon(profile, rng, intensity)
        dsp.add_at(out, max(0.0, index * interval + timing), shot,
                   rng.uniform(0.92, 1.06))
    return out


def fire_laser(rng: random.Random, intensity: float = 0.85) -> list[float]:
    """Fast descending coherent-energy report."""

    intensity = _check_intensity(intensity)
    duration = 0.48
    detune = rng.uniform(0.96, 1.04)
    frequencies = dsp.ramp(duration, 6_200.0 * detune, 720.0 * detune, 0.70)
    carrier = dsp.sine_track(frequencies)
    harmonic = dsp.sine_track([f * 1.997 for f in frequencies], rng.random())
    envelope = dsp.bell_env(duration, 0.040, 0.018, 0.13)
    transient = _noise_envelope(
        duration, rng, 2_600.0, 15_000.0, 0.010, order=4,
    )
    return dsp.mix(
        dsp.mul(carrier, envelope),
        dsp.scale(dsp.mul(harmonic, envelope), 0.32 + 0.18 * intensity),
        dsp.scale(transient, 0.34),
    )


def fire_plasma(rng: random.Random, intensity: float = 0.85) -> list[float]:
    """Charged low-mid energy launch with an unstable discharge."""

    intensity = _check_intensity(intensity)
    duration = 0.92
    charge_env = dsp.bell_env(duration, 0.14, 0.090, 0.075)
    charge_noise = dsp.bandpass(dsp.noise(duration, rng), 500.0, 5_500.0, 4)
    charge = dsp.mul(charge_noise, charge_env)

    discharge = dsp.silence(duration)
    release_duration = 0.70
    frequencies = dsp.ramp(release_duration, 610.0, 74.0, 0.58)
    tone = dsp.mul(
        dsp.sine_track(frequencies, rng.uniform(0.0, dsp.TAU)),
        dsp.exp_env(release_duration, 0.20, 0.002),
    )
    sputter = _noise_envelope(
        release_duration, rng, 160.0, 4_800.0, 0.16, attack=0.001, order=2,
    )
    dsp.add_at(discharge, 0.13, tone, 0.90 + 0.25 * intensity)
    dsp.add_at(discharge, 0.13, sputter, 0.48)
    return dsp.mix(dsp.scale(charge, 0.34), discharge)


def launch_projectile(
    projectile: ProjectileProfile,
    rng: random.Random,
    intensity: float = 0.85,
) -> list[float]:
    """Render separation and propulsion at the projectile's launch point."""

    intensity = _check_intensity(intensity)
    if projectile.name == "laser":
        return fire_laser(rng, intensity)
    if projectile.name == "plasma":
        return fire_plasma(rng, intensity)

    duration = projectile.launch_duration
    ignition = _noise_envelope(
        duration, rng, 180.0, 9_500.0,
        0.030 + 0.018 * projectile.weight, order=4,
    )
    kick_freqs = dsp.ramp(
        duration, max(65.0, projectile.pitch * 0.30),
        max(28.0, projectile.pitch * 0.10), 0.65,
    )
    kick = dsp.mul(
        dsp.sine_track(kick_freqs),
        dsp.exp_env(duration, 0.055 + 0.035 * projectile.weight, 0.001),
    )
    out = dsp.mix(
        dsp.scale(ignition, 0.58 + 0.20 * intensity),
        dsp.scale(kick, 0.42 + 0.24 * projectile.weight),
    )

    if projectile.kind == "motor":
        motor_noise = dsp.bandpass(
            dsp.noise(duration, rng), 55.0, 3_600.0 * projectile.brightness, 4,
        )
        motor_env = dsp.bell_env(
            duration, duration * 0.34, duration * 0.18, duration * 0.52,
        )
        motor = dsp.mul(motor_noise, motor_env)
        whine = dsp.mul(
            dsp.sine_track(dsp.ramp(duration, projectile.pitch * 0.72,
                                    projectile.pitch * 1.18, 0.82)),
            motor_env,
        )
        out = dsp.mix(
            out,
            dsp.scale(motor, 0.58 * projectile.motor_gain),
            dsp.scale(whine, 0.24 * projectile.motor_gain),
        )
    else:
        separation = _noise_envelope(
            min(duration, 0.25), rng, 900.0, 8_000.0, 0.028, order=2,
        )
        dsp.add_at(out, 0.018, separation, 0.22)
    return out


def projectile_flyby(
    projectile: ProjectileProfile,
    rng: random.Random,
    intensity: float = 0.85,
) -> list[float]:
    """Render a close pass with an audible Doppler descent."""

    intensity = _check_intensity(intensity)
    duration = projectile.flyby_duration
    peak = duration * 0.48
    envelope = dsp.bell_env(duration, peak, duration * 0.18, duration * 0.14)
    centre = [
        180.0 + projectile.brightness * (850.0 + 4_300.0 * value ** 1.5)
        for value in envelope
    ]
    air = dsp.mul(
        dsp.svf_bandpass(dsp.noise(duration, rng), centre, q=1.15),
        envelope,
    )
    doppler = dsp.mul(
        dsp.sine_track(dsp.ramp(duration, projectile.pitch * 1.65,
                                projectile.pitch * 0.58, 0.78),
                       rng.uniform(0.0, dsp.TAU)),
        [value ** 1.35 for value in envelope],
    )
    out = dsp.mix(
        dsp.scale(air, 0.78 + 0.18 * intensity),
        dsp.scale(doppler, 0.12 + 0.18 * min(projectile.weight, 1.5)),
    )

    if projectile.kind == "motor":
        motor = dsp.mul(
            dsp.bandpass(dsp.noise(duration, rng), 45.0, 1_300.0, 4),
            envelope,
        )
        out = dsp.mix(out, dsp.scale(motor, 0.58 * projectile.motor_gain))
    elif projectile.kind == "energy":
        electric = dsp.mul(
            dsp.sine_track(dsp.ramp(duration, projectile.pitch * 2.1,
                                    projectile.pitch * 0.92, 0.64)),
            [value ** 2.0 for value in envelope],
        )
        out = dsp.mix(out, dsp.scale(electric, 0.34))
    return out



def ricochet(
    projectile: ProjectileProfile,
    material: ImpactMaterial,
    rng: random.Random,
    intensity: float = 0.85,
) -> list[float]:
    """Render contact, deflection ping, and a receding flight tail."""

    intensity = _check_intensity(intensity)
    duration = 0.58 + 0.10 * min(projectile.weight, 1.5)
    hard_gain = 1.0 if material.kind in ("hard", "brittle") else 0.34
    snap = _noise_envelope(
        duration, rng, max(650.0, material.contact_low),
        min(16_000.0, max(4_000.0, material.contact_high)), 0.0055,
        order=4,
    )
    base = (material.modes[1][0] if len(material.modes) > 1 else 1_450.0)
    base *= 1.0 / math.sqrt(max(0.35, projectile.weight))
    ping = dsp.mul(
        dsp.sine_track(dsp.ramp(duration, base * 1.55, base * 0.74, 0.62),
                       rng.uniform(0.0, dsp.TAU)),
        dsp.exp_env(duration, 0.12 * hard_gain + 0.025, 0.001),
    )
    tail_duration = min(0.34, duration - 0.045)
    tail_env = dsp.bell_env(tail_duration, tail_duration * 0.34,
                            tail_duration * 0.20, tail_duration * 0.34)
    centres = [700.0 + 3_600.0 * value for value in tail_env]
    tail = dsp.mul(
        dsp.svf_bandpass(dsp.noise(tail_duration, rng), centres, q=1.6),
        tail_env,
    )
    out = dsp.mix(dsp.scale(snap, 0.60), dsp.scale(ping, 0.72 * hard_gain))
    dsp.add_at(out, 0.042, tail, 0.36 + 0.12 * intensity)
    return out



def explosion(
    profile: ExplosiveProfile,
    rng: random.Random,
    intensity: float = 0.85,
) -> list[float]:
    """Render a layered explosion from shock through debris tail."""

    intensity = _check_intensity(intensity)
    duration = profile.duration
    shock = _noise_envelope(
        duration, rng, 650.0, min(16_000.0, profile.blast_cutoff * 6.0),
        0.004 + 0.002 * min(profile.scale, 1.8), order=4,
    )
    blast = _low_noise(
        duration, rng, profile.blast_cutoff, profile.blast_decay,
        attack=0.0008,
    )
    pressure = dsp.mul(
        dsp.sine_track(dsp.ramp(duration, profile.body_frequency * 1.25,
                                max(20.0, profile.body_frequency * 0.44), 0.58),
                       rng.uniform(0.0, dsp.TAU)),
        dsp.exp_env(duration, profile.blast_decay * 1.65, 0.001),
    )
    rumble = _low_noise(
        duration, rng, 520.0, profile.rumble_decay, attack=0.012,
    )
    air = _noise_envelope(
        duration, rng, 120.0, 2_200.0, profile.blast_decay * 2.1,
        attack=0.006, order=2,
    )
    out = dsp.mix(
        dsp.scale(shock, 0.48 + 0.16 * intensity),
        dsp.scale(blast, 0.88 + 0.24 * profile.scale),
        dsp.scale(pressure, 0.62 + 0.28 * profile.scale),
        dsp.scale(rumble, 0.46 + 0.30 * min(profile.scale, 1.8)),
        dsp.scale(air, 0.24),
    )

    fragments = [0.0] * len(out)
    for _ in range(profile.fragment_count):
        onset = rng.random() ** 0.72 * duration * 0.58
        index = min(len(fragments) - 1, round(onset * SR))
        fragments[index] += rng.uniform(-1.0, 1.0) * (1.0 - onset / duration)
    fragments = dsp.bandpass(fragments, 850.0, 10_000.0, 2)
    out = dsp.mix(out, dsp.scale(fragments, 0.24))

    if profile.energy:
        frequencies = dsp.ramp(duration, 2_800.0, 95.0, 0.48)
        energy = dsp.mul(
            dsp.sine_track(frequencies, rng.uniform(0.0, dsp.TAU)),
            dsp.exp_env(duration, 0.32, 0.002),
        )
        arc = dsp.modal_ring(
            duration,
            [(1_180.0, 0.22, 0.22), (2_740.0, 0.15, 0.14),
             (4_920.0, 0.09, 0.08)],
            rng,
        )
        out = dsp.mix(out, dsp.scale(energy, 0.42), arc)
    return out


def firearm_named(name: str) -> FirearmProfile:
    try:
        return FIREARMS[name]
    except KeyError:
        raise ValueError(
            f"unknown weapon {name!r}; expected one of {sorted(FIREARMS)}"
        ) from None


def projectile_named(name: str) -> ProjectileProfile:
    try:
        return PROJECTILES[name]
    except KeyError:
        raise ValueError(
            f"unknown projectile {name!r}; expected one of {sorted(PROJECTILES)}"
        ) from None


def explosive_named(name: str) -> ExplosiveProfile:
    try:
        return EXPLOSIVES[name]
    except KeyError:
        raise ValueError(
            f"unknown explosive {name!r}; expected one of {sorted(EXPLOSIVES)}"
        ) from None


def material_named(name: str) -> ImpactMaterial:
    try:
        return MATERIALS[name]
    except KeyError:
        raise ValueError(
            f"unknown material {name!r}; expected one of {sorted(MATERIALS)}"
        ) from None
