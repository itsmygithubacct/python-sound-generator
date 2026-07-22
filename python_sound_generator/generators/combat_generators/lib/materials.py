"""Materials a weapon can be made of, and targets a strike can land on.

Two separate ideas live here:

- `Material` is what something is made of. Density and Young's modulus are real
  handbook figures, and they do real work: the bar wave speed sqrt(E/rho) sets
  how high a thing rings, so bronze lands lower than steel for free, without a
  tuning table. `q_factor` is the damping and is a listening judgement, not a
  measurement.
- `Target` is what the blade hits, described as a stack of layers. A strike is
  never one sound: it is a contact snap, a broadband impact, a ring, a tail, a
  low body knock, and sometimes a second chink as the two part.

The layer order in every Target is fixed (snap, impact, ring, tail, body,
chink) because lib/strikes.py renders them in that order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    description: str
    density: float      # kg/m^3
    youngs: float       # Pa
    q_factor: float     # damping; higher rings longer. Tuned by ear.

    @property
    def wave_speed(self) -> float:
        """Bar wave speed sqrt(E/rho), in m/s.

        This is the physically honest part of the model: it is why bronze
        (~3540 m/s) rings lower than steel (~5050 m/s) at the same geometry.
        """
        return math.sqrt(self.youngs / self.density)


STEEL = Material("steel", "hardened blade steel", 7850.0, 200e9, 1.00)
IRON = Material("iron", "soft wrought iron", 7870.0, 190e9, 0.72)
BRONZE = Material("bronze", "cast bronze, older blades", 8800.0, 110e9, 0.85)
# What a fist, a shin or an elbow strikes with. Density and modulus are
# handbook cortical bone, so the wave speed falls out at ~3080 m/s -- under
# bronze, and that is the whole reason a punch rings what it hits lower than
# any blade does, without a table saying so.
#
# The q_factor is the one number here that is not a measurement, and it is
# further from one than the others. Bone in a living body is wrapped in tissue
# that kills a ring outright, so a measured figure would sit near stone's 0.10.
# But q_factor also sets how long the whole event runs (see ring_time_scale in
# lib/weapons.py), and at 0.10 a punch collapses to a 30 ms click with no thump
# left. 0.55 is tuned to leave the thump in. That coupling is a limitation of
# the model, not a property of bone.
BONE = Material("bone", "knuckle, shin, elbow: bone under thin tissue",
                1900.0, 18e9, 0.55)
WOOD = Material("wood", "seasoned oak", 750.0, 11e9, 0.22)
STONE = Material("stone", "dressed stone", 2600.0, 50e9, 0.10)

MATERIALS = {m.name: m for m in (STEEL, IRON, BRONZE, BONE, WOOD, STONE)}


Modes = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class Target:
    """What a strike lands on, as a stack of layers.

    Frequencies here are for the reference weapon (see lib/weapons.py); the
    weapon scales them. Layers with a zero gain are skipped.
    """
    name: str
    description: str
    duration: float

    # contact snap: the first 2-5 ms, bright filtered noise
    snap_hz: float = 3200.0
    snap_decay: float = 0.0045
    snap_gain: float = 0.50

    # impact burst: broadband, around the contact
    impact_low: float = 1600.0
    impact_high: float = 9500.0
    impact_decay: float = 0.030
    impact_gain: float = 0.85

    # ring: inharmonic modes. Empty for things that do not ring.
    modes: Modes = ()

    # tail: what happens just after contact
    tail_kind: str = "none"          # scrape | wobble | rattle | none
    tail_low: float = 2400.0
    tail_high: float = 7600.0
    tail_at: float = 0.045           # scrape: centre of the burst
    tail_width: float = 0.030        # scrape: width of the burst
    tail_decay: float = 0.10         # wobble: decay
    tail_rate: float = 90.0          # rattle: slips per second
    tail_gain: float = 0.0

    # body: low knock through guard, grip, or the struck object
    body_hz: float = 168.0
    body_decay: float = 0.045
    body_gain: float = 0.32

    # chink: the lighter second contact as the two part
    chink_modes: Modes = ()
    chink_at: float = 0.072
    chink_duration: float = 0.16
    chink_gain: float = 0.55

    # output level. Impacts sit louder than swings so a mix reads correctly.
    peak_db: float = -9.0

    scales_with_weapon: bool = True


# Blade on blade. These numbers are the ones that survived listening tests:
# the modes are inharmonic on purpose and all die inside ~100 ms. Ring them
# longer, or tune them to a harmonic series, and the hit becomes a church bell.
BLADE = Target(
    name="blade",
    description="another blade, edge on edge",
    duration=0.44,
    modes=(
        (2210.0, 0.085, 0.30),
        (2967.0, 0.070, 0.34),
        (3743.0, 0.060, 0.26),
        (4638.0, 0.048, 0.20),
        (5512.0, 0.038, 0.15),
        (6890.0, 0.028, 0.10),
        (1493.0, 0.055, 0.14),
    ),
    tail_kind="scrape",
    tail_gain=0.28,
    chink_modes=(
        (2530.0, 0.045, 0.30),
        (3410.0, 0.035, 0.22),
        (4470.0, 0.028, 0.15),
    ),
)

# Blade on plate armour: the plate eats the top end and flexes instead.
PLATE = Target(
    name="plate",
    description="plate armour, blade skids off steel",
    duration=0.40,
    snap_hz=1800.0, snap_decay=0.006, snap_gain=0.30,
    impact_low=500.0, impact_high=4200.0, impact_decay=0.045, impact_gain=0.80,
    modes=(
        (742.0, 0.075, 0.30),
        (1046.0, 0.062, 0.24),
        (1523.0, 0.050, 0.18),
        (2130.0, 0.038, 0.12),
        (2870.0, 0.028, 0.08),
    ),
    tail_kind="wobble",
    tail_low=900.0, tail_decay=0.10, tail_gain=0.22,
    body_hz=96.0, body_decay=0.070, body_gain=0.50,
    peak_db=-9.5,
)

# Mail is hundreds of loose rings: almost no coherent ring, a lot of jingle.
MAIL = Target(
    name="mail",
    description="chainmail, blade bites and the links jingle",
    duration=0.46,
    snap_hz=2400.0, snap_decay=0.005, snap_gain=0.34,
    impact_low=900.0, impact_high=6500.0, impact_decay=0.022, impact_gain=0.60,
    modes=(
        (3120.0, 0.020, 0.12),
        (4310.0, 0.016, 0.10),
        (5680.0, 0.012, 0.07),
    ),
    tail_kind="rattle",
    tail_low=2600.0, tail_high=9000.0, tail_rate=150.0, tail_gain=0.55,
    body_hz=140.0, body_decay=0.030, body_gain=0.20,
    peak_db=-10.0,
)

# A shield is a wooden disc with a steel boss: thock plus a little ring.
SHIELD = Target(
    name="shield",
    description="wooden shield, blade bites the face",
    duration=0.42,
    snap_hz=1500.0, snap_decay=0.004, snap_gain=0.32,
    impact_low=280.0, impact_high=3400.0, impact_decay=0.028, impact_gain=0.85,
    modes=(
        (386.0, 0.045, 0.26),
        (612.0, 0.032, 0.16),
        (988.0, 0.022, 0.10),
        (1840.0, 0.014, 0.06),
    ),
    tail_kind="wobble",
    tail_low=700.0, tail_decay=0.055, tail_gain=0.20,
    body_hz=88.0, body_decay=0.090, body_gain=0.62,
    peak_db=-9.5,
)

# Wood: the blade sticks. Mid thunk, no metal at all.
WOOD_TARGET = Target(
    name="wood",
    description="timber, post or shaft; the blade bites and holds",
    duration=0.34,
    snap_hz=1200.0, snap_decay=0.0035, snap_gain=0.26,
    impact_low=220.0, impact_high=2800.0, impact_decay=0.024, impact_gain=0.90,
    modes=(
        (296.0, 0.038, 0.24),
        (452.0, 0.026, 0.14),
        (764.0, 0.018, 0.08),
    ),
    tail_kind="wobble",
    tail_low=520.0, tail_decay=0.040, tail_gain=0.16,
    body_hz=104.0, body_decay=0.060, body_gain=0.55,
    peak_db=-9.5,
)

# Stone: a bright crack, grit, and no ring whatsoever.
STONE_TARGET = Target(
    name="stone",
    description="stone, the edge skates and sparks",
    duration=0.30,
    snap_hz=4200.0, snap_decay=0.003, snap_gain=0.62,
    impact_low=1800.0, impact_high=11000.0, impact_decay=0.014, impact_gain=0.80,
    modes=(),
    tail_kind="scrape",
    tail_low=3000.0, tail_high=9500.0, tail_at=0.030, tail_width=0.022,
    tail_gain=0.34,
    body_hz=132.0, body_decay=0.028, body_gain=0.30,
    peak_db=-9.5,
)

# Flesh: no snap worth the name, no ring, a wet slap and a dull thump.
FLESH = Target(
    name="flesh",
    description="an unarmoured body; wet, dull, no ring",
    duration=0.32,
    snap_hz=900.0, snap_decay=0.004, snap_gain=0.14,
    impact_low=140.0, impact_high=1900.0, impact_decay=0.030, impact_gain=0.85,
    modes=(),
    tail_kind="wobble",
    tail_low=380.0, tail_decay=0.045, tail_gain=0.30,
    body_hz=74.0, body_decay=0.055, body_gain=0.70,
    peak_db=-10.0,
)

# Leather and padding: a slap with a bit of body behind it.
LEATHER = Target(
    name="leather",
    description="boiled leather or a padded gambeson",
    duration=0.30,
    snap_hz=1600.0, snap_decay=0.004, snap_gain=0.24,
    impact_low=260.0, impact_high=3000.0, impact_decay=0.020, impact_gain=0.80,
    modes=(),
    tail_kind="wobble",
    tail_low=600.0, tail_decay=0.030, tail_gain=0.22,
    body_hz=92.0, body_decay=0.042, body_gain=0.52,
    peak_db=-10.5,
)

# --------------------------------------------------------------------------
# unarmed targets
#
# `flesh` above is a blade opening someone up. These three are what a blunt
# strike lands on, and they are still described as the reference sword hits
# them -- that is the contract every target in this file keeps. A fist is a
# ratio against that sword, not a second reference. A sword to the head is a
# real medieval sound; it is the same table.
# --------------------------------------------------------------------------

# A torso folding. Where `flesh` is wet and mid, this is the deep one: almost
# no top end survives, and the low end carries it.
#
# That low end has to come from the `wobble` tail -- lowpassed noise, no pitch
# -- and not from the `body` layer, which is a pure decaying sine. A sine held
# for four or five cycles at 60-odd Hz is a tuned tom, and a body blow that
# lands on a note reads as a drum being tapped rather than someone being hit.
# So `body` is kept short enough to thump and quiet enough not to lead, and the
# thud sits in the tail. This is the one target where those two layers trade
# places; every other target here rings, and wants the sine.
BODY = Target(
    name="body",
    description="a torso taking a blunt hit; deep, damped, ribs under it",
    duration=0.38,
    snap_hz=1000.0, snap_decay=0.005, snap_gain=0.14,
    impact_low=70.0, impact_high=1500.0, impact_decay=0.055, impact_gain=1.00,
    modes=(),
    tail_kind="wobble",
    tail_low=260.0, tail_decay=0.110, tail_gain=1.30,
    body_hz=54.0, body_decay=0.048, body_gain=0.10,
    peak_db=-9.0,
)

# A skull is bone a few millimetres under the skin, so unlike the other soft
# targets it does ring -- but barely. The modes are inharmonic and dead inside
# 30 ms; ring them any longer and a head takes on a pitch, which reads as
# comedy rather than violence.
HEAD = Target(
    name="head",
    description="a skull: bone close under the skin, a crack and a short knock",
    duration=0.30,
    snap_hz=1900.0, snap_decay=0.0035, snap_gain=0.42,
    impact_low=320.0, impact_high=4200.0, impact_decay=0.018, impact_gain=0.90,
    modes=(
        (486.0, 0.028, 0.22),
        (742.0, 0.020, 0.13),
        (1156.0, 0.014, 0.07),
    ),
    tail_kind="wobble",
    tail_low=560.0, tail_decay=0.026, tail_gain=0.18,
    body_hz=112.0, body_decay=0.048, body_gain=0.62,
    peak_db=-9.0,
)

# A blocked hit. The defining quality is that it stops dead: a braced forearm
# gives nothing back, so every decay here is shorter than the target it saved.
GUARD = Target(
    name="guard",
    description="a raised forearm or shin taking the hit instead",
    duration=0.26,
    snap_hz=1900.0, snap_decay=0.0035, snap_gain=0.46,
    impact_low=260.0, impact_high=4400.0, impact_decay=0.015, impact_gain=0.85,
    modes=(),
    tail_kind="wobble",
    tail_low=700.0, tail_decay=0.022, tail_gain=0.26,
    body_hz=124.0, body_decay=0.038, body_gain=0.55,
    peak_db=-10.0,
)

TARGETS = {t.name: t for t in (BLADE, PLATE, MAIL, SHIELD, WOOD_TARGET,
                               STONE_TARGET, FLESH, LEATHER, BODY, HEAD,
                               GUARD)}
