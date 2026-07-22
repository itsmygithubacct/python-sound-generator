"""Weapons, described physically, and how those numbers become sound.

A weapon is length, thickness, mass, material and edge. Everything audible is
derived from those five numbers, so adding a weapon means measuring one, not
tuning a mode table by hand.

Honesty about the model
-----------------------
Only one part of this is derived physics: the material's bar wave speed
sqrt(E/rho) scales the ring, which is why bronze lands below steel for free.

The exponents below are **perceptual fits, not derived from beam theory**.
They are informed by it -- a bending mode goes as thickness/length^2, so bigger
means lower and heavier means longer -- but they are not that formula. The real
one does not apply here: Euler-Bernoulli is only valid for the lowest bending
modes, while the audible clang of a struck blade lives in modes well above
where it stops being true. Rather than dress a fitted curve up as physics, the
exponents are tuned so the ordering and the ratios sound right, and every one
is anchored to a reference weapon.

The reference is the arming sword. Its scale factors are exactly 1.0, so it
reproduces the hand-tuned tables in materials.py byte for byte, and every other
weapon is a ratio against it. tests/test_weapons.py pins that.
"""

from __future__ import annotations

from dataclasses import dataclass

from .materials import BONE, BRONZE, IRON, STEEL, Material, Modes, Target

# The arming sword. Every scale factor is a ratio against these.
REF_LENGTH = 0.80       # m
REF_THICKNESS = 0.0045  # m
REF_MASS = 1.20         # kg
REF_MATERIAL = STEEL
REF_EDGE = 1.0


@dataclass(frozen=True)
class Weapon:
    """The striking object. A fist is one of these; see LIMBS below."""
    name: str
    description: str
    length: float       # m, the acoustically relevant striking part
    thickness: float    # m; decorative -- nothing below reads it
    mass: float         # kg; for a limb, the effective striking mass
    material: Material
    edge: float         # 0.0 blunt .. 1.0 razor
    cloth: float = 0.0  # 0.0 bare .. 1.0 a heavy sleeve moving with it

    # -- derived scale factors, all 1.0 for the reference sword -------------

    @property
    def _speed_ratio(self) -> float:
        return self.material.wave_speed / REF_MATERIAL.wave_speed

    @property
    def ring_scale(self) -> float:
        """Ring pitch: heavier and longer ring lower, stiffer rings higher."""
        return ((REF_MASS / self.mass) ** 0.25
                * (REF_LENGTH / self.length) ** 0.25
                * self._speed_ratio)

    @property
    def ring_time_scale(self) -> float:
        """Ring length: mass sustains it, material damping kills it."""
        return ((self.mass / REF_MASS) ** 0.35
                * (self.material.q_factor / REF_MATERIAL.q_factor))

    @property
    def body_scale(self) -> float:
        """The low knock through guard and grip."""
        return ((REF_MASS / self.mass) ** 0.30
                * (REF_LENGTH / self.length) ** 0.20)

    @property
    def swing_time_scale(self) -> float:
        """Heavy, long weapons take longer to come round."""
        return ((self.mass / REF_MASS) ** 0.30
                * (self.length / REF_LENGTH) ** 0.40)

    @property
    def swing_freq_scale(self) -> float:
        """Big weapons move more air and whoosh lower."""
        return ((REF_MASS / self.mass) ** 0.18
                * (REF_LENGTH / self.length) ** 0.12)

    @property
    def snap_scale(self) -> float:
        """A razor edge cracks; a blunt lump thumps."""
        return 0.55 + 0.45 * self.edge

    def scaled_modes(self, modes: Modes) -> Modes:
        return tuple((f * self.ring_scale, d * self.ring_time_scale, g)
                     for f, d, g in modes)

    def describe(self) -> dict[str, str]:
        return {
            "length": f"{self.length:.2f} m",
            "thickness": f"{self.thickness * 1000:.1f} mm",
            "mass": f"{self.mass:.2f} kg",
            "material": self.material.name,
            "edge": f"{self.edge:.2f}",
            "cloth": f"{self.cloth:.2f}",
            "wave speed": f"{self.material.wave_speed:.0f} m/s",
            "ring": f"x{self.ring_scale:.3f}",
            "ring time": f"x{self.ring_time_scale:.3f}",
            "body": f"x{self.body_scale:.3f}",
            "swing time": f"x{self.swing_time_scale:.3f}",
        }


SWORD = Weapon("sword", "arming sword, the reference weapon",
               REF_LENGTH, REF_THICKNESS, REF_MASS, REF_MATERIAL, REF_EDGE)
DAGGER = Weapon("dagger", "rondel dagger, quick and bright",
                0.30, 0.0035, 0.30, STEEL, 1.0)
SHORTSWORD = Weapon("shortsword", "gladius-length blade",
                    0.55, 0.0040, 0.75, STEEL, 1.0)
LONGSWORD = Weapon("longsword", "hand-and-a-half blade",
                   0.95, 0.0048, 1.55, STEEL, 0.95)
GREATSWORD = Weapon("greatsword", "two-handed, slow and deep",
                    1.20, 0.0055, 2.80, STEEL, 0.85)
KATANA = Weapon("katana", "thick-spined single edge",
                0.72, 0.0060, 1.10, STEEL, 1.0)
AXE = Weapon("axe", "bearded axe, iron head on a haft",
             0.75, 0.0090, 1.80, IRON, 0.80)
MACE = Weapon("mace", "flanged mace, no edge at all",
              0.70, 0.0250, 2.00, IRON, 0.05)
SPEAR = Weapon("spear", "leaf-blade spearhead",
               0.30, 0.0040, 1.30, STEEL, 0.90)
BRONZE_SWORD = Weapon("bronze_sword", "cast bronze, rings lower than steel",
                      0.65, 0.0055, 1.10, BRONZE, 0.90)

BLADED = {w.name: w for w in (DAGGER, SHORTSWORD, SWORD, LONGSWORD,
                              GREATSWORD, KATANA, AXE, MACE, SPEAR,
                              BRONZE_SWORD)}


# --------------------------------------------------------------------------
# limbs
#
# An unarmed fighter is still swinging a striking object at a target, so a
# limb is a Weapon and gets the whole matrix for free: a fist against plate,
# a boot against a shield. Two of the five numbers mean something slightly
# different for a limb, and both are worth saying out loud rather than
# leaving for someone to trip over:
#
# - `mass` is the EFFECTIVE striking mass -- the part of the body actually
#   behind the blow, ~3 kg for a jab and ~6 kg for a knee. It is not what the
#   limb weighs. For a sword those two are the same number, which is why the
#   field never had to distinguish them before.
# - `length` is the contact geometry (the knuckles, the shin), not the reach.
#   The model conflates "how long the striking part is" with "how long it
#   takes to come round" because for a blade those track together. For a limb
#   they do not, and the contact length is the one that matters: it is what
#   sets the ring, and it happens to give fast swing times, which is right for
#   a punch anyway. If a limb ever needs a swing slower than its contact
#   geometry implies, that conflation is what will have to give.
#
# The exponents above were fitted to blades, and a fist sits outside the range
# they were fitted over -- shorter than a dagger, heavier than a greatsword.
# They are extrapolated here on the evidence that the ordering still holds
# (a kick lands lower and slower than a jab; an elbow snaps hardest) and that
# the results sound right. That is the same standard the blades were tuned to,
# but it is an extrapolation, and the limb numbers are where to look first if
# something sounds wrong.
# --------------------------------------------------------------------------

# `edge` on a limb reads as how concentrated the contact is, not how sharp it
# is -- an elbow lands on a point, a boot lands on a length of shin. The values
# stay low because edge also decides how much the striker hisses through air,
# and no part of a person cuts it.
FIST = Weapon("fist", "bare knuckles, a thrown punch",
              0.10, 0.055, 3.00, BONE, 0.15, cloth=0.90)
ELBOW = Weapon("elbow", "the sharpest point on a body, and the shortest",
               0.07, 0.050, 4.20, BONE, 0.25, cloth=0.70)
KNEE = Weapon("knee", "driven up, most of the body behind it",
              0.10, 0.090, 6.00, BONE, 0.20, cloth=0.85)
BOOT = Weapon("boot", "shin or instep, the heaviest thing a fighter throws",
              0.16, 0.080, 5.50, BONE, 0.10, cloth=1.00)

LIMBS = {w.name: w for w in (FIST, ELBOW, KNEE, BOOT)}

WEAPONS = {**BLADED, **LIMBS}


def scaled_target(weapon: Weapon, target: Target) -> dict:
    """Resolve a (weapon, target) pair into concrete render parameters.

    Returns plain floats so lib/strikes.py stays a renderer and this stays the
    only place that knows how physical parameters become frequencies.
    """
    ring = weapon.ring_scale if target.scales_with_weapon else 1.0
    ring_t = weapon.ring_time_scale if target.scales_with_weapon else 1.0
    body = weapon.body_scale if target.scales_with_weapon else 1.0
    return {
        "duration": target.duration * ring_t,
        "snap_hz": target.snap_hz * weapon.snap_scale,
        "snap_decay": target.snap_decay,
        "snap_gain": target.snap_gain * weapon.snap_scale,
        "impact_low": target.impact_low * ring,
        "impact_high": target.impact_high * weapon.snap_scale,
        "impact_decay": target.impact_decay * ring_t,
        "impact_gain": target.impact_gain,
        "modes": tuple((f * ring, d * ring_t, g) for f, d, g in target.modes),
        "tail_kind": target.tail_kind,
        "tail_low": target.tail_low * ring,
        "tail_high": target.tail_high * ring,
        "tail_at": target.tail_at,
        "tail_width": target.tail_width,
        "tail_decay": target.tail_decay * ring_t,
        "tail_rate": target.tail_rate,
        "tail_gain": target.tail_gain,
        "body_hz": target.body_hz * body,
        "body_decay": target.body_decay * ring_t,
        "body_gain": target.body_gain,
        "chink_modes": tuple((f * ring, d * ring_t, g)
                             for f, d, g in target.chink_modes),
        "chink_at": target.chink_at,
        "chink_duration": target.chink_duration * ring_t,
        "chink_gain": target.chink_gain,
        "peak_db": target.peak_db,
    }
