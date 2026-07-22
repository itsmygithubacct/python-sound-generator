"""Combat events that are not a strike landing.

The rest of a melee exchange: blades sliding off each other, a sword coming out
of a scabbard, mail moving, boots, and a body going down. These are bespoke
rather than table-driven because each one has its own shape -- a draw is a
rising scrape, a fall is three events in a row -- and forcing them through the
strike layer stack would obscure more than it saved.
"""

from __future__ import annotations

import math
import random

from ...._audio import dsp
from .materials import BLADE
from .weapons import Weapon


def parry(weapon: Weapon, rng: random.Random) -> list[float]:
    """Blades meet and slide instead of biting.

    Same contact as a clash but softer, with the scrape carrying the event: the
    blades stay in contact far longer, and part late.
    """
    duration = 0.52 * weapon.ring_time_scale
    t = dsp.timebase(duration)
    ring = weapon.ring_scale

    snap = dsp.mul(dsp.highpass(dsp.noise(duration, rng),
                                2600.0 * weapon.snap_scale, 4),
                   [math.exp(-x / 0.004) for x in t])
    impact = dsp.mul(dsp.bandpass(dsp.noise(duration, rng),
                                  1400.0 * ring, 8200.0, 4),
                     [math.exp(-x / 0.022) for x in t])
    modes = weapon.scaled_modes(tuple((f, d, g * 0.7) for f, d, g in BLADE.modes))
    rings = dsp.modal_ring(duration, modes, rng)
    # the long slide: this is the layer that makes it a parry and not a hit
    scrape = dsp.mul(dsp.bandpass(dsp.noise(duration, rng),
                                  2200.0 * ring, 7800.0 * ring, 2),
                     [math.exp(-(((x - 0.090) / 0.070) ** 2)) for x in t])
    body = dsp.mul(dsp.sine(168.0 * weapon.body_scale, duration),
                   [math.exp(-x / 0.050) for x in t])

    out = dsp.mix(dsp.scale(snap, 0.30), dsp.scale(impact, 0.55), rings,
                  dsp.scale(scrape, 0.45), dsp.scale(body, 0.26))
    # they disengage late, and the free blades ring
    dsp.add_at(out, 0.185, dsp.modal_ring(
        0.20 * weapon.ring_time_scale,
        weapon.scaled_modes(((2530.0, 0.055, 0.30), (3410.0, 0.042, 0.22),
                             (4470.0, 0.032, 0.15))), rng), 0.45)
    return out


def draw(weapon: Weapon, rng: random.Random) -> list[float]:
    """Sword leaving a scabbard: the rising scrape, then the tip rings free.

    The pitch rise is the whole trick. Contact area shrinks as the blade slides
    out, so the scrape climbs; the zing only exists once the tip clears.
    """
    ts = weapon.swing_time_scale
    duration = 0.62 * ts
    ring = weapon.ring_scale
    clear = 0.72          # fraction of the draw where the tip comes free

    # scrape climbs as the blade slides out
    env = dsp.ramp(duration, 0.15, 1.0, curve=1.6)
    freqs = [(620.0 + 2600.0 * (e ** 1.8)) * ring for e in env]
    scrape = dsp.mul(dsp.svf_bandpass(dsp.noise(duration, rng), freqs, q=3.0),
                     env)
    # leather and wood of the scabbard, under it
    friction = dsp.mul(dsp.lowpass(dsp.noise(duration, rng), 700.0, 4),
                       [e * 0.6 for e in env])
    out = dsp.mix(dsp.scale(scrape, 0.85), dsp.scale(friction, 0.30))

    # the tip clears and the free blade rings, long, because nothing damps it
    zing = weapon.scaled_modes(tuple((f, d * 3.2, g)
                                     for f, d, g in BLADE.modes[:4]))
    dsp.add_at(out, duration * clear,
               dsp.modal_ring(duration * (1.0 - clear) + 0.22, zing, rng), 0.70)
    return out


def sheathe(weapon: Weapon, rng: random.Random) -> list[float]:
    """Blade going back in: the scrape falls, and it ends on a stop."""
    ts = weapon.swing_time_scale
    duration = 0.46 * ts
    ring = weapon.ring_scale

    env = dsp.bell_env(duration, duration * 0.45, 0.16 * ts, 0.10 * ts)
    freqs = [(2400.0 * (0.30 + 0.70 * (1.0 - i / max(1, len(env) - 1)))) * ring
             for i in range(len(env))]
    scrape = dsp.mul(dsp.svf_bandpass(dsp.noise(duration, rng), freqs, q=2.4),
                     env)
    friction = dsp.mul(dsp.lowpass(dsp.noise(duration, rng), 620.0, 4), env)
    out = dsp.mix(dsp.scale(scrape, 0.70), dsp.scale(friction, 0.34))

    # guard meets scabbard throat: a wood-and-leather stop, not a ring
    t_stop = duration * 0.86
    stop_dur = duration - t_stop + 0.10
    st = dsp.timebase(stop_dur)
    stop = dsp.mix(
        dsp.scale(dsp.mul(dsp.bandpass(dsp.noise(stop_dur, rng), 300.0,
                                       3200.0, 4),
                          [math.exp(-x / 0.016) for x in st]), 0.80),
        dsp.scale(dsp.mul(dsp.sine(128.0 * weapon.body_scale, stop_dur),
                          [math.exp(-x / 0.045) for x in st]), 0.50))
    dsp.add_at(out, t_stop, stop, 0.9)
    return out


def rustle(rng: random.Random, duration: float = 0.5,
           heaviness: float = 1.0) -> list[float]:
    """Mail moving: hundreds of links, each letting go at its own moment."""
    slips = dsp.stick_slip(duration, rng, rate=170.0 * heaviness,
                           freq=3400.0, jitter=0.95)
    env = dsp.bell_env(duration, duration * 0.45, duration * 0.35,
                       duration * 0.30)
    jingle = dsp.mul(dsp.bandpass(slips, 2400.0, 9500.0, 2), env)
    # the padding under the mail
    cloth = dsp.mul(dsp.lowpass(dsp.noise(duration, rng), 900.0, 4),
                    [e * 0.5 for e in env])
    return dsp.mix(dsp.scale(jingle, 0.85), dsp.scale(cloth, 0.20))


SURFACES = {
    "stone": (2600.0, 0.008, 150.0, 0.030, 0.55),
    "dirt": (1200.0, 0.014, 92.0, 0.045, 0.30),
    "wood": (1800.0, 0.010, 118.0, 0.060, 0.70),
}


def footstep(rng: random.Random, surface: str = "stone",
             armored: bool = False) -> list[float]:
    """A boot landing. `armored` adds the mail that moves with the leg."""
    if surface not in SURFACES:
        raise ValueError(f"unknown surface {surface!r}; "
                         f"expected one of {sorted(SURFACES)}")
    grit_hz, click_decay, thud_hz, thud_decay, hardness = SURFACES[surface]
    duration = 0.26
    t = dsp.timebase(duration)

    click = dsp.mul(dsp.highpass(dsp.noise(duration, rng), grit_hz, 4),
                    [math.exp(-x / click_decay) for x in t])
    thud = dsp.mul(dsp.sine(thud_hz, duration),
                   [math.exp(-x / thud_decay) for x in t])
    scuff = dsp.mul(dsp.bandpass(dsp.noise(duration, rng), 700.0, 4200.0, 2),
                    [math.exp(-(((x - 0.022) / 0.018) ** 2)) for x in t])
    out = dsp.mix(dsp.scale(click, 0.45 * hardness),
                  dsp.scale(thud, 0.75),
                  dsp.scale(scuff, 0.22))
    if armored:
        dsp.add_at(out, 0.004, rustle(rng, 0.22, heaviness=0.7), 0.45)
    return out


def body_fall(rng: random.Random, armored: bool = True) -> list[float]:
    """A body going down: the impact, the mail, then the weapon it dropped."""
    duration = 0.85
    t = dsp.timebase(duration)

    thump = dsp.mul(dsp.sine(58.0, duration),
                    [math.exp(-x / 0.11) for x in t])
    flesh = dsp.mul(dsp.lowpass(dsp.noise(duration, rng), 420.0, 4),
                    [math.exp(-x / 0.055) for x in t])
    out = dsp.mix(dsp.scale(thump, 0.80), dsp.scale(flesh, 0.45))

    if armored:
        dsp.add_at(out, 0.006, rustle(rng, 0.40, heaviness=1.4), 0.55)
        # the sword lands a moment after its owner
        clatter_dur = 0.34
        ct = dsp.timebase(clatter_dur)
        clatter = dsp.mix(
            dsp.scale(dsp.modal_ring(clatter_dur, ((1840.0, 0.045, 0.30),
                                                   (2610.0, 0.034, 0.22),
                                                   (3720.0, 0.026, 0.14)),
                                     rng), 1.0),
            dsp.scale(dsp.mul(dsp.sine(112.0, clatter_dur),
                              [math.exp(-x / 0.040) for x in ct]), 0.40))
        dsp.add_at(out, 0.155, clatter, 0.50)
    return out
