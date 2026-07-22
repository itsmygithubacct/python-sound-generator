"""Strike renderers: a weapon meeting a target, and the swing that gets it there.

A strike is a stack of short layers rather than one tone:

- contact is 2-5 ms of bright filtered noise. It carries most of the "metal"
  impression and is over almost immediately.
- the ring is inharmonic decaying sinusoids that die inside ~100 ms. Ring them
  longer, or tune them to a harmonic series, and a sword hit turns into a
  church bell.
- the whoosh is noise through a bandpass whose centre sweeps with the amplitude
  envelope, so the blade brightens as it passes and dulls as it recedes.

Layer order is fixed and load-bearing: snap, impact, ring, tail, body, chink.
Each layer draws from `rng` in that order, so the reference sword reproduces
the original hand-tuned renders byte for byte. Reorder them and every seeded
render in the project changes.
"""

from __future__ import annotations

import math
import random

from ...._audio import dsp
from .materials import Target
from .weapons import Weapon, scaled_target


def swing(weapon: Weapon, rng: random.Random,
          duration: float = 0.36, peak: float = 0.20) -> list[float]:
    """The blade moving through air: loudest and brightest as it passes.

    `duration` and `peak` are for the reference sword and are scaled by the
    weapon, so a greatsword takes visibly longer to come round than a dagger.

    This is also a thrown punch. A limb carries a sleeve, and the cloth layer
    is most of what you hear -- a fist has no edge to hiss and moves too little
    air to rush, so without it a punch is nearly silent.
    """
    ts = weapon.swing_time_scale
    fs = weapon.swing_freq_scale
    duration = duration * ts
    peak = peak * ts
    env = dsp.bell_env(duration, peak, 0.085 * ts, 0.055 * ts)

    # air sweeps up with the swing: a low rush, then a bright pass
    freqs = [(260.0 + 2100.0 * (e ** 1.4)) * fs for e in env]
    air = dsp.mul(dsp.svf_bandpass(dsp.noise(duration, rng), freqs, q=1.4), env)
    # the edge only hisses at the closest point, and only if it is sharp
    hiss = dsp.bandpass(dsp.noise(duration, rng), 3200.0 * fs, 9000.0 * fs, 4)
    hiss = dsp.mul(hiss, [e ** 3.0 for e in env])
    # a little body so it is not all top end
    body = dsp.lowpass(dsp.noise(duration, rng), 220.0 * fs, 4)
    body = dsp.mul(body, [e ** 1.6 for e in env])

    # Two of these gains answer to the striker, and both matter only because a
    # limb exists now. The reference sword has edge 1.0 and cloth 0.0, so its
    # swing is untouched to the byte and the goldens pin that.
    #
    # air:  cloth buys it down as it buys the fabric layer up, which is one
    #       fact and not two -- a sleeved limb is a bluff body wrapped in
    #       fabric, with no edge to shed a clean rush off and a sleeve that
    #       spoils what flow there is. Without it a fist whooshes brighter than
    #       a swung mace: swing_freq_scale reads a fist as "small, therefore
    #       high", which is the blade exponents extrapolating past their range.
    # hiss: driven by `edge`, which is what the line above it always claimed
    #       ("only if it is sharp") and never did -- snap_scale floors at 0.55,
    #       so a mace used to whoosh like a razor at 55%, and so did a fist.
    #       This quiets the blunt end of the rack (mace, axe, greatsword) to
    #       what each could actually cut.
    parts = [dsp.scale(air, 1.0 - 0.72 * weapon.cloth),
             dsp.scale(hiss, 0.55 * weapon.edge),
             dsp.scale(body, 0.9)]

    # A sleeve or a trouser leg, snapping taut at the fastest point. This is
    # skipped for a bare blade rather than mixed in at zero gain: it draws from
    # rng, and a silent draw would still shift every seeded blade render in the
    # project. See the layer-order note at the top of this file.
    if weapon.cloth > 0.0:
        fabric = dsp.bandpass(dsp.noise(duration, rng),
                              140.0 * fs, 1600.0 * fs, 2)
        parts.append(dsp.scale(dsp.mul(fabric, [e ** 1.5 for e in env]),
                               2.80 * weapon.cloth))

    return dsp.mix(*parts)


def _tail(kind: str, p: dict, duration: float, t: list[float],
          rng: random.Random) -> list[float]:
    """What happens just after contact, per target."""
    if kind == "scrape":
        # blades slide across each other for a few milliseconds
        env = [math.exp(-(((x - p["tail_at"]) / p["tail_width"]) ** 2))
               for x in t]
        return dsp.mul(dsp.bandpass(dsp.noise(duration, rng),
                                    p["tail_low"], p["tail_high"], 2), env)
    if kind == "wobble":
        # the struck thing flexes: low, short, no pitch
        env = [math.exp(-x / p["tail_decay"]) for x in t]
        return dsp.mul(dsp.lowpass(dsp.noise(duration, rng),
                                   p["tail_low"], 4), env)
    if kind == "rattle":
        # hundreds of loose links, each let go at its own moment
        slips = dsp.stick_slip(duration, rng, rate=p["tail_rate"],
                               freq=p["tail_low"], jitter=0.9)
        env = [math.exp(-x / p["tail_decay"]) for x in t]
        return dsp.mul(dsp.highpass(slips, p["tail_low"] * 0.5, 2), env)
    return dsp.silence(duration)


def strike(weapon: Weapon, target: Target, rng: random.Random) -> list[float]:
    """One hit: weapon meets target. Returns an unmastered buffer."""
    p = scaled_target(weapon, target)
    duration = p["duration"]
    t = dsp.timebase(duration)

    # 1. contact snap
    snap = dsp.mul(dsp.highpass(dsp.noise(duration, rng), p["snap_hz"], 4),
                   [math.exp(-x / p["snap_decay"]) for x in t])
    # 2. impact burst
    impact = dsp.mul(dsp.bandpass(dsp.noise(duration, rng), p["impact_low"],
                                  p["impact_high"], 4),
                     [math.exp(-x / p["impact_decay"]) for x in t])
    # 3. ring
    rings = dsp.modal_ring(duration, p["modes"], rng)
    # 4. tail
    tail = _tail(p["tail_kind"], p, duration, t, rng)
    # 5. body knock
    body = dsp.mul(dsp.sine(p["body_hz"], duration),
                   [math.exp(-x / p["body_decay"]) for x in t])

    out = dsp.mix(dsp.scale(snap, p["snap_gain"]),
                  dsp.scale(impact, p["impact_gain"]),
                  rings,
                  dsp.scale(tail, p["tail_gain"]),
                  dsp.scale(body, p["body_gain"]))

    # 6. the two part with a lighter second chink
    if p["chink_modes"]:
        dsp.add_at(out, p["chink_at"],
                   dsp.modal_ring(p["chink_duration"], p["chink_modes"], rng),
                   p["chink_gain"])
    return out


def full_strike(weapon: Weapon, target: Target,
                rng: random.Random) -> list[float]:
    """A swing that lands: contact sits just under the peak of the whoosh.

    The six millisecond offset is what makes it read as one event rather than
    two sounds played back to back.
    """
    ts = weapon.swing_time_scale
    peak = 0.20 * ts
    impact_at = peak - 0.006 * ts
    swing_buf = swing(weapon, rng, 0.42, 0.20)
    strike_buf = strike(weapon, target, rng)

    total = max(0.78 * ts, impact_at + len(strike_buf) / dsp.SR + 0.02)
    out = dsp.silence(total)
    dsp.add_at(out, 0.0, swing_buf, 0.70)
    dsp.add_at(out, impact_at, strike_buf, 1.0)
    return out
