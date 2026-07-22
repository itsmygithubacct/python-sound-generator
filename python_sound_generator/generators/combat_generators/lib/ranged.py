"""Bow and arrow.

A bow is a spring made of wood and a string, so the sounds come from three
places: the limbs creaking under load, the string released, and the shaft
arriving. The string uses Karplus-Strong (lib/dsp.pluck) because that is
literally a plucked string model; the creak uses stick-slip because wood under
load does not slide smoothly, it grabs and lets go.
"""

from __future__ import annotations

import math
import random

from ...._audio import dsp


def bow_draw(rng: random.Random, duration: float = 0.85) -> list[float]:
    """Limbs loading up: creak, and the string stretching."""
    # the wood grabs and releases as it bends, faster as tension climbs
    creak = dsp.stick_slip(duration, rng, rate=26.0, freq=360.0, jitter=0.75)
    env = dsp.ramp(duration, 0.10, 1.0, curve=1.8)
    creak = dsp.mul(dsp.bandpass(creak, 220.0, 2600.0, 2), env)
    # string fibres under load: thin, high, quiet
    stretch = dsp.mul(dsp.bandpass(dsp.noise(duration, rng), 3000.0, 8000.0, 2),
                      [e ** 2.2 * 0.5 for e in env])
    # the low groan of the riser
    groan = dsp.mul(dsp.sine(88.0, duration), [e * 0.35 for e in env])
    return dsp.mix(dsp.scale(creak, 0.90), dsp.scale(stretch, 0.35),
                   dsp.scale(groan, 0.30))


def bow_release(rng: random.Random) -> list[float]:
    """Loose: string snaps forward, limbs slam, arrow leaves.

    The string is the event people recognise, but the limb thump underneath is
    what gives it weight.
    """
    duration = 0.45
    t = dsp.timebase(duration)

    # the string itself, damped hard because the fingers just left it
    string = dsp.pluck(146.0, duration, rng, damping=0.82, blend=0.94)
    string = dsp.mul(string, [math.exp(-x / 0.075) for x in t])
    # limbs arriving at brace: low, immediate
    limb = dsp.mul(dsp.sine(72.0, duration), [math.exp(-x / 0.055) for x in t])
    # the crack of the release
    crack = dsp.mul(dsp.bandpass(dsp.noise(duration, rng), 900.0, 6500.0, 4),
                    [math.exp(-x / 0.012) for x in t])
    # arrow off the rest
    off = dsp.mul(dsp.bandpass(dsp.noise(duration, rng), 1800.0, 7000.0, 2),
                  [math.exp(-(((x - 0.020) / 0.016) ** 2)) for x in t])

    return dsp.mix(dsp.scale(string, 0.70), dsp.scale(limb, 0.62),
                   dsp.scale(crack, 0.55), dsp.scale(off, 0.30))


def arrow_flight(rng: random.Random, duration: float = 0.40) -> list[float]:
    """An arrow going past: thin, fast, and spinning.

    Same swept-bandpass idea as a sword swing, but a shaft is far thinner than
    a blade, so it sits higher and passes quicker. The flutter is the fletching
    spinning; without it this is just a whoosh.
    """
    peak = duration * 0.5
    env = dsp.bell_env(duration, peak, 0.070, 0.045)
    freqs = [900.0 + 3400.0 * (e ** 1.5) for e in env]
    air = dsp.mul(dsp.svf_bandpass(dsp.noise(duration, rng), freqs, q=1.8), env)

    # fletching spin: amplitude flutter, rising as it approaches
    flutter = [1.0 + 0.35 * e * math.sin(dsp.TAU * 62.0 * x)
               for x, e in zip(dsp.timebase(duration), env)]
    air = dsp.mul(air, flutter)

    hiss = dsp.mul(dsp.bandpass(dsp.noise(duration, rng), 4000.0, 11000.0, 4),
                   [e ** 3.0 for e in env])
    return dsp.mix(dsp.scale(air, 1.0), dsp.scale(hiss, 0.40))


def arrow_hit_wood(rng: random.Random) -> list[float]:
    """Thunk into a post, and then the shaft rings itself out.

    The wobble is the point: the shaft is a thin rod clamped at one end, so it
    oscillates for a surprisingly long time and the pitch drops as it settles.
    """
    duration = 0.70
    t = dsp.timebase(duration)

    bite = dsp.mul(dsp.bandpass(dsp.noise(duration, rng), 400.0, 5200.0, 4),
                   [math.exp(-x / 0.010) for x in t])
    thock = dsp.mul(dsp.sine(126.0, duration),
                    [math.exp(-x / 0.048) for x in t])
    wood = dsp.mul(dsp.modal_ring(duration, ((318.0, 0.030, 0.24),
                                             (496.0, 0.020, 0.14)), rng),
                   [1.0] * len(t))
    # shaft wobble: pitch sags as the amplitude dies
    wob_f = [172.0 * (1.0 - 0.16 * (1.0 - math.exp(-x / 0.18))) for x in t]
    wobble = dsp.mul(dsp.sine_track(wob_f),
                     [math.exp(-x / 0.16) * (1.0 - math.exp(-x / 0.004))
                      for x in t])

    return dsp.mix(dsp.scale(bite, 0.70), dsp.scale(thock, 0.62), wood,
                   dsp.scale(wobble, 0.38))


def arrow_hit_armor(rng: random.Random) -> list[float]:
    """A deflection: bright, brief, and it goes away spinning."""
    duration = 0.42
    t = dsp.timebase(duration)

    ping = dsp.modal_ring(duration, ((3180.0, 0.038, 0.30),
                                     (4420.0, 0.028, 0.22),
                                     (5910.0, 0.020, 0.14),
                                     (1720.0, 0.030, 0.12)), rng)
    crack = dsp.mul(dsp.highpass(dsp.noise(duration, rng), 2800.0, 4),
                    [math.exp(-x / 0.005) for x in t])
    plate = dsp.mul(dsp.lowpass(dsp.noise(duration, rng), 800.0, 4),
                    [math.exp(-x / 0.040) for x in t])
    body = dsp.mul(dsp.sine(104.0, duration), [math.exp(-x / 0.038) for x in t])
    out = dsp.mix(ping, dsp.scale(crack, 0.55), dsp.scale(plate, 0.20),
                  dsp.scale(body, 0.34))
    # the shaft tumbles away
    dsp.add_at(out, 0.085, arrow_flight(rng, 0.22), 0.22)
    return out
