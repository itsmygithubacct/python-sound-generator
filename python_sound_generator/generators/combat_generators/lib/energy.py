"""Energy attacks: the one thing here that is invented rather than derived.

Every other cue in this project answers to something real. A blade's ring comes
out of sqrt(E/rho); a limb's thump comes out of an effective striking mass; a
boot on dirt comes out of how grit behaves. The model can be wrong, but it can
be *checked*.

A fireball cannot. Nobody has measured one, there is no material to look up, and
any physics dressed onto it would be decoration. So these two cues do not
pretend: the parameters below are named for what they do to the ear, the shapes
are borrowed from what the genre has taught players to expect, and the only test
that can be written against them is whether they do the job the game needs.

They are built from the same primitives as everything else, so they land in the
same mix at the same levels without special handling.
"""

from __future__ import annotations

import math
import random

from ...._audio import dsp


def energy_charge(rng: random.Random, duration: float = 0.62) -> list[float]:
    """Power gathering, and stopping without resolving.

    The cue's whole job is to promise a release, so it ends at its loudest and
    brightest instead of decaying. An envelope that fell away at the end would
    read as the move fizzling out, which is the opposite of the message.
    """
    env = dsp.ramp(duration, 0.04, 1.0, curve=2.0)

    # air dragged inward: the bandpass climbs with the envelope, so it tightens
    # rather than just getting louder
    freqs = [600.0 + 5400.0 * (e ** 1.7) for e in env]
    air = dsp.mul(dsp.svf_bandpass(dsp.noise(duration, rng), freqs, q=2.2), env)

    # the drone: three partials sweeping up together. The ratios are irrational
    # on purpose -- tuned to a chord it sounds like music starting, not power
    # building. It sits high and quiet: a charge that leads with its low end
    # reads as something heavy arriving rather than something building.
    drone = dsp.silence(duration)
    for ratio, gain in ((1.0, 0.55), (1.47, 0.30), (2.13, 0.18)):
        track = dsp.ramp(duration, 240.0 * ratio, 660.0 * ratio, curve=1.5)
        voice = dsp.sine_track(track, phase=rng.uniform(0.0, dsp.TAU))
        dsp.add_at(drone, 0.0, dsp.mul(voice, [e ** 1.2 for e in env]), gain)

    # sparkle: bright grains, arriving irregularly, louder as it winds up.
    # This is the layer that says "energy" rather than "engine".
    grains = dsp.stick_slip(duration, rng, rate=48.0, freq=6400.0, jitter=0.8)
    sparkle = dsp.mul(dsp.highpass(grains, 3400.0, 2), [e ** 2.4 for e in env])

    return dsp.mix(dsp.scale(air, 0.80), dsp.scale(drone, 0.30),
                   dsp.scale(sparkle, 1.10))


def energy_release(rng: random.Random, duration: float = 0.78) -> list[float]:
    """The shot leaving: a crack, a body of noise, and something departing.

    The departure is the point, and it is the only layer that could not be left
    out. A blast that merely decays sits where it was fired; this one slides
    down and thins as it goes, so it reads as a thing that went somewhere.
    """
    t = dsp.timebase(duration)

    # the crack: the only part of this that is instantaneous
    crack = dsp.mul(dsp.highpass(dsp.noise(duration, rng), 1800.0, 4),
                    [math.exp(-x / 0.008) for x in t])
    # the body of the blast, falling away
    blast = dsp.mul(dsp.bandpass(dsp.noise(duration, rng), 90.0, 2600.0, 2),
                    [math.exp(-x / 0.13) for x in t])
    # the low end, dropping so the shot has weight leaving the hand
    boom = dsp.mul(dsp.sine_track(dsp.ramp(duration, 128.0, 46.0, curve=0.6)),
                   [math.exp(-x / 0.16) for x in t])
    # what departs: a narrow band sliding down and out
    freqs = [2400.0 * math.exp(-x / 0.26) + 190.0 for x in t]
    departure = dsp.mul(dsp.svf_bandpass(dsp.noise(duration, rng), freqs,
                                         q=3.4),
                        [math.exp(-x / 0.30) for x in t])

    return dsp.mix(dsp.scale(crack, 0.30), dsp.scale(blast, 0.95),
                   dsp.scale(boom, 1.40), dsp.scale(departure, 0.55))
