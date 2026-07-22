"""Procedural magic and energy synthesis. Standard library only.

Like ``weapons_destruction_generator`` and ``environment_ambience_generator``
this has no recordings and no third-party dependencies: it shares the
standard-library DSP core from ``combat_generators`` and, for the one loop cue,
the seam-safe extensions from ``environment_ambience_generator.loopdsp``. There
is nothing to license here because nothing is sampled.

Why this is synthesized and fire is not
---------------------------------------
The project has learned by ear where procedural DSP wins and where it loses.
It loses on chaotic transients -- fire, flesh impacts, a real crackle -- which
is why ``fire_sound_generator`` is built on recordings. It *wins* on anything
with a rate: a pitched sweep, a shimmering tremolo, a harmonic charge, a
resonant zap. Magic is the pure case of the winning half. Every cue here has a
rate the ear can follow -- a crescent that sweeps up and lets go, a ward that
rings and settles into a hum, dust that sprinkles in tuned grains -- and none
of it is turbulence. This is the same technique that made the laser, plasma,
energy_charge and energy_release cues that were signed off by ear.

The beam
--------
``beam`` is the priority: Fantasy Adventure's enchanted-sword projectile launch, and the
one cue the magic bank exists to unblock. The brief is specific -- "bright steel
resonance snaps into a compact magical crescent, followed by one short focused
airy energy streak" -- and it is built in exactly those three parts, in that
order, in :func:`beam`. It is deliberately *not* a descending laser: the
crescent rises and then releases, which is the line the brief draws between an
enchantment and a phaser.

Elements
--------
The one axis with real teeth is the element (``style``): arcane, holy, shadow,
nature. It is not a coat of paint -- each style moves the register, the voice
detune, the spectral tilt, the harmonic ratios and the shimmer rate, and every
cue reads those five fields. Holy sits an octave up, nearly pure, on a
consonant overtone stack; shadow sits low and beats on a minor-second/tritone
stack; nature is fourth-based and woody; arcane is the mystic default. Nothing
here varies an axis it does not use.

Determinism
-----------
Every stochastic choice draws from one ``random.Random`` passed in by the
caller, in a fixed order, so identical ``(cue, options, seed)`` produces
byte-identical PCM. The rng draw order is load-bearing exactly as it is in
combat_generators: reordering or adding a layer shifts every draw after it. The
goldens in the test suite pin that.
"""

from __future__ import annotations

from ..._audio.dsp import write_wav as write_wav

import math
import random
from dataclasses import dataclass
from typing import Sequence

from ..._audio import dsp
from ..._audio import loops as loopdsp


SR = dsp.SR
TAU = dsp.TAU

# Loop bounds, for the ward hum. A magical drone does not need minutes.
MIN_SECONDS = 1.0
MAX_SECONDS = 60.0


# --------------------------------------------------------------------------
# elements
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Style:
    """An element, and what it does to a spell.

    Five fields, all of them heard:

    - ``octave`` shifts every pitch. Holy sits high and bright, shadow sits low
      and heavy; a spell is not the same spell an octave away.
    - ``detune`` spreads a harmonic stack's voices apart. Near zero it is pure
      (holy); wide it beats, which is what makes shadow read as unstable.
    - ``brightness`` tilts the airy and noise layers and moves filter cutoffs.
    - ``ratios`` is the harmonic character. Consonant octaves and fifths read as
      benevolent; a minor second and a tritone read as a curse.
    - ``shimmer_hz`` is the tremolo/shimmer rate, the flutter over the tone.
    """

    name: str
    label: str
    description: str
    octave: float
    detune: float
    brightness: float
    ratios: tuple[float, ...]
    shimmer_hz: float


STYLES: dict[str, Style] = {
    # The mystic default: a fifth-and-octave stack, lightly detuned so it
    # shimmers rather than sits still. Neither bright nor dark.
    "arcane": Style(
        "arcane", "Arcane", "the mystic default: fifths, a little unstable",
        1.0, 0.006, 1.0, (1.0, 1.5, 2.01, 3.0, 4.02), 7.0,
    ),
    # High, near-pure, a full consonant overtone series. Reads as blessing.
    "holy": Style(
        "holy", "Holy", "high and near-pure, a consonant overtone stack",
        1.26, 0.0018, 1.5, (1.0, 2.0, 3.0, 4.0, 5.0), 5.0,
    ),
    # Low and beating on a minor second and a tritone: the dissonant element.
    "shadow": Style(
        "shadow", "Shadow", "low and beating, minor-second and tritone",
        0.63, 0.016, 0.55, (1.0, 1.19, 1.41, 2.0), 4.0,
    ),
    # Fourth-based and woody, a touch flat: the earthy element.
    "nature": Style(
        "nature", "Nature", "fourth-based and woody, a touch earthbound",
        0.84, 0.009, 0.82, (1.0, 1.335, 2.0, 2.67, 3.34), 6.5,
    ),
}


def style_named(name: str) -> Style:
    try:
        return STYLES[name]
    except KeyError:
        raise ValueError(
            f"unknown style {name!r}; expected one of {sorted(STYLES)}"
        ) from None


# --------------------------------------------------------------------------
# small shared shapes
# --------------------------------------------------------------------------

def _tremolo(duration: float, rate: float, depth: float,
             phase: float = 0.0) -> list[float]:
    """A shimmer: amplitude wandering in ``[1 - depth, 1]`` at ``rate`` Hz."""

    return [
        (1.0 - depth) + depth * (0.5 + 0.5 * math.sin(TAU * rate * t + phase))
        for t in dsp.timebase(duration)
    ]


def _stack(freqs: list[float], style: Style, rng: random.Random,
           gains: Sequence[float] | None = None) -> list[float]:
    """A detuned harmonic stack following one per-sample frequency track.

    The whole element idea lives here: ``style.ratios`` colours the chord and
    ``style.detune`` spreads the voices so they beat. A cue passes the track it
    wants swept -- up for a crescent, down for a dissolve -- and gets the same
    element character over it.
    """

    n = len(freqs)
    out = [0.0] * n
    ratios = style.ratios
    for k, ratio in enumerate(ratios):
        gain = gains[k] if gains is not None and k < len(gains) else 1.0 / (k + 1)
        # spread the voices around the centre one so the stack beats
        detune = 1.0 + style.detune * (k - 1)
        track = [f * ratio * detune for f in freqs]
        voice = dsp.sine_track(track, phase=rng.uniform(0.0, TAU))
        for i in range(n):
            out[i] += voice[i] * gain
    return out


def _falling(duration: float, hi: float, lo: float, tau: float) -> list[float]:
    """An exponentially falling frequency track, for a streak sliding away."""

    return [lo + (hi - lo) * math.exp(-t / tau) for t in dsp.timebase(duration)]


def _air_streak(duration: float, freqs: list[float], q: float,
                decay: float, rng: random.Random,
                attack: float = 0.004) -> list[float]:
    """Focused, airy noise pushed through a moving bandpass: an energy streak.

    ``q`` is what makes it *focused* rather than a general whoosh -- a narrow
    band reads as one coherent streak of energy, a wide one as wind.
    """

    band = dsp.svf_bandpass(dsp.noise(duration, rng), freqs, q)
    return dsp.mul(band, dsp.exp_env(duration, decay, attack))


def _grain(freq: float, duration: float, rng: random.Random,
           ratio2: float = 2.76) -> list[float]:
    """One tiny pitched grain: a struck bell too small to ring, magic dust."""

    modes = [(freq, duration * 0.55, 1.0), (freq * ratio2, duration * 0.3, 0.4)]
    voice = dsp.modal_ring(duration, modes, rng)
    return dsp.mul(voice, dsp.exp_env(duration, duration * 0.4, 0.0008))


# --------------------------------------------------------------------------
# projectile cues
# --------------------------------------------------------------------------

def beam(rng: random.Random, style: Style,
         intensity: float = 0.85) -> list[float]:
    """The enchanted-sword projectile launch. Fantasy Adventure's ``combat/beam``.

    Three parts, in the order the brief names them:

    1. a bright steel resonance that *snaps* -- a struck-blade modal ring on
       high, inharmonic modes with a hard attack and a fast decay, fronted by a
       noise tick so it reads as a strike and not a bloom;
    2. a compact magical crescent -- a detuned harmonic stack whose pitch sweeps
       up and then eases, under a bell envelope that peaks early and releases.
       It rises and falls; that is the whole difference from a laser, which only
       falls, and the reason this is an enchantment and not a phaser;
    3. one short, focused, airy energy streak -- narrow-band noise through a
       falling filter, launched as the crescent lets go, sliding out and thin.

    ~0.48 s, which is where "compact" lands.
    """

    duration = 0.48
    bright = style.brightness

    # 1) bright steel resonance snaps -----------------------------------------
    root = 2600.0 * bright
    steel_modes = [
        (root * 1.00, 0.055, 1.00),
        (root * 1.63, 0.040, 0.72),
        (root * 2.47, 0.030, 0.50),
        (root * 3.51, 0.024, 0.34),
        (root * 4.90, 0.018, 0.22),
    ]
    steel = dsp.mul(dsp.modal_ring(duration, steel_modes, rng),
                    dsp.exp_env(duration, 0.05, attack=0.0015))
    tick = dsp.mul(
        dsp.highpass(dsp.noise(duration, rng), 4200.0 * bright, 2),
        [math.exp(-t / 0.0035) for t in dsp.timebase(duration)],
    )

    # 2) the magical crescent -------------------------------------------------
    peak_hz = (520.0 + 900.0 * intensity) * style.octave
    start_hz = 300.0 * style.octave
    rise = dsp.ramp(0.30, start_hz, peak_hz, curve=0.7)
    ease = dsp.ramp(duration - 0.30, peak_hz, peak_hz * 0.85, curve=1.5)
    crescent_track = rise + ease
    crescent = _stack(crescent_track, style, rng,
                      gains=(1.0, 0.5, 0.32, 0.2, 0.14))
    # peaks early (0.22 s) and releases: a swell that lets go, not a decay
    crescent = dsp.mul(crescent, dsp.bell_env(duration, 0.22, 0.14, 0.20))
    crescent = dsp.mul(crescent, _tremolo(duration, style.shimmer_hz, 0.16))

    # 3) the airy energy streak -----------------------------------------------
    streak = dsp.silence(duration)
    streak_len = 0.22
    streak_track = _falling(streak_len, 6400.0 * bright, 1300.0 * bright, 0.10)
    airy = _air_streak(streak_len, streak_track, q=4.5, decay=0.06, rng=rng)
    dsp.add_at(streak, 0.24, airy, 1.0)

    return dsp.mix(
        dsp.scale(steel, 0.85),
        dsp.scale(tick, 0.40),
        dsp.scale(crescent, 1.10),
        dsp.scale(streak, 0.55 + 0.25 * intensity),
    )


def bolt(rng: random.Random, style: Style,
         intensity: float = 0.85) -> list[float]:
    """A short offensive discharge: a zap. Crack, descending tone, buzz.

    The zap is fast and mostly downward -- an offensive release, the opposite of
    a charge. The buzz layer is a bandpassed noise chopped by a fast tremolo,
    which is what puts the *electric* in it rather than leaving a clean tone.
    """

    duration = 0.30
    bright = style.brightness

    crack = dsp.mul(
        dsp.highpass(dsp.noise(duration, rng), 2600.0 * bright, 4),
        [math.exp(-t / 0.006) for t in dsp.timebase(duration)],
    )
    track = dsp.ramp(duration, 1900.0 * style.octave, 320.0 * style.octave, 0.5)
    zap = _stack(track, style, rng, gains=(1.0, 0.4, 0.25))
    zap = dsp.mul(zap, dsp.exp_env(duration, 0.055, attack=0.001))

    buzz_freqs = _falling(duration, 3200.0 * bright, 700.0 * bright, 0.08)
    buzz = dsp.svf_bandpass(dsp.noise(duration, rng), buzz_freqs, q=3.0)
    buzz = dsp.mul(buzz, _tremolo(duration, 90.0 + 40.0 * intensity, 0.9))
    buzz = dsp.mul(buzz, dsp.exp_env(duration, 0.07))

    return dsp.mix(dsp.scale(crack, 0.5), dsp.scale(zap, 1.0),
                   dsp.scale(buzz, 0.35 + 0.2 * intensity))


def teleport(rng: random.Random, style: Style,
             intensity: float = 0.85) -> list[float]:
    """A warp: a whoosh with a pitch discontinuity where the thing relocates.

    The tell of a teleport is not the whoosh -- anything moving whooshes -- it
    is the *cut*: a pitched tone sweeps up, then jumps in one sample to a new
    frequency and continues. The jump is the moment the caster is somewhere
    else. A little transient is dropped on the seam so the cut lands.
    """

    duration = 0.55
    bright = style.brightness
    cut = 0.30

    # whoosh: a bandpass that rushes up and back down
    whoosh_freqs = dsp.ramp(duration, 300.0, 5200.0 * bright, 1.4)
    half = dsp.samples(duration) // 2
    for i in range(half, len(whoosh_freqs)):
        # fall away over the second half
        span = len(whoosh_freqs) - half
        whoosh_freqs[i] = whoosh_freqs[half] * (1.0 - 0.85 * (i - half) / span) + 260.0
    whoosh = dsp.svf_bandpass(dsp.noise(duration, rng), whoosh_freqs, q=1.3)
    whoosh = dsp.mul(whoosh, dsp.bell_env(duration, cut, 0.16, 0.16))

    # the pitched tone with a discontinuity at the cut
    before = dsp.ramp(cut, 420.0 * style.octave, 1500.0 * style.octave, 0.8)
    after = dsp.ramp(duration - cut, 900.0 * style.octave,
                     240.0 * style.octave, 0.9)
    tone_track = before + after
    tone = _stack(tone_track, style, rng, gains=(1.0, 0.4, 0.25))
    # energy builds to the cut (attack then hold), then decays after it
    tone_env = ([min(1.0, t / 0.05) for t in dsp.timebase(cut)]
                + [math.exp(-t / 0.16) for t in dsp.timebase(duration - cut)])
    tone = dsp.mul(tone, tone_env[:len(tone)])

    # a transient marking the cut
    spark = dsp.silence(duration)
    tick = dsp.mul(dsp.highpass(dsp.noise(0.05, rng), 3000.0 * bright, 2),
                   [math.exp(-t / 0.008) for t in dsp.timebase(0.05)])
    dsp.add_at(spark, cut, tick, 0.9)

    return dsp.mix(dsp.scale(whoosh, 0.7 + 0.2 * intensity),
                   dsp.scale(tone, 0.9), dsp.scale(spark, 0.5))


# --------------------------------------------------------------------------
# cast cues
# --------------------------------------------------------------------------

def cast(rng: random.Random, style: Style,
         intensity: float = 0.85) -> list[float]:
    """A spell being cast: a rising shimmer that resolves to a soft pop.

    The shimmer is a detuned stack sweeping up under a tremolo, with a sparkle
    of grains riding over it. It arrives at a soft pop -- a short, lowpassed
    tone-and-noise burst -- which is the spell taking hold. Soft on purpose: a
    hard pop reads as a hit, and this is a cast, not an impact.
    """

    duration = 0.60
    bright = style.brightness

    track = dsp.ramp(0.50, 260.0 * style.octave, 940.0 * style.octave, 1.4)
    track = track + [track[-1]] * (dsp.samples(duration) - len(track))
    shimmer = _stack(track, style, rng, gains=(1.0, 0.5, 0.3, 0.2))
    swell = [min(1.0, t / 0.42) ** 1.5 for t in dsp.timebase(duration)]
    shimmer = dsp.mul(shimmer, swell)
    shimmer = dsp.mul(shimmer, _tremolo(duration, style.shimmer_hz, 0.3))

    # sparkle grains, more of them as intensity rises
    grains = dsp.silence(duration)
    count = int(6 + 14 * intensity)
    for _ in range(count):
        onset = rng.uniform(0.0, 0.5)
        pitch = rng.uniform(2200.0, 6000.0) * bright * style.octave
        grain = _grain(pitch, 0.06, rng)
        dsp.add_at(grains, onset, grain, rng.uniform(0.2, 0.5))

    # the soft pop: it lands where the sweep does
    pop = dsp.silence(duration)
    pop_tone = dsp.mul(dsp.sine(720.0 * style.octave, 0.14),
                       dsp.exp_env(0.14, 0.05, attack=0.006))
    pop_noise = dsp.mul(dsp.lowpass(dsp.noise(0.14, rng), 1800.0 * bright, 2),
                        dsp.exp_env(0.14, 0.03, attack=0.004))
    dsp.add_at(pop, 0.48, dsp.mix(dsp.scale(pop_tone, 0.8),
                                  dsp.scale(pop_noise, 0.5)), 1.0)

    return dsp.mix(dsp.scale(shimmer, 0.9), dsp.scale(grains, 0.8),
                   dsp.scale(pop, 0.9))


def charge(rng: random.Random, style: Style,
           intensity: float = 0.85) -> list[float]:
    """Energy gathering: a rising harmonic tremolo that ends at its loudest.

    The cousin of combat_generators' ``energy_charge``, and it borrows that
    cue's one rule: it ends bright and full instead of decaying, because its
    whole job is to promise a release. What is magic rather than mechanical
    about it is the harmonic stack -- ``style.ratios`` under a tremolo -- where
    the energy cue used a plain drone.
    """

    duration = 0.72
    bright = style.brightness
    env = dsp.ramp(duration, 0.04, 1.0, curve=2.0)

    # air dragged inward: the band climbs and tightens with the envelope
    freqs = [600.0 + 5200.0 * bright * (e ** 1.7) for e in env]
    air = dsp.mul(dsp.svf_bandpass(dsp.noise(duration, rng), freqs, q=2.2), env)

    # the harmonic charge: the element's stack sweeping up together, and a
    # tremolo over all of it -- the "rising harmonic tremolo".
    track = dsp.ramp(duration, 150.0 * style.octave, 430.0 * style.octave, 1.5)
    drone = _stack(track, style, rng)
    drone = dsp.mul(drone, [e ** 1.2 for e in env])
    drone = dsp.mul(drone, _tremolo(duration, style.shimmer_hz, 0.4))

    # sparkle building, louder as it winds up
    grains = dsp.stick_slip(duration, rng, rate=46.0,
                            freq=6200.0 * bright, jitter=0.8)
    sparkle = dsp.mul(dsp.highpass(grains, 3200.0 * bright, 2),
                      [e ** 2.4 for e in env])

    return dsp.mix(dsp.scale(air, 0.7), dsp.scale(drone, 0.55),
                   dsp.scale(sparkle, 0.9 + 0.3 * intensity))


def heal(rng: random.Random, style: Style,
         intensity: float = 0.85) -> list[float]:
    """A gentle ascending chime-shimmer: restoration.

    A small ascending arpeggio of soft chimes -- consonant intervals of the
    element -- each with a slow attack so nothing strikes, and a shimmer of dust
    over the top. Warm and unhurried; the gentleness is the message.
    """

    duration = 0.90
    bright = style.brightness
    base = 523.0 * style.octave

    out = dsp.silence(duration)
    # an ascending run over the element's consonant ratios
    ratios = [r for r in style.ratios if r <= 5.5][:4] or [1.0, 1.5, 2.0]
    steps = len(ratios) + 1
    for k in range(steps):
        ratio = ratios[k] if k < len(ratios) else ratios[-1] * 1.5
        onset = 0.08 + 0.16 * k
        pitch = base * ratio
        modes = [(pitch, 0.55, 1.0), (pitch * 2.0, 0.35, 0.35),
                 (pitch * 3.01, 0.22, 0.16)]
        chime = dsp.mul(dsp.modal_ring(0.55, modes, rng),
                        dsp.exp_env(0.55, 0.24, attack=0.02))
        chime = dsp.mul(chime, _tremolo(0.55, style.shimmer_hz, 0.18))
        dsp.add_at(out, onset, chime, 0.6 - 0.05 * k)

    # a fine dust of grains over the run
    dust = dsp.silence(duration)
    count = int(5 + 8 * intensity)
    for _ in range(count):
        onset = rng.uniform(0.05, 0.75)
        pitch = rng.uniform(3000.0, 7000.0) * bright * style.octave
        grain = _grain(pitch, 0.05, rng)
        dsp.add_at(dust, onset, grain, rng.uniform(0.12, 0.3))

    return dsp.mix(dsp.scale(out, 1.0), dsp.scale(dust, 0.5))


def sparkle(rng: random.Random, style: Style,
            intensity: float = 0.85) -> list[float]:
    """A sprinkle of tiny pitched grains: magic dust.

    Nothing but grains -- many short, tuned pings scattered in time on a
    pentatonic-ish set drawn from the element, denser and a touch louder as
    intensity rises. No sustained tone, because dust does not hold a note.
    """

    duration = 0.60
    bright = style.brightness
    base = 1200.0 * bright * style.octave
    # a pleasant set of intervals to draw grain pitches from
    scale = (1.0, 1.125, 1.25, 1.5, 1.6875, 2.0, 2.25, 2.5, 3.0)

    out = dsp.silence(duration)
    count = int(14 + 30 * intensity)
    for _ in range(count):
        onset = rng.uniform(0.0, 0.52)
        step = rng.choice(scale) * rng.choice((1.0, 1.0, 2.0))
        grain = _grain(base * step, rng.uniform(0.03, 0.07), rng)
        dsp.add_at(out, onset, grain, rng.uniform(0.15, 0.55))
    return out


# --------------------------------------------------------------------------
# state cues
# --------------------------------------------------------------------------

def shield(rng: random.Random, style: Style,
           intensity: float = 0.85) -> list[float]:
    """A magical ward forming: a ring that settles into a hum.

    A bright shimmer snaps the ward into place, a bell-like ring sounds on the
    element's consonant modes, and as that ring decays a low harmonic hum fades
    up under it and sustains. The cross-fade from ring to hum is the "settling"
    -- the ward stops ringing and starts holding.

    The steady hum on its own, loopable, is :func:`shield_loop`.
    """

    duration = 1.00
    bright = style.brightness
    base = 320.0 * style.octave

    # the ring: consonant modes, medium decay
    ring_modes = []
    for k, ratio in enumerate(style.ratios):
        ring_modes.append((base * ratio, 0.45 - 0.05 * k, 1.0 / (k + 1)))
    ring = dsp.mul(dsp.modal_ring(duration, ring_modes, rng),
                   dsp.exp_env(duration, 0.35, attack=0.004))

    # a shimmer sweep snapping it into place at the front
    sweep_track = dsp.ramp(0.18, 900.0 * bright, 2600.0 * bright, 1.2)
    snap = _air_streak(0.18, sweep_track, q=1.6, decay=0.05, rng=rng)
    snap_full = dsp.silence(duration)
    dsp.add_at(snap_full, 0.0, snap, 1.0)

    # the hum fading up as the ring falls: low stack, gentle tremolo
    hum_track = [130.0 * style.octave] * dsp.samples(duration)
    hum = _stack(hum_track, style, rng)
    fade_up = [min(1.0, t / 0.55) for t in dsp.timebase(duration)]
    hum = dsp.mul(hum, fade_up)
    hum = dsp.mul(hum, _tremolo(duration, style.shimmer_hz * 0.5, 0.2))

    return dsp.mix(dsp.scale(ring, 0.9), dsp.scale(snap_full, 0.5),
                   dsp.scale(hum, 0.55 + 0.2 * intensity))


def shield_loop(rng: random.Random, seconds: float, style: Style,
                intensity: float = 0.85) -> list[float]:
    """The steady ward hum, as a seamless loop.

    The loopable variant of :func:`shield`: no forming transient, just the hum
    the ward holds -- a detuned harmonic drone under a slow shimmer, with an
    airy high bed breathing over it. Built entirely from
    ``environment_ambience_generator.loopdsp`` so it is seamless by
    construction; ``verify`` measures the join.
    """

    seconds = _check_seconds(seconds)
    bright = style.brightness
    root = 130.0 * style.octave

    # the drone: the element's stack, each voice quantised to whole cycles so
    # it arrives back at its starting phase exactly at the wrap
    drone = [0.0] * dsp.samples(seconds)
    for k, ratio in enumerate(style.ratios):
        gain = 1.0 / (k + 1)
        detune = 1.0 + style.detune * (k - 1)
        voice = loopdsp.sine(root * ratio * detune, seconds,
                             phase=rng.uniform(0.0, TAU), minimum=1)
        drone = [a + b * gain for a, b in zip(drone, voice)]
    trem = loopdsp.lfo(style.shimmer_hz * 0.35, seconds, depth=0.25 * intensity,
                       phase=rng.uniform(0.0, TAU))
    drone = [a * b for a, b in zip(drone, trem)]

    # the airy bed: high bandpassed noise, primed circularly, slowly breathing
    air = loopdsp.bandpass(dsp.noise(seconds, rng),
                           2400.0 * bright, 6800.0 * bright, 4)
    air_mod = loopdsp.lfo(0.6, seconds, depth=0.55, phase=rng.uniform(0.0, TAU))
    air = [a * b for a, b in zip(air, air_mod)]

    return dsp.mix(loopdsp.level(drone, 0.16),
                   loopdsp.level(air, 0.03 + 0.03 * intensity))


def dissolve(rng: random.Random, style: Style,
             intensity: float = 0.85) -> list[float]:
    """A short descending magical life-energy dissolve.

    Fantasy Adventure's ``player/death`` wants this as a layer: a downward pitched
    shimmer fading to nothing. A detuned stack slides down while a tremolo
    flutters over it, an airy band slides down with it, and a thinning dust
    trails off -- everything decaying to silence, which is the "to nothing".
    """

    duration = 0.72
    bright = style.brightness

    track = dsp.ramp(duration, 880.0 * style.octave, 150.0 * style.octave, 0.7)
    stack = _stack(track, style, rng, gains=(1.0, 0.5, 0.3, 0.2))
    stack = dsp.mul(stack, dsp.exp_env(duration, 0.28, attack=0.006))
    stack = dsp.mul(stack, _tremolo(duration, style.shimmer_hz, 0.35))

    air_track = _falling(duration, 5200.0 * bright, 600.0 * bright, 0.30)
    air = _air_streak(duration, air_track, q=2.0, decay=0.26, rng=rng)

    # dust thinning as the energy leaves
    dust = dsp.silence(duration)
    count = int(6 + 8 * intensity)
    for _ in range(count):
        onset = rng.uniform(0.0, 0.5)
        # grains fall in pitch the later they arrive
        pitch = (4200.0 - 5000.0 * onset) * bright * style.octave
        grain = _grain(max(500.0, pitch), 0.05, rng)
        dsp.add_at(dust, onset, grain, rng.uniform(0.1, 0.3) * (1.0 - onset))

    return dsp.mix(dsp.scale(stack, 1.0), dsp.scale(air, 0.5),
                   dsp.scale(dust, 0.4))


def curse(rng: random.Random, style: Style,
          intensity: float = 0.85) -> list[float]:
    """A dark descending detune: a debuff settling over the target.

    Two low voices a few Hz apart slide down together, and the beat between
    them slows and deepens as they descend -- that widening beat is the "detune"
    the brief asks for, and it is what makes it read as something going wrong
    rather than merely something low. A dissonant upper voice from the element
    and a dark noise wash sit under it. Everything lowpassed: a curse has no
    top end.
    """

    duration = 0.85
    bright = style.brightness
    base = dsp.ramp(duration, 300.0 * style.octave, 66.0 * style.octave, 0.8)

    spread = 0.02 + 0.04 * style.detune * 10.0  # audible beat, style-scaled
    voice_a = dsp.sine_track([f * (1.0 - spread) for f in base],
                             rng.uniform(0.0, TAU))
    voice_b = dsp.sine_track([f * (1.0 + spread) for f in base],
                             rng.uniform(0.0, TAU))
    # a dissonant upper partial of the element, quiet, for colour
    upper_ratio = style.ratios[1] if len(style.ratios) > 1 else 1.5
    upper = dsp.sine_track([f * upper_ratio for f in base], rng.uniform(0.0, TAU))

    tone = dsp.mix(dsp.scale(voice_a, 0.6), dsp.scale(voice_b, 0.6),
                   dsp.scale(upper, 0.18))
    tone = dsp.mul(tone, dsp.exp_env(duration, 0.45, attack=0.02))

    wash = dsp.mul(dsp.lowpass(dsp.noise(duration, rng), 900.0 * bright, 4),
                   dsp.exp_env(duration, 0.5, attack=0.03))

    out = dsp.mix(dsp.scale(tone, 1.0), dsp.scale(wash, 0.35 + 0.15 * intensity))
    # a final darkening so nothing bright survives
    return dsp.lowpass(out, 1600.0 * bright, 2)


# --------------------------------------------------------------------------
# validation, mastering and io
# --------------------------------------------------------------------------

def _check_seconds(seconds: float) -> float:
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise ValueError(
            f"seconds must be between {MIN_SECONDS} and {MAX_SECONDS}; "
            f"got {seconds}"
        )
    return seconds


def master(signal: list[float], peak_db: float, loop: bool = False) -> list[float]:
    """Final chain, shared by every cue.

    A loop routes to ``loopdsp.master`` -- ``dsp.master`` without the edge fades,
    because a fade to zero at the ends is itself a seam. One-shots take the
    ordinary master and want their fades.
    """

    if loop:
        return loopdsp.master(signal, peak_db)
    return dsp.master(signal, peak_db)
