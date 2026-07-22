"""Procedural environmental ambience synthesis.

Like ``weapons_destruction_generator`` this shares the standard-library DSP
core from ``combat_generators``, and adds the loop machinery in
:mod:`.loopdsp`. There are no recordings here, so nothing in this generator
carries a licence.

The split that matters is between beds and events. A bed is stationary noise
shaped by filters and slow modulation -- wind, rumble, hum -- and it is made
seamless by priming its filters circularly and quantising its modulators. An
event is a drip, a bubble, a bird, a clank: it is rendered with the ordinary
one-shot DSP and then placed with :func:`loopdsp.add_wrapped`, which sends
anything overhanging the end round to the start. An event never needs to be
periodic itself; the wrap is what makes the loop close.

Every renderer here takes ``(rng, seconds, intensity)`` and returns unmastered
mono floating-point PCM. The cue registry owns mastering, and chooses the
fade-free loop master for beds so the loop point survives it.

`intensity` is not a volume control -- mastering normalises that away
afterwards. It is how much is happening: breeze to gale, an empty room to a
packed one.
"""

from __future__ import annotations

from ..._audio.dsp import write_wav as write_wav

import math
import random

from ..._audio import dsp

from ..._audio import loops as loopdsp
from . import birds, wind


SR = dsp.SR
TAU = dsp.TAU

MIN_SECONDS = 0.25
MAX_SECONDS = 180.0


def _check_intensity(intensity: float) -> float:
    if not 0.05 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    return intensity


def _check_seconds(seconds: float) -> float:
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise ValueError(
            f"seconds must be between {MIN_SECONDS} and {MAX_SECONDS}"
        )
    return seconds


def _bed(seconds: float, rng: random.Random, low: float, high: float,
         order: int = 4) -> list[float]:
    """Band-limited stationary noise, primed so it loops."""

    return loopdsp.bandpass(dsp.noise(seconds, rng), low, high, order)


# The level helper lives in loopdsp so every texture module shares one
# definition of what a layer target means; see loopdsp.level.
_level = loopdsp.level


def _events(seconds: float, rng: random.Random, count: int,
            build) -> list[float]:
    """Scatter `count` rendered events across the loop, wrapping the tails."""

    out = dsp.silence(seconds)
    for onset in loopdsp.scatter(seconds, rng, count):
        event, gain = build(rng)
        loopdsp.add_wrapped(out, onset, event, gain)
    return out


def _per_loop(rate: float, seconds: float, minimum: int = 1) -> int:
    """Event count for a per-second rate, never rounding a texture to silence."""

    return max(minimum, int(round(rate * loopdsp.period(seconds))))


# --------------------------------------------------------------------------
# reusable events
# --------------------------------------------------------------------------

def bubble(rng: random.Random, freq: float,
           duration: float | None = None) -> list[float]:
    """One Minnaert bubble.

    A bubble is a spring: the gas is the compliance, the water around it is the
    mass, and it rings at roughly 3.26/radius Hz -- 3 kHz for a millimetre,
    650 Hz for five. It also rises in pitch as it goes, because the bubble is
    shrinking while it rings, and that rise is most of what makes it read as
    underwater rather than as a beep.
    """

    duration = duration or rng.uniform(0.018, 0.055)
    rise = rng.uniform(1.25, 2.30)
    freqs = dsp.ramp(duration, freq, freq * rise, 0.65)
    envelope = dsp.exp_env(duration, duration * 0.34, 0.0012)
    return dsp.mul(dsp.sine_track(freqs, rng.uniform(0.0, TAU)), envelope)


def drip_event(rng: random.Random, size: float = 1.0) -> list[float]:
    """A drop striking still water: contact tick, then the cavity ringing.

    The tick alone is a stone on a floor. What says *water* is the pitched
    rise straight after it, which is the cavity the drop punched closing again.
    """

    duration = 0.11 + 0.05 * size
    tick = dsp.mul(
        dsp.bandpass(dsp.noise(duration, rng), 2_400.0, 13_000.0, 2),
        dsp.exp_env(duration, 0.0035),
    )
    voice = bubble(rng, rng.uniform(1_500.0, 3_400.0) / max(size, 0.35),
                   duration * 0.62)
    return dsp.mix(dsp.scale(tick, 0.34), dsp.scale(voice, 0.85))


def stone_hit(rng: random.Random, size: float = 1.0) -> list[float]:
    """A rock landing: contact, inharmonic ring, and shed grit."""

    duration = 0.16 + 0.22 * size
    scale = 1.0 / math.sqrt(max(size, 0.25))
    contact = dsp.mul(
        dsp.bandpass(dsp.noise(duration, rng), 600.0, 9_000.0, 2),
        dsp.exp_env(duration, 0.008),
    )
    ring = dsp.modal_ring(
        duration,
        [(610.0 * scale * rng.uniform(0.85, 1.2), 0.055 * size, 0.5),
         (1_370.0 * scale * rng.uniform(0.85, 1.2), 0.038 * size, 0.3),
         (2_480.0 * scale * rng.uniform(0.85, 1.2), 0.022 * size, 0.16)],
        rng,
    )
    body = dsp.mul(
        dsp.lowpass(dsp.noise(duration, rng), 220.0 * scale, 4),
        dsp.exp_env(duration, 0.05 * size, 0.001),
    )
    return dsp.mix(
        dsp.scale(contact, 0.5), ring, dsp.scale(body, 0.6 * size),
    )


def chirp(rng: random.Random) -> list[float]:
    """A bird: two or three whistled notes, each sliding.

    Birdsong is close to a pure tone with a fast glide. Noise in the voice is
    what makes a synthetic bird sound like a kettle, so there is none here
    beyond the breath transient.
    """

    notes = rng.randint(2, 4)
    gap = rng.uniform(0.055, 0.11)
    span = notes * gap + 0.12
    out = dsp.silence(span)
    base = rng.uniform(2_100.0, 4_600.0)
    for index in range(notes):
        duration = rng.uniform(0.035, 0.075)
        start = base * rng.uniform(0.88, 1.14)
        end = start * rng.uniform(0.82, 1.28)
        note = dsp.mul(
            dsp.sine_track(dsp.ramp(duration, start, end, 0.7),
                           rng.uniform(0.0, TAU)),
            dsp.bell_env(duration, duration * 0.42, duration * 0.24,
                         duration * 0.34),
        )
        dsp.add_at(out, index * gap, note, rng.uniform(0.6, 1.0))
    return out


def syllable(rng: random.Random, duration: float | None = None) -> list[float]:
    """One unvoiced syllable's worth of energy, before any formant shaping.

    Speech at a distance is not words, it is the rhythm of syllables. Get the
    envelope right and the formant bank downstream does the rest.
    """

    duration = duration or rng.uniform(0.07, 0.19)
    source = dsp.bandpass(dsp.noise(duration, rng), 90.0, 3_200.0, 2)
    envelope = dsp.bell_env(duration, duration * 0.4, duration * 0.28,
                            duration * 0.42)
    return dsp.mul(source, envelope)


def clank(rng: random.Random, freq: float = 480.0) -> list[float]:
    """Loose metal: a struck, inharmonic, quickly damped ring."""

    duration = rng.uniform(0.18, 0.45)
    ratios = (1.0, 2.41, 3.83, 5.17)
    modes = [
        (freq * ratio * rng.uniform(0.95, 1.06),
         duration * 0.30 / (1.0 + 0.5 * index), 0.62 / (1.0 + index))
        for index, ratio in enumerate(ratios)
    ]
    ring = dsp.modal_ring(duration, modes, rng)
    contact = dsp.mul(
        dsp.bandpass(dsp.noise(duration, rng), 1_500.0, 11_000.0, 2),
        dsp.exp_env(duration, 0.004),
    )
    return dsp.mix(ring, dsp.scale(contact, 0.4))


# --------------------------------------------------------------------------
# interiors
# --------------------------------------------------------------------------

def cave(rng: random.Random, seconds: float, intensity: float) -> list[float]:
    """A big wet hole in the rock.

    Caves are mostly defined by what is *not* there: no wind to speak of, no
    high end, and long gaps. The drips are reverberated but the rumble is not,
    because the rumble is already diffuse -- it has no transient for a room to
    smear, and paying for a second reverb pass to prove that is waste.
    """

    _check_seconds(seconds)
    _check_intensity(intensity)
    rumble = dsp.mul(
        loopdsp.lowpass(dsp.noise(seconds, rng), 105.0, 4),
        loopdsp.drift(seconds, (0.031, 0.077), rng, 0.45),
    )
    air = dsp.mul(
        _bed(seconds, rng, 180.0, 1_100.0, 2),
        loopdsp.drift(seconds, (0.053, 0.13), rng, 0.55),
    )
    drips = _events(
        seconds, rng, _per_loop(0.35 + 1.4 * intensity, seconds),
        lambda r: (drip_event(r, r.uniform(0.6, 1.4)), r.uniform(0.35, 1.0)),
    )
    wet = loopdsp.reverb(drips, room=0.86, damp=0.30)
    return dsp.mix(
        _level(rumble, 0.30),
        _level(air, 0.05 + 0.06 * intensity),
        dsp.scale(drips, 0.5),
        dsp.scale(wet, 1.5),
    )


def dungeon(rng: random.Random, seconds: float,
            intensity: float) -> list[float]:
    """A cave someone built, and left something in.

    The difference from `cave` is intent: a drone that sits a minor second
    apart so it never settles, iron that moves when nothing should be moving,
    and a longer, squarer room around it.
    """

    _check_seconds(seconds)
    _check_intensity(intensity)
    rumble = dsp.mul(
        loopdsp.lowpass(dsp.noise(seconds, rng), 95.0, 4),
        loopdsp.drift(seconds, (0.029, 0.067), rng, 0.4),
    )
    drone = dsp.silence(seconds)
    root = 54.0
    for freq, gain in ((root, 1.0), (root * 1.0595, 0.55), (root * 2.0, 0.3)):
        part = loopdsp.sine(freq, seconds, rng.uniform(0.0, TAU))
        drone = dsp.mix(drone, dsp.scale(part, gain))
    drone = dsp.mul(drone, loopdsp.drift(seconds, (0.041, 0.11), rng, 0.5))

    groan = dsp.mul(
        loopdsp.resonator(dsp.noise(seconds, rng), 185.0, q=17.0),
        loopdsp.drift(seconds, (0.023, 0.061), rng, 0.85, curve=1.6),
    )
    events = _events(
        seconds, rng, _per_loop(0.3 + 1.1 * intensity, seconds),
        lambda r: (drip_event(r, r.uniform(0.7, 1.5)), r.uniform(0.3, 0.9)),
    )
    iron = _events(
        seconds, rng, _per_loop(0.2 + 0.8 * intensity, seconds),
        lambda r: (clank(r, r.uniform(300.0, 700.0)), r.uniform(0.06, 0.22)),
    )
    room = loopdsp.reverb(dsp.mix(events, iron), room=0.90, damp=0.24)
    return dsp.mix(
        _level(rumble, 0.26),
        _level(drone, 0.14),
        _level(groan, 0.04 + 0.08 * intensity),
        dsp.scale(events, 0.4),
        dsp.scale(iron, 0.5),
        dsp.scale(room, 1.6),
    )


# --------------------------------------------------------------------------
# life
# --------------------------------------------------------------------------

def forest(rng: random.Random, seconds: float, intensity: float,
           foliage: float = 0.8) -> list[float]:
    """Daytime woodland: leaves, insects, and birds.

    The leaves come from :func:`wind.foliage_bed` -- two mechanisms, broadleaf
    flap and conifer hiss, on fixed bands -- rather than one gust-tracked
    filter, and the birds are real species from :mod:`.birds` rather than the
    whistled glides that used to stand in for them.

    The insects sit at a fixed rate because cicadas genuinely do, and the birds
    sing in bouts and then stop, because a bird every second is an aviary.
    """

    _check_seconds(seconds)
    _check_intensity(intensity)
    if not 0.0 <= foliage <= 1.0:
        raise ValueError("foliage must be between 0 and 1")

    # A wood is not still, but nor is it a gale: the trees answer a light wind.
    breeze = wind.gust_envelope(seconds, rng, 0.35)
    leaves = wind.foliage_bed(seconds, rng, breeze, foliage)
    under = dsp.mul(
        loopdsp.lowpass(dsp.noise(seconds, rng), 320.0, 4),
        [0.5 + 0.5 * value for value in breeze],
    )
    insects = dsp.mul(
        loopdsp.resonator(dsp.noise(seconds, rng), 4_900.0, q=7.0),
        [value ** 3 for value in loopdsp.lfo(23.0, seconds, 1.0)],
    )
    song = birds.chorus(rng, seconds, intensity)
    parts = [
        _level(under, 0.14),
        _level(insects, 0.02 + 0.06 * intensity),
        dsp.scale(loopdsp.reverb(song, room=0.72, damp=0.6), 0.4),
        dsp.scale(song, 0.9),
    ]
    if max((abs(v) for v in leaves), default=0.0) > 1e-9:
        parts.append(_level(leaves, 0.22))
    return dsp.mix(*parts)


def crowd(rng: random.Random, seconds: float,
          intensity: float) -> list[float]:
    """Many people talking, none of them close enough to understand.

    Babble is syllable rhythm plus formants. The syllables are scattered noise
    grains; the formants are two resonator banks -- one lower, one higher --
    because a crowd through a single bank is one enormous person.
    """

    _check_seconds(seconds)
    _check_intensity(intensity)
    voices = _per_loop(6 + 26 * intensity, seconds)

    def bank(count: int, formants: tuple[tuple[float, float], ...],
             ) -> list[float]:
        grains = _events(
            seconds, rng, count,
            lambda r: (syllable(r), r.uniform(0.4, 1.0)),
        )
        out = dsp.silence(seconds)
        for freq, gain in formants:
            shaped = loopdsp.resonator(
                grains, freq * rng.uniform(0.94, 1.07), q=6.0
            )
            out = dsp.mix(out, dsp.scale(shaped, gain))
        return out

    lower = bank(voices, ((540.0, 1.0), (1_150.0, 0.6), (2_500.0, 0.22)))
    upper = bank(max(1, voices // 2),
                 ((760.0, 0.8), (1_900.0, 0.5), (3_100.0, 0.18)))
    murmur = dsp.mul(
        loopdsp.lowpass(dsp.noise(seconds, rng), 480.0, 4),
        loopdsp.drift(seconds, (0.11, 0.27), rng, 0.35),
    )
    # The voice banks keep an intensity-tracking target rather than a fixed
    # one: a packed room is louder than an empty one, and pinning them would
    # normalise that away.
    lower = _level(lower, 0.12 + 0.20 * intensity)
    upper = _level(upper, 0.07 + 0.12 * intensity)
    room = loopdsp.reverb(dsp.mix(lower, upper), room=0.78, damp=0.5)
    return dsp.mix(
        lower,
        upper,
        _level(murmur, 0.05 + 0.08 * intensity),
        dsp.scale(room, 0.8),
    )


# --------------------------------------------------------------------------
# machines
# --------------------------------------------------------------------------

def mechanical(rng: random.Random, seconds: float,
               intensity: float) -> list[float]:
    """Machinery running: motor hum, blade pass, bearing whine, and rattle.

    A motor's pitch is set by how fast it turns, so the blade-pass thrum and
    the hum harmonics are locked to one rotation rate rather than chosen
    independently. Quantising that rate to the loop is what stops the thrum
    from stuttering at the wrap.
    """

    _check_seconds(seconds)
    _check_intensity(intensity)
    rotation = loopdsp.quantize(23.5, seconds, minimum=1)

    hum = dsp.silence(seconds)
    for harmonic, gain in ((2, 1.0), (4, 0.5), (6, 0.28), (9, 0.12)):
        part = loopdsp.sine(rotation * harmonic, seconds,
                            rng.uniform(0.0, TAU))
        hum = dsp.mix(hum, dsp.scale(part, gain))

    blades = [value ** 4 for value in loopdsp.lfo(rotation * 3, seconds, 1.0)]
    thrum = dsp.mul(loopdsp.lowpass(dsp.noise(seconds, rng), 900.0, 2), blades)
    whine = loopdsp.resonator(dsp.noise(seconds, rng), 3_150.0, q=22.0)
    broadband = _bed(seconds, rng, 260.0, 4_200.0, 2)
    rattle = _events(
        seconds, rng, _per_loop(1.5 + 7 * intensity, seconds),
        lambda r: (clank(r, r.uniform(600.0, 1_500.0)), r.uniform(0.03, 0.11)),
    )
    return dsp.mix(
        _level(hum, 0.26),
        _level(thrum, 0.18),
        _level(whine, 0.02 + 0.07 * intensity),
        _level(broadband, 0.04 + 0.08 * intensity),
        dsp.scale(rattle, 0.5),
    )


def space(rng: random.Random, seconds: float,
          intensity: float) -> list[float]:
    """A ship's interior: life support, a deep drone, and hull noise.

    Space itself is silent, so this is the room tone of the thing keeping you
    alive. The drone is detuned against itself to beat slowly; note that in a
    loop shorter than a few seconds the frequency grid is coarser than the
    detune, so the beat flattens out -- a slow beat needs a long loop to exist
    in, which is physics rather than a shortcut.
    """

    _check_seconds(seconds)
    _check_intensity(intensity)
    drone = dsp.silence(seconds)
    for freq, gain in ((38.0, 1.0), (38.6, 0.7), (57.0, 0.35), (76.4, 0.18)):
        part = loopdsp.sine(freq, seconds, rng.uniform(0.0, TAU))
        drone = dsp.mix(drone, dsp.scale(part, gain))
    drone = dsp.mul(drone, loopdsp.lfo(0.05, seconds, 0.2))

    life_support = dsp.mul(
        _bed(seconds, rng, 380.0, 5_800.0, 2),
        loopdsp.drift(seconds, (0.047, 0.12), rng, 0.4),
    )
    shimmer = dsp.mul(
        loopdsp.highpass(dsp.noise(seconds, rng), 6_500.0, 2),
        loopdsp.drift(seconds, (0.083, 0.19), rng, 0.6),
    )
    hull = dsp.mul(
        loopdsp.resonator(dsp.noise(seconds, rng), 128.0, q=19.0),
        loopdsp.drift(seconds, (0.019, 0.053), rng, 0.8, curve=1.5),
    )
    return dsp.mix(
        _level(drone, 0.30),
        _level(life_support, 0.06 + 0.10 * intensity),
        _level(shimmer, 0.015 + 0.03 * intensity),
        _level(hull, 0.05),
    )


# --------------------------------------------------------------------------
# one-shots
# --------------------------------------------------------------------------

def thunder(rng: random.Random, seconds: float,
            intensity: float) -> list[float]:
    """Thunder. `intensity` is proximity, not volume.

    Distance is the whole model. Air absorbs high frequencies with range, so a
    far strike has no crack left -- only the low rumble survives the trip, and
    it arrives smeared across seconds because the sound left every point of a
    kilometres-long channel at once and those points are not equidistant from
    you. Close, that spread collapses and you get the crack.
    """

    _check_intensity(intensity)
    near = intensity
    duration = 6.5 - 3.0 * near

    out = dsp.silence(duration)
    crack_high = 900.0 + 11_000.0 * near ** 2
    crack = dsp.mul(
        dsp.bandpass(dsp.noise(0.5, rng), 120.0, crack_high, 2),
        dsp.exp_env(0.5, 0.02 + 0.05 * (1.0 - near), 0.0015),
    )
    dsp.add_at(out, 0.0, crack, 0.35 + 0.65 * near)

    rolls = 7 + round(9 * (1.0 - near))
    cutoff = 260.0 + 900.0 * near
    for index in range(rolls):
        span = rng.uniform(0.4, 1.3)
        onset = rng.random() ** 0.7 * duration * 0.8
        roll = dsp.mul(
            dsp.lowpass(dsp.noise(span, rng), cutoff * rng.uniform(0.6, 1.4), 4),
            dsp.bell_env(span, span * 0.4, span * 0.3, span * 0.5),
        )
        fade = 1.0 - 0.55 * onset / duration
        dsp.add_at(out, onset, roll, rng.uniform(0.35, 1.0) * fade)

    sub = dsp.mul(
        dsp.lowpass(dsp.noise(duration, rng), 70.0, 4),
        dsp.bell_env(duration, duration * 0.22, duration * 0.16,
                     duration * 0.55),
    )
    out = dsp.mix(out, dsp.scale(sub, 0.7 + 0.5 * near))
    return dsp.mix(out, dsp.scale(loopdsp.tail(out, room=0.8, damp=0.65), 0.5))


def rockfall(rng: random.Random, seconds: float,
             intensity: float) -> list[float]:
    """Rock coming down. `intensity` is how much of it.

    A rockfall is not a volley of identical impacts: it starts with a crack,
    peaks quickly, and tails off in gravel long after the big pieces have
    stopped. The size distribution is skewed small for the same reason.
    """

    _check_intensity(intensity)
    duration = 1.8 + 1.6 * intensity
    out = dsp.silence(duration)

    crack = dsp.mul(
        dsp.bandpass(dsp.noise(0.3, rng), 500.0, 9_000.0, 2),
        dsp.exp_env(0.3, 0.012, 0.001),
    )
    dsp.add_at(out, 0.0, crack, 0.5)

    for _ in range(10 + round(34 * intensity)):
        onset = rng.random() ** 0.55 * duration * 0.86
        size = 0.15 + 1.5 * rng.random() ** 2.4 * intensity
        fade = 1.0 - 0.5 * onset / duration
        dsp.add_at(out, onset, stone_hit(rng, size),
                   rng.uniform(0.25, 0.8) * fade)

    grit = dsp.bandpass(
        dsp.mul(
            dsp.noise(duration, rng),
            dsp.bell_env(duration, duration * 0.3, duration * 0.2,
                         duration * 0.6),
        ),
        600.0, 7_000.0, 2,
    )
    rumble = dsp.mul(
        dsp.lowpass(dsp.noise(duration, rng), 130.0, 4),
        dsp.bell_env(duration, duration * 0.28, duration * 0.18,
                     duration * 0.5),
    )
    return dsp.mix(
        out, dsp.scale(grit, 0.35), dsp.scale(rumble, 0.55 * intensity),
    )
