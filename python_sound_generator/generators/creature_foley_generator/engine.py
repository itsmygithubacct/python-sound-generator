"""Seeded creature Foley built from real CC0 recordings.

This is a hybrid generator in the same lineage as ``door_sound_generator``:
NumPy and SciPy select, retune, filter, and layer bundled recordings rather
than synthesizing vocal cords from oscillators.  Procedural DSP can do a
convincing wing beat or shell crack, but a growl is a chaotic glottal source in
a fleshy resonator, and a synthesized one reliably sounds synthesized.  The
recordings and their licensing live in ``sources/provenance.json``.

Creature size is varispeed.  Resampling a vocalization moves its pitch and its
formants together, which is what actually distinguishes a small throat from a
large one -- pitch-shifting alone leaves the formants behind and turns a big
animal into a slowed-down small one.  That is also why size is applied to the
recording rather than to a filter afterwards.

All renderers return unmastered mono floating-point PCM plus the source files
they consumed.  The cue registry owns mastering levels, so every public render
follows the same contract, and the consumed filenames reach the manifest.
"""

from __future__ import annotations

from ..._audio.assets import resolve_asset

from ..._audio.foley_numpy import read_recording

from ..._audio.foley_numpy import write_wav as write_wav

from ..._audio.foley_numpy import trim_activity as trim_recording_activity

from ..._audio.foley_numpy import fade_edges, mix_at_frames, varispeed_configured


import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt

SR = 44_100
SOURCE_DIRECTORY = Path(__file__).resolve().parent / "sources"


@dataclass(frozen=True)
class SizeProfile:
    """How body mass reads acoustically, mostly as varispeed and low-end."""

    name: str
    label: str
    description: str
    speed: float
    weight: float
    wing_rate: float
    duration_scale: float


@dataclass(frozen=True)
class CharacterProfile:
    """Source-pool bias and spectral tilt for one kind of creature."""

    name: str
    label: str
    description: str
    speed_bias: float
    warmth: float
    brightness: float
    wet: float
    air: float
    pools: dict[str, tuple[str, ...]] = field(default_factory=dict)


SIZES: dict[str, SizeProfile] = {
    "tiny": SizeProfile(
        "tiny", "Tiny", "insect, hatchling, or fish", 1.92, 0.16, 15.0, 0.55
    ),
    "small": SizeProfile(
        "small", "Small", "bird, cat, or small lizard", 1.40, 0.40, 8.0, 0.76
    ),
    "medium": SizeProfile(
        "medium", "Medium", "wolf, raptor, or man-sized beast", 1.00, 1.00, 4.2, 1.00
    ),
    "large": SizeProfile(
        "large", "Large", "bear, ogre, or mount", 0.74, 1.90, 2.3, 1.32
    ),
    "huge": SizeProfile(
        "huge", "Huge", "dragon, giant, or leviathan", 0.52, 3.30, 1.3, 1.68
    ),
}


def _names(prefix: str, *stems: str) -> tuple[str, ...]:
    return tuple(f"{prefix}{stem}.wav" for stem in stems)


_CREATURE = "rubberduck_creature_"
_BREAK = "rubberduck_break_"
_RPG = "artisticdude_"

CHIRP_SMALL = _names(_CREATURE, "cute_01", "cute_03", "cute_05", "cute_08")
CHIRP_BUG = _names(_CREATURE, "bug_01", "bug_03", "bug_05", "bug_09")
GROWL_BEAST = _names(
    _CREATURE, "monster_01", "monster_03", "monster_06", "monster_08",
    "monster_12", "monster_18",
) + _names(_RPG, "mnstr1", "mnstr3", "mnstr5", "mnstr7", "mnstr9", "mnstr12")
GROWL_BIG = _names(_CREATURE, "troll_01", "troll_03") + _names(_RPG, "ogre1", "ogre3")
GROWL_WET = _names(
    _CREATURE, "burble_01", "burble_02", "slime_01", "slime_05"
) + _names(_RPG, "slime1", "slime5")
GROWL_SPECTRAL = _names(_CREATURE, "alien_08", "alien_12") + _names(
    _RPG, "shade1", "shade7"
)
ROAR = _names(_CREATURE, "roar_01", "roar_02", "roar_03", "roar_04", "roar_05")
ROAR_BIG = _names(_RPG, "giant1", "giant2", "giant4")
SCREECH = _names(_CREATURE, "scream_01", "scream_02", "roar_03")
HOWL = _names(_CREATURE, "howl") + _names(_RPG, "wolfman")
PAIN = _names(_CREATURE, "hurt_01", "hurt_02", "hurt_03", "hurt_05")
DEFEAT = _names(_CREATURE, "die_01", "die_02", "die_03", "die_04")
BREATH = _names(_CREATURE, "breath", "breath_02", "snore", "snore_02")
GRUNT = _names(_CREATURE, "grunt_01", "grunt_04", "grunt_06")
BITE = _names(_CREATURE, "attack_01", "attack_02", "attack_03", "attack_05")
CHEW = _names(_CREATURE, "eat_01", "eat_03")
BITE_SMALL = _names(_RPG, "bite_small_01", "bite_small_02", "bite_small_03")
CLOTH = _names(_RPG, "cloth", "cloth_heavy", "armor_light", "chainmail")
STOMP = _names(_CREATURE, "stomp_01")
SHELL = _names(
    _BREAK, "wood_breaking_01", "wood_breaking_02", "wood_breaking_03",
    "rock_breaking_01", "rock_breaking_02", "glass_breaking_01",
    "glass_breaking_03",
)
DEBRIS = _names(_BREAK, "rock_falling_01", "rock_falling_03", "wood_falling_01")
IMPACT = _names(_BREAK, "hit_01", "hit_03", "hit_05", "wood_hit_01", "wood_hit_02")
WINGS = _names("antum_", "wings_flap_large")


DEFAULT_POOLS: dict[str, tuple[str, ...]] = {
    "chirp": CHIRP_SMALL,
    "purr": BREATH,
    "growl": GROWL_BEAST,
    "roar": ROAR + ROAR_BIG,
    "pain": PAIN,
    "defeat": DEFEAT,
    "bite": BITE + BITE_SMALL,
}


CHARACTERS: dict[str, CharacterProfile] = {
    "beast": CharacterProfile(
        "beast", "Beast", "fur, flesh, and a deep chest", 1.00, 0.14, -0.02,
        0.04, 0.20,
        {"growl": GROWL_BEAST + GROWL_BIG, "roar": ROAR + ROAR_BIG + HOWL},
    ),
    "bird": CharacterProfile(
        "bird", "Bird", "light, bright, and quick", 1.52, -0.12, 0.30, 0.0, 0.10,
        {
            "chirp": CHIRP_SMALL,
            "purr": CHIRP_SMALL + _names(_CREATURE, "breath"),
            "growl": SCREECH,
            "roar": SCREECH,
            "bite": BITE_SMALL,
        },
    ),
    "insect": CharacterProfile(
        "insect", "Insect", "chitinous, dry, and buzzing", 1.68, -0.20, 0.36,
        0.0, 0.05,
        {
            "chirp": CHIRP_BUG,
            "purr": CHIRP_BUG,
            "growl": CHIRP_BUG,
            "roar": CHIRP_BUG + GROWL_SPECTRAL,
            "pain": CHIRP_BUG + PAIN,
            "defeat": CHIRP_BUG + DEFEAT,
            "bite": BITE_SMALL,
        },
    ),
    "reptile": CharacterProfile(
        "reptile", "Reptile", "dry, hissing, and cold-blooded", 0.94, -0.05,
        0.16, 0.0, 0.38,
        {
            "chirp": CHIRP_BUG + CHIRP_SMALL,
            "purr": _names(_CREATURE, "snore", "snore_02", "breath_02"),
            "growl": GROWL_BEAST,
            "roar": ROAR + ROAR_BIG,
        },
    ),
    "slime": CharacterProfile(
        "slime", "Slime", "wet, boneless, and gurgling", 0.88, 0.08, -0.14,
        0.85, 0.12,
        {
            "chirp": GROWL_WET,
            "purr": GROWL_WET,
            "growl": GROWL_WET,
            "roar": GROWL_WET + ROAR,
            "pain": GROWL_WET + PAIN,
            "defeat": GROWL_WET + DEFEAT,
            "bite": CHEW + GROWL_WET,
        },
    ),
    "spectral": CharacterProfile(
        "spectral", "Spectral", "airy, unnatural, and hollow", 0.82, -0.16, 0.22,
        0.0, 0.55,
        {
            "chirp": GROWL_SPECTRAL,
            "purr": GROWL_SPECTRAL + _names(_CREATURE, "breath_02"),
            "growl": GROWL_SPECTRAL,
            "roar": GROWL_SPECTRAL + ROAR,
            "pain": GROWL_SPECTRAL + PAIN,
            "defeat": GROWL_SPECTRAL + DEFEAT,
        },
    ),
}


def _pool(cue: str, character: CharacterProfile) -> tuple[str, ...]:
    return character.pools.get(cue) or DEFAULT_POOLS[cue]




def _trim_activity(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    return trim_recording_activity(
            signal, sample_rate=sample_rate, window_seconds=0.012,
            threshold_ratio=0.018, threshold_floor_db=-58.0,
            pre_seconds=0.014, post_seconds=0.014,
        )


@lru_cache(maxsize=128)
def _load_source(filename: str) -> np.ndarray:
    path = resolve_asset(SOURCE_DIRECTORY / filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing bundled creature source {path}; see sources/provenance.json"
        )
    _, signal = read_recording(
        path, expected_rate=SR, trim=lambda data, rate: _trim_activity(data, rate),
        label="creature source",
    )
    return signal


def _varispeed(signal: np.ndarray, speed: float) -> np.ndarray:
    return varispeed_configured(
        signal, speed, minimum=0.25, maximum=4.0, denominator=512
    )


def _tilt(signal: np.ndarray, warmth: float, brightness: float) -> np.ndarray:
    if abs(warmth) < 1e-6 and abs(brightness) < 1e-6:
        return signal
    low = sosfilt(
        butter(2, 320.0, btype="lowpass", fs=SR, output="sos"), signal
    )
    high = signal - sosfilt(
        butter(2, 2_600.0, btype="lowpass", fs=SR, output="sos"), signal
    )
    return signal + warmth * low + brightness * high


def _body(signal: np.ndarray) -> np.ndarray:
    """Low band of the recording itself, used to add mass without synthesis."""

    result = sosfilt(butter(2, 38.0, btype="highpass", fs=SR, output="sos"), signal)
    return sosfilt(butter(3, 340.0, btype="lowpass", fs=SR, output="sos"), result)


def _fades(signal: np.ndarray, seconds: float=0.004) -> np.ndarray:
    return fade_edges(signal, seconds, sample_rate=SR)


def _prepare(
    filename: str,
    size: SizeProfile,
    character: CharacterProfile,
    rng: np.random.Generator,
    speed_multiplier: float = 1.0,
) -> np.ndarray:
    """Load one recording retuned and tilted for this creature."""

    cached = _load_source(filename)
    speed = size.speed * character.speed_bias * speed_multiplier
    signal = _varispeed(cached, speed * float(rng.uniform(0.972, 1.029)))
    signal = _tilt(signal, character.warmth, character.brightness)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1e-9:
        signal = signal / peak
    return _fades(signal)


def _choose(rng: np.random.Generator, choices: Sequence[str]) -> str:
    return choices[int(rng.integers(0, len(choices)))]


def _mix_at(destination: np.ndarray, source: np.ndarray, start_frame: int, gain: float) -> None:
    mix_at_frames(destination, source, start_frame, gain)


def _fit(signal: np.ndarray, maximum_frames: int) -> np.ndarray:
    """Trim rather than resample, so a fitted vocal keeps its retuned pitch."""

    if len(signal) <= maximum_frames or maximum_frames < 2:
        return signal.copy()
    return _fades(signal[:maximum_frames])


@lru_cache(maxsize=4)
def _flap_segments(filename: str) -> tuple[tuple[int, int], ...]:
    """Split the multi-flap wing recording into individual beats.

    The bundled flap holds three beats in one file, so a single wing beat has
    to be cut out of it before it can be re-timed to a wing rate.
    """

    signal = _load_source(filename)
    envelope = np.sqrt(uniform_filter1d(signal * signal, max(3, round(0.008 * SR))))
    threshold = float(np.max(envelope)) * 0.16
    active = envelope >= threshold
    segments: list[tuple[int, int]] = []
    start: int | None = None
    gap = round(0.060 * SR)
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start > round(0.010 * SR):
                segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(signal)))

    merged: list[list[int]] = []
    for first, last in segments:
        if merged and first - merged[-1][1] < gap:
            merged[-1][1] = last
        else:
            merged.append([first, last])
    pad = round(0.012 * SR)
    return tuple(
        (max(0, first - pad), min(len(signal), last + pad)) for first, last in merged
    )


def _single_flap(
    size: SizeProfile,
    character: CharacterProfile,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    filename = _choose(rng, WINGS)
    segments = _flap_segments(filename)
    if not segments:
        return _prepare(filename, size, character, rng), filename
    first, last = segments[int(rng.integers(0, len(segments)))]
    cached = _load_source(filename)[first:last]
    speed = size.speed * character.speed_bias * float(rng.uniform(0.94, 1.07))
    flap = _varispeed(cached, speed)
    flap = _tilt(flap, character.warmth, character.brightness)
    peak = float(np.max(np.abs(flap))) if len(flap) else 0.0
    if peak > 1e-9:
        flap = flap / peak
    return _fades(flap, 0.006), filename


def _vocal(
    cue: str,
    size: SizeProfile,
    character: CharacterProfile,
    frames: int,
    intensity: float,
    rng: np.random.Generator,
    speed_multiplier: float = 1.0,
) -> tuple[np.ndarray, list[str]]:
    """Shared body of every recorded vocalization."""

    output = np.zeros(frames, dtype=float)
    name = _choose(rng, _pool(cue, character))
    primary = _prepare(name, size, character, rng, speed_multiplier)
    primary = _fit(primary, max(1, frames - round(0.03 * SR)))
    start = round(0.012 * SR)
    _mix_at(output, primary, start, 0.86)
    used = [name]

    # Mass is reinforced from the recording's own low band; a synthesized
    # sub-tone under a real throat is exactly what gives hybrids away.
    if size.weight > 0.6:
        weight = min(1.0, (size.weight - 0.6) / 2.7)
        _mix_at(output, _body(primary), start + round(0.004 * SR), 0.20 + 0.34 * weight)

    if character.air > 0.04 and frames > round(0.12 * SR):
        breath_name = _choose(rng, BREATH)
        breath = _prepare(breath_name, size, character, rng, 0.86)
        breath = _fit(breath, max(1, frames - round(0.02 * SR)))
        _mix_at(output, breath, round(0.008 * SR), character.air * (0.22 + 0.16 * intensity))
        used.append(breath_name)

    if character.wet > 0.04:
        wet_name = _choose(rng, GROWL_WET)
        wet = _prepare(wet_name, size, character, rng, 1.05)
        wet = _fit(wet, max(1, frames - round(0.02 * SR)))
        _mix_at(output, wet, round(0.020 * SR), character.wet * 0.30)
        used.append(wet_name)
    return output, used


def chirp(size, character, frames, intensity, rng):
    """Short call: a contact chirp, peep, or chitter."""

    return _vocal("chirp", size, character, frames, intensity, rng, 1.06)


def purr(size, character, frames, intensity, rng):
    """Sustained contentment rumble, pulsed at a body-scaled purr rate.

    A purr is a real breath recording pulsed at roughly 25 Hz rather than a
    tone: the modulation is the sound, so it is applied to recorded breath.
    """

    output = np.zeros(frames, dtype=float)
    pool = _pool("purr", character)
    used: list[str] = []

    # Draw a fresh recording for every tile and jitter the hop. Tiling one clip
    # at a fixed hop puts a strong comb at the repeat rate -- measurably louder
    # than the purr itself -- and the bed just reads as that clip looping.
    cursor = 0
    while cursor < frames:
        name = _choose(rng, pool)
        source = _prepare(name, size, character, rng, 0.78)
        if len(source) < 2:
            break
        if name not in used:
            used.append(name)
        take = min(len(source), frames - cursor)
        segment = _fades(source[:take], 0.010)
        _mix_at(output, segment, cursor, float(rng.uniform(0.82, 1.0)))
        cursor += max(1, take - round(float(rng.uniform(0.02, 0.07)) * SR))
    if not used:
        return output, [_choose(rng, pool)]

    rate = float(min(48.0, max(11.0, 26.0 / math.sqrt(max(size.weight, 0.05)))))
    phase = np.arange(frames) / SR * rate * 2.0 * math.pi
    depth = 0.34 + 0.22 * intensity
    output *= (1.0 - depth) + depth * (0.5 + 0.5 * np.sin(phase))
    envelope = np.minimum(
        1.0, np.linspace(0.0, frames / SR, frames) / max(0.08, frames / SR * 0.16)
    )
    return output * envelope, used


def growl(size, character, frames, intensity, rng):
    """Low threat vocalization."""

    return _vocal("growl", size, character, frames, intensity, rng, 0.94)


def roar(size, character, frames, intensity, rng):
    """Full-power call, with a second stacked voice for weight."""

    output, used = _vocal("roar", size, character, frames, intensity, rng, 0.92)
    if size.weight > 1.2 and frames > round(0.25 * SR):
        # A large roar is layered in practice: a second, slower take an octave
        # down thickens it without turning it to mud.
        name = _choose(rng, _pool("roar", character))
        stacked = _prepare(name, size, character, rng, 0.62)
        stacked = _fit(stacked, max(1, frames - round(0.04 * SR)))
        _mix_at(output, stacked, round(0.018 * SR), 0.30 + 0.16 * intensity)
        used.append(name)
    return output, used


def pain(size, character, frames, intensity, rng):
    """Hurt reaction."""

    return _vocal("pain", size, character, frames, intensity, rng, 1.04)


def defeat(size, character, frames, intensity, rng):
    """Death or defeat cry, trailing into a final breath."""

    output, used = _vocal("defeat", size, character, frames, intensity, rng, 0.96)
    if frames > round(0.35 * SR):
        name = _choose(rng, BREATH)
        last = _prepare(name, size, character, rng, 0.70)
        last = _fit(last, max(1, frames - round(0.22 * SR)))
        _mix_at(output, last, round(0.20 * SR), 0.26 + 0.10 * intensity)
        used.append(name)
    return output, used


def bite(size, character, frames, intensity, rng):
    """Jaw snap, with flesh contact and a heavy creature's jaw mass."""

    output = np.zeros(frames, dtype=float)
    name = _choose(rng, _pool("bite", character))
    primary = _prepare(name, size, character, rng)
    primary = _fit(primary, max(1, frames - round(0.02 * SR)))
    _mix_at(output, primary, round(0.006 * SR), 0.88)
    used = [name]

    if intensity > 0.4 and frames > round(0.18 * SR):
        chew_name = _choose(rng, CHEW)
        chew = _prepare(chew_name, size, character, rng, 1.08)
        chew = _fit(chew, max(1, frames - round(0.06 * SR)))
        _mix_at(output, chew, round(0.05 * SR), 0.24 + 0.18 * intensity)
        used.append(chew_name)

    if size.weight > 1.2:
        impact_name = _choose(rng, IMPACT)
        impact = _prepare(impact_name, size, character, rng, 0.80)
        impact = _fit(impact, max(1, round(0.22 * SR)))
        _mix_at(output, impact, round(0.004 * SR), 0.18 + 0.14 * min(1.0, size.weight / 3.3))
        used.append(impact_name)
    return output, used


def claw(size, character, frames, intensity, rng):
    """Claw swipe: two or three strokes of contact plus limb movement."""

    output = np.zeros(frames, dtype=float)
    used: list[str] = []
    strokes = 2 if size.weight < 1.5 else 3
    for index in range(strokes):
        name = _choose(rng, IMPACT)
        stroke = _prepare(name, size, character, rng, 1.35)
        stroke = _fit(stroke, max(1, round(0.20 * SR)))
        onset = round((0.012 + index * float(rng.uniform(0.055, 0.095))) * SR)
        if onset >= frames:
            break
        _mix_at(output, stroke, onset, (0.62 + 0.24 * intensity) * (1.0 - 0.16 * index))
        used.append(name)

    cloth_name = _choose(rng, CLOTH)
    cloth = _prepare(cloth_name, size, character, rng, 1.18)
    cloth = _fit(cloth, max(1, frames - round(0.02 * SR)))
    _mix_at(output, cloth, 0, 0.30 + 0.14 * intensity)
    used.append(cloth_name)
    return output, used


def wing(size, character, frames, intensity, rng, beats: int = 3):
    """Wing beats re-timed from a real flap to this creature's wing rate.

    Wing rate scales inversely with size: a hatchling blurs, a dragon heaves.
    """

    output = np.zeros(frames, dtype=float)
    used: list[str] = []
    interval = 1.0 / max(0.4, size.wing_rate)
    for index in range(beats):
        flap, name = _single_flap(size, character, rng)
        onset = index * interval + (0.0 if index == 0 else float(rng.uniform(-0.05, 0.05)) * interval)
        onset_frames = round(max(0.0, onset) * SR)
        if onset_frames >= frames:
            break
        _mix_at(output, flap, onset_frames, (0.70 + 0.26 * intensity) * float(rng.uniform(0.88, 1.06)))
        if name not in used:
            used.append(name)

        # Big wings are membrane and muscle, not just air: a cloth layer under
        # each beat is the standard Foley trick and keeps them from sounding thin.
        if size.weight > 0.9:
            cloth_name = _choose(rng, CLOTH)
            cloth = _prepare(cloth_name, size, character, rng, 0.82)
            cloth = _fit(cloth, max(1, round(0.30 * SR)))
            _mix_at(output, cloth, onset_frames, 0.16 + 0.12 * min(1.0, size.weight / 3.3))
            if cloth_name not in used:
                used.append(cloth_name)
    return output, used


def move(size, character, frames, intensity, rng):
    """Body movement: hide, limbs, and a heavy creature's footfall."""

    output = np.zeros(frames, dtype=float)
    name = _choose(rng, CLOTH)
    primary = _prepare(name, size, character, rng, 0.92)
    primary = _fit(primary, max(1, frames - round(0.02 * SR)))
    _mix_at(output, primary, round(0.008 * SR), 0.78)
    used = [name]

    if size.weight > 1.2:
        stomp_name = _choose(rng, STOMP)
        stomp = _prepare(stomp_name, size, character, rng, 0.74)
        stomp = _fit(stomp, max(1, frames - round(0.02 * SR)))
        gain = 0.30 + 0.30 * min(1.0, (size.weight - 1.2) / 2.1)
        _mix_at(output, stomp, round(0.03 * SR), gain * (0.7 + 0.3 * intensity))
        used.append(stomp_name)
        _mix_at(output, _body(stomp), round(0.03 * SR), gain * 0.5)

    if character.wet > 0.04:
        wet_name = _choose(rng, GROWL_WET)
        wet = _prepare(wet_name, size, character, rng, 1.1)
        wet = _fit(wet, max(1, frames - round(0.02 * SR)))
        _mix_at(output, wet, round(0.012 * SR), character.wet * 0.34)
        used.append(wet_name)
    return output, used


def egg_crack(size, character, frames, intensity, rng):
    """Shell fracturing, plus the shell fragments that follow."""

    output = np.zeros(frames, dtype=float)
    used: list[str] = []
    # Shell is small and brittle: the break recordings are re-tuned well above
    # their source material so they read as eggshell rather than as a crate.
    shell_speed = 1.9 / max(0.45, math.sqrt(max(size.weight, 0.05)))

    cracks = 2 if size.weight < 1.0 else 3
    for index in range(cracks):
        name = _choose(rng, SHELL)
        crack = _prepare(name, size, character, rng, shell_speed)
        crack = _fit(crack, max(1, round(0.34 * SR)))
        onset = round((0.010 + index * float(rng.uniform(0.10, 0.20))) * SR)
        if onset >= frames:
            break
        _mix_at(output, crack, onset, (0.52 + 0.30 * intensity) * (0.82 + 0.24 * index))
        used.append(name)

    debris_name = _choose(rng, DEBRIS)
    debris = _prepare(debris_name, size, character, rng, shell_speed * 0.92)
    debris = _fit(debris, max(1, frames - round(0.16 * SR)))
    _mix_at(output, debris, round(0.14 * SR), 0.26 + 0.14 * intensity)
    used.append(debris_name)
    return output, used


def hatch(size, character, frames, intensity, rng):
    """Full hatch: shell gives way, the hatchling pushes out and calls."""

    output = np.zeros(frames, dtype=float)
    crack_frames = max(1, round(frames * 0.62))
    cracks, used = egg_crack(size, character, crack_frames, intensity, rng)
    _mix_at(output, cracks, 0, 0.82)

    shell_speed = 1.9 / max(0.45, math.sqrt(max(size.weight, 0.05)))
    debris_name = _choose(rng, DEBRIS)
    debris = _prepare(debris_name, size, character, rng, shell_speed)
    debris = _fit(debris, max(1, round(0.45 * SR)))
    _mix_at(output, debris, round(frames * 0.44), 0.34 + 0.12 * intensity)
    used.append(debris_name)

    # The newborn is smaller than the adult the size names, so its first call
    # is retuned above the requested size rather than at it.
    call_frames = max(1, frames - round(frames * 0.58))
    call, call_used = _vocal(
        "chirp", size, character, call_frames, intensity, rng, 1.35
    )
    _mix_at(output, call, round(frames * 0.58), 0.90)
    used.extend(call_used)
    return output, used


def early_reflections(
    signal: np.ndarray,
    room: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place the creature in a space with a few short reflections."""

    result = signal.copy()
    if room <= 0.0:
        return result
    for index in range(4):
        delay = round(float(rng.uniform(0.009, 0.042)) * SR)
        gain = room * float(rng.uniform(0.035, 0.090)) * math.exp(-0.30 * index)
        if 0 < delay < len(signal):
            result[delay:] += signal[:-delay] * gain
    return result


def master(signal: np.ndarray, peak_db: float) -> list[float]:
    """Final chain shared by every cue: band-limit, fade, normalize."""

    processed = sosfilt(butter(2, 26.0, btype="highpass", fs=SR, output="sos"), signal)
    processed = sosfilt(
        butter(2, 17_000.0, btype="lowpass", fs=SR, output="sos"), processed
    )
    fade_in = min(len(processed), max(1, round(0.002 * SR)))
    fade_out = min(len(processed), max(1, round(0.030 * SR)))
    processed[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
    processed[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    peak = float(np.max(np.abs(processed))) if len(processed) else 0.0
    if peak > 1e-12:
        processed = processed * (10.0 ** (peak_db / 20.0) / peak)
    return np.clip(processed, -0.98, 0.98).astype(float).tolist()


def size_named(name: str) -> SizeProfile:
    try:
        return SIZES[name]
    except KeyError:
        raise ValueError(
            f"unknown size {name!r}; expected one of {sorted(SIZES)}"
        ) from None


def character_named(name: str) -> CharacterProfile:
    try:
        return CHARACTERS[name]
    except KeyError:
        raise ValueError(
            f"unknown character {name!r}; expected one of {sorted(CHARACTERS)}"
        ) from None


def bundled_sources() -> tuple[str, ...]:
    """Every recording the pools can reach, for provenance checks."""

    pools = (
        CHIRP_SMALL, CHIRP_BUG, GROWL_BEAST, GROWL_BIG, GROWL_WET,
        GROWL_SPECTRAL, ROAR, ROAR_BIG, SCREECH, HOWL, PAIN, DEFEAT, BREATH,
        GRUNT, BITE, CHEW, BITE_SMALL, CLOTH, STOMP, SHELL, DEBRIS, IMPACT,
        WINGS,
    )
    names: set[str] = set()
    for pool in pools:
        names.update(pool)
    for character in CHARACTERS.values():
        for pool in character.pools.values():
            names.update(pool)
    return tuple(sorted(names))
