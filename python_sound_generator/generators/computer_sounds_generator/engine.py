"""Computer-machine synthesis and CC0 mechanical event processing.

Continuous sounds have measurable rates: fan blade-pass, spindle RPM, and mains
frequency. Physical controls do not, so their contacts come from dry CC0
recordings. Fan synthesis uses the shared neutral rotor model so computer
sounds do not initialize the aircraft or recording-driven Foley packages.
"""

from __future__ import annotations

from ..._audio.assets import resolve_asset

import math
import random
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

from ..._audio import dsp
from ..._audio import loops as loopdsp
from ..._audio.rotor import RotorProfile, blade_pass


SR = dsp.SR
TAU = dsp.TAU
MIN_SECONDS = 0.5
MAX_SECONDS = 30.0
SOURCES = Path(__file__).resolve().parent / "sources"


@dataclass(frozen=True)
class Size:
    name: str
    label: str
    description: str
    fan_rps: float
    blades: int
    spindle_rpm: float
    fan_gain: float
    drive_gain: float
    coil_hz: float


@dataclass(frozen=True)
class Region:
    name: str
    label: str
    mains_hz: float


SIZES: dict[str, Size] = {
    "laptop": Size(
        "laptop", "Laptop", "small fast fan and 5400 RPM mobile drive",
        52.0, 7, 5_400.0, 0.70, 0.35, 5_600.0,
    ),
    "desktop": Size(
        "desktop", "Desktop", "120 mm case fan and 7200 RPM hard drive",
        20.0, 7, 7_200.0, 0.85, 0.55, 7_200.0,
    ),
    "server": Size(
        "server", "Server", "small high-speed fan bank and 10000 RPM drive",
        78.0, 9, 10_000.0, 1.0, 0.75, 9_400.0,
    ),
}


REGIONS: dict[str, Region] = {
    "ntsc": Region("ntsc", "60 Hz mains", 60.0),
    "pal": Region("pal", "50 Hz mains", 50.0),
}


def size_named(name: str) -> Size:
    try:
        return SIZES[name]
    except KeyError:
        raise ValueError(f"unknown size {name!r}; expected one of {sorted(SIZES)}") from None


def region_named(name: str) -> Region:
    try:
        return REGIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown region {name!r}; expected one of {sorted(REGIONS)}"
        ) from None


def _check(seconds: float, intensity: float) -> None:
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise ValueError(
            f"seconds must be between {MIN_SECONDS} and {MAX_SECONDS}; got {seconds}"
        )
    if not 0.05 <= intensity <= 1.0:
        raise ValueError(f"intensity must be between 0.05 and 1; got {intensity}")


def _rms(signal: list[float]) -> float:
    return math.sqrt(sum(sample * sample for sample in signal) / max(1, len(signal)))


def _level(signal: list[float], target: float) -> list[float]:
    return dsp.scale(signal, target / max(_rms(signal), 1e-12))


def _fan_profile(size: Size) -> RotorProfile:
    """Describe a computer fan using only blade-pass fields."""

    return RotorProfile(
        name=f"computer_{size.name}_fan",
        shaft_rate=size.fan_rps,
        blades=size.blades,
        open_rotor=False,
        tone_gain=0.62,
        noise_gain=0.55,
        roughness=0.10,
        idle=0.72,
    )


def fan(size: Size, region: Region, seconds: float, rng: random.Random,
        intensity: float) -> list[float]:
    """Seamless cooling fan loop, delegated to aircraft blade-pass."""

    del region
    _check(seconds, intensity)
    return blade_pass(
        _fan_profile(size), rng, power=0.25 + 0.70 * intensity,
        duration=seconds, intensity=intensity,
    )


def drive(size: Size, region: Region, seconds: float, rng: random.Random,
          intensity: float) -> tuple[list[float], tuple[str, ...]]:
    """Seamless hard-drive idle loop cut from the approved 7200 RPM Foley."""

    del region
    _check(seconds, intensity)
    recorded = _read_source("drive_idle_cc0.wav")
    recorded = _resample(recorded, size.spindle_rpm / 7_200.0)
    loop = _recording_loop(recorded, seconds, rng)
    return dsp.scale(loop, 0.72 + 0.28 * intensity), ("drive_idle_cc0.wav",)


def psu(size: Size, region: Region, seconds: float, rng: random.Random,
        intensity: float) -> list[float]:
    """Seamless transformer hum, rectifier harmonics, and coil whine."""

    _check(seconds, intensity)
    hum = dsp.silence(seconds)
    for harmonic, gain in ((1, 0.30), (2, 1.0), (3, 0.20), (4, 0.28)):
        hum = dsp.mix(
            hum,
            dsp.scale(
                loopdsp.sine(
                    region.mains_hz * harmonic, seconds,
                    rng.uniform(0.0, TAU),
                ),
                gain,
            ),
        )
    coil = loopdsp.sine(size.coil_hz, seconds, rng.uniform(0.0, TAU))
    air = loopdsp.bandpass(dsp.noise(seconds, rng), 120.0, 2_800.0, 2)
    return dsp.mix(
        _level(hum, 0.08 + 0.04 * intensity),
        _level(coil, 0.012 + 0.012 * intensity),
        _level(air, 0.012),
    )


def _read_source(name: str) -> list[float]:
    path = resolve_asset(SOURCES / name)
    with wave.open(str(path), "rb") as wav_file:
        if (
            wav_file.getnchannels() != 1
            or wav_file.getsampwidth() != 2
            or wav_file.getframerate() != SR
        ):
            raise ValueError(f"source must be mono {SR} Hz PCM16: {path}")
        frames = wav_file.readframes(wav_file.getnframes())
    values = struct.unpack(f"<{len(frames) // 2}h", frames)
    return [value / 32768.0 for value in values]


def _resample(signal: list[float], ratio: float) -> list[float]:
    """Varispeed a recording; drive RPM changes pitch and duration together."""

    if ratio <= 0.0:
        raise ValueError("resample ratio must be positive")
    if abs(ratio - 1.0) < 1e-9:
        return list(signal)
    count = int(len(signal) / ratio)
    if count < 2:
        raise ValueError("resample ratio leaves nothing behind")
    last = len(signal) - 1
    out = []
    for index in range(count):
        position = index * ratio
        left = int(position)
        fraction = position - left
        right = min(left + 1, last)
        out.append(
            signal[left] * (1.0 - fraction) + signal[right] * fraction
        )
    return out


def _fold(first: list[float], second: list[float], fade: int) -> list[float]:
    """Join two recording stretches under an equal-power crossfade."""

    fade = min(fade, len(first), len(second))
    out = list(first[:len(first) - fade])
    for index in range(fade):
        position = index / max(1, fade - 1)
        out.append(
            first[len(first) - fade + index]
            * math.cos(position * math.pi * 0.5)
            + second[index] * math.sin(position * math.pi * 0.5)
        )
    return out + list(second[fade:])


def _recording_loop(signal: list[float], seconds: float,
                    rng: random.Random) -> list[float]:
    """Cut a seamless requested-duration loop from longer recorded Foley."""

    want = dsp.samples(seconds)
    if len(signal) < 16:
        raise ValueError("recording is too short to loop")
    fade = max(2, min(dsp.samples(0.25), len(signal) // 3))
    built = list(signal)
    while len(built) < want + fade:
        built = _fold(built, signal, fade)
    start = rng.randrange(0, len(built) - want - fade + 1)
    body = list(built[start:start + want])
    tail = built[start + want:start + want + fade]
    for index in range(min(fade, len(tail), len(body))):
        position = index / max(1, fade - 1)
        body[index] = (
            body[index] * math.sin(position * math.pi * 0.5)
            + tail[index] * math.cos(position * math.pi * 0.5)
        )
    return body


def _peak_clip(name: str, duration: float, before: float = 0.012) -> list[float]:
    signal = _read_source(name)
    peak = max(range(len(signal)), key=lambda index: abs(signal[index]))
    start = max(0, peak - dsp.samples(before))
    length = dsp.samples(duration)
    clip = signal[start:start + length]
    if len(clip) < length:
        clip += [0.0] * (length - len(clip))
    return clip


def key(size: Size, region: Region, rng: random.Random,
        intensity: float) -> tuple[list[float], tuple[str, ...]]:
    del size, region, rng
    _check(1.0, intensity)
    return _read_source("key_cc0.wav"), ("key_cc0.wav",)


def mouse_click(size: Size, region: Region, rng: random.Random,
                intensity: float) -> tuple[list[float], tuple[str, ...]]:
    """One physical mouse click: button-down, then button-up 120 ms later."""

    del size, region, rng
    _check(1.0, intensity)
    press = _read_source("mouse_press_cc0.wav")
    release = _read_source("mouse_release_cc0.wav")
    release_at = 0.12
    # Leave the shared one-shot master's edge fade in silence rather than over
    # the short release transient.
    out = dsp.silence(release_at + len(release) / SR + 0.06)
    dsp.add_at(out, 0.0, press)
    dsp.add_at(out, release_at, release)
    return out, ("mouse_press_cc0.wav", "mouse_release_cc0.wav")


def switch(size: Size, region: Region, rng: random.Random,
           intensity: float) -> tuple[list[float], tuple[str, ...]]:
    del size, region, rng
    _check(1.0, intensity)
    clip = _peak_clip("switch_cc0.wav", 0.30, 0.018)
    return dsp.highpass(clip, 120.0, 2), ("switch_cc0.wav",)


def seek(size: Size, region: Region, rng: random.Random,
         intensity: float) -> tuple[list[float], tuple[str, ...]]:
    del region
    _check(1.0, intensity)
    recorded = _resample(
        _read_source("seek_cc0.wav"), size.spindle_rpm / 7_200.0
    )
    count = min(len(recorded), dsp.samples(1.1 + 0.9 * intensity))
    start = rng.randrange(0, len(recorded) - count + 1)
    return recorded[start:start + count], ("seek_cc0.wav",)


def dial(size: Size, region: Region, rng: random.Random,
         intensity: float) -> list[float]:
    del size
    _check(1.0, intensity)
    frequencies = (350.0, 440.0) if region.name == "ntsc" else (425.0,)
    out = dsp.silence(1.0)
    for frequency in frequencies:
        out = dsp.mix(
            out, dsp.scale(dsp.sine(frequency, 1.0, rng.uniform(0, TAU)), 0.5)
        )
    return out


def modem(size: Size, region: Region, rng: random.Random,
          intensity: float) -> tuple[list[float], tuple[str, ...]]:
    """A real dial-up connection handshake."""

    del size, region, rng
    _check(1.0, intensity)
    return _read_source("modem_cc0.wav"), ("modem_cc0.wav",)


def spin_up(size: Size, region: Region, rng: random.Random,
            intensity: float) -> tuple[list[float], tuple[str, ...]]:
    del region, rng
    _check(1.0, intensity)
    recorded = _resample(
        _read_source("drive_spin_up_cc0.wav"), size.spindle_rpm / 7_200.0
    )
    return recorded, ("drive_spin_up_cc0.wav",)


def power_on(size: Size, region: Region, rng: random.Random,
             intensity: float) -> tuple[list[float], tuple[str, ...]]:
    del size, region, rng
    _check(1.0, intensity)
    source = _read_source("pc_startup_cc0.wav")
    return source[:dsp.samples(6.5)], ("pc_startup_cc0.wav",)


def boot(size: Size, region: Region, rng: random.Random,
         intensity: float) -> tuple[list[float], tuple[str, ...]]:
    del size, region, rng
    _check(1.0, intensity)
    return _read_source("pc_startup_cc0.wav"), ("pc_startup_cc0.wav",)


def shutdown(size: Size, region: Region, rng: random.Random,
             intensity: float) -> tuple[list[float], tuple[str, ...]]:
    del region, rng
    _check(1.0, intensity)
    recorded = _resample(
        _read_source("drive_spin_down_cc0.wav"), size.spindle_rpm / 7_200.0
    )
    return recorded, ("drive_spin_down_cc0.wav",)


def write_wav(path: Path, signal: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dsp.write_wav(path, signal)
    return path
