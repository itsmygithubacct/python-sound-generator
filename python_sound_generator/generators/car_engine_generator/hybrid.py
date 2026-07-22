#!/usr/bin/env python3
"""Recording-driven car engine renderer using NumPy and SciPy.

Sample banks contain steady engine recordings tagged with RPM and normalized
load. During rendering, every recording is a pitch anchor: playback follows the
requested RPM curve and nearby anchors are blended in two dimensions. A small
procedural layer supplies mechanical detail, gear thumps, and overrun pops.
"""

from __future__ import annotations

from ..._audio.assets import resolve_asset
from ..._audio.foley_numpy import pcm_to_float
from ..._audio.runtime import derive_seed

import argparse
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
    from scipy import signal
    from scipy.io import wavfile
except ImportError as error:  # pragma: no cover - exercised on minimal installs
    raise SystemExit(
        "Hybrid rendering requires NumPy and SciPy. Install them with "
        "python3 -m pip install numpy scipy"
    ) from error

from .car_engine_generator import (
    DEFAULT_SAMPLE_RATE,
    ENGINE_PROFILES,
    MODES,
    AudioBuffer,
    _resolve_rpms,
    _rpm_and_load_curves,
    _synthesize_layers,
    _validate_settings,
    write_wav,
)


BANK_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SampleLayer:
    """One steady recording used as an RPM/load interpolation anchor."""

    path: Path
    rpm: float
    load: float
    start: float = 0.0
    end: float | None = None
    loop_seconds: float | None = 2.0
    crossfade_ms: float = 80.0
    gain_db: float = 0.0
    label: str = ""


@dataclass(frozen=True)
class SampleBank:
    """Validated sample-bank metadata and its recording layers."""

    path: Path
    name: str
    engine: str
    cylinders: int
    layers: tuple[SampleLayer, ...]
    target_rms_db: float = -18.0
    rpm_blend_octaves: float = 0.48
    load_blend_width: float = 0.34
    source: str = ""
    license: str = ""


def _require_number(
    value: Any, field: str, low: float | None = None, high: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    if low is not None and parsed < low:
        raise ValueError(f"{field} must be at least {low:g}")
    if high is not None and parsed > high:
        raise ValueError(f"{field} must be at most {high:g}")
    return parsed


def load_sample_bank(path: str | Path) -> SampleBank:
    """Load and validate a JSON sample-bank manifest."""

    manifest = Path(path).expanduser().resolve()
    if manifest.is_dir():
        manifest = manifest / "manifest.json"
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read sample bank {manifest}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {manifest}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("sample-bank manifest must contain a JSON object")
    if raw.get("format_version") != BANK_FORMAT_VERSION:
        raise ValueError(
            f"sample-bank format_version must be {BANK_FORMAT_VERSION}"
        )

    name = str(raw.get("name", "Unnamed engine bank")).strip()
    engine = str(raw.get("engine", "")).strip()
    if engine not in ENGINE_PROFILES:
        raise ValueError(
            f"sample-bank engine must be one of {', '.join(ENGINE_PROFILES)}"
        )
    cylinders = int(raw.get("cylinders", ENGINE_PROFILES[engine].cylinders))
    if not 1 <= cylinders <= 16:
        raise ValueError("sample-bank cylinders must be between 1 and 16")
    layer_values = raw.get("layers")
    if not isinstance(layer_values, list) or not layer_values:
        raise ValueError("sample-bank layers must be a non-empty array")

    layers: list[SampleLayer] = []
    for index, value in enumerate(layer_values):
        prefix = f"layers[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{prefix} must be an object")
        relative = value.get("file")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"{prefix}.file must be a non-empty string")
        recording = (manifest.parent / relative).resolve()
        if not recording.is_file():
            raise ValueError(f"{prefix}.file does not exist: {recording}")
        end_value = value.get("end")
        loop_value = value.get("loop_seconds", 2.0)
        layer = SampleLayer(
            path=recording,
            rpm=_require_number(value.get("rpm"), f"{prefix}.rpm", 300.0, 20_000.0),
            load=_require_number(value.get("load"), f"{prefix}.load", 0.0, 1.0),
            start=_require_number(value.get("start", 0.0), f"{prefix}.start", 0.0),
            end=(
                None
                if end_value is None
                else _require_number(end_value, f"{prefix}.end", 0.0)
            ),
            loop_seconds=(
                None
                if loop_value is None
                else _require_number(
                    loop_value, f"{prefix}.loop_seconds", 0.15, 30.0
                )
            ),
            crossfade_ms=_require_number(
                value.get("crossfade_ms", 80.0),
                f"{prefix}.crossfade_ms",
                2.0,
                500.0,
            ),
            gain_db=_require_number(
                value.get("gain_db", 0.0), f"{prefix}.gain_db", -36.0, 24.0
            ),
            label=str(value.get("label", "")).strip(),
        )
        if layer.end is not None and layer.end <= layer.start:
            raise ValueError(f"{prefix}.end must be later than start")
        layers.append(layer)

    return SampleBank(
        path=manifest,
        name=name,
        engine=engine,
        cylinders=cylinders,
        layers=tuple(layers),
        target_rms_db=_require_number(
            raw.get("target_rms_db", -18.0), "target_rms_db", -36.0, -3.0
        ),
        rpm_blend_octaves=_require_number(
            raw.get("rpm_blend_octaves", 0.48),
            "rpm_blend_octaves",
            0.08,
            2.0,
        ),
        load_blend_width=_require_number(
            raw.get("load_blend_width", 0.34),
            "load_blend_width",
            0.05,
            1.0,
        ),
        source=str(raw.get("source", "")).strip(),
        license=str(raw.get("license", "")).strip(),
    )


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Read PCM or floating-point WAV as frames by channels float64."""

    path = resolve_asset(path)
    try:
        sample_rate, raw = wavfile.read(str(path))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read WAV {path}: {error}") from error
    audio = pcm_to_float(raw, mixdown=False)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[1] < 1:
        raise ValueError(f"unsupported channel layout in {path}")
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"recording contains non-finite samples: {path}")
    return int(sample_rate), audio


def _prepare_loop(
    layer: SampleLayer,
    target_rms_db: float,
    source_cache: dict[Path, tuple[int, np.ndarray]] | None = None,
) -> tuple[int, np.ndarray]:
    if source_cache is not None and layer.path in source_cache:
        sample_rate, source = source_cache[layer.path]
    else:
        sample_rate, source = read_wav(layer.path)
        if source_cache is not None:
            source_cache[layer.path] = (sample_rate, source)
    audio = source
    start = round(layer.start * sample_rate)
    end = len(audio) if layer.end is None else round(layer.end * sample_rate)
    start = max(0, min(start, len(audio)))
    end = max(start, min(end, len(audio)))
    audio = np.asarray(audio[start:end], dtype=np.float64)
    if len(audio) < round(0.15 * sample_rate):
        raise ValueError(f"loop region is too short in {layer.path}")

    if layer.loop_seconds is not None:
        wanted = round(layer.loop_seconds * sample_rate)
        if len(audio) > wanted:
            offset = (len(audio) - wanted) // 2
            audio = audio[offset:offset + wanted]

    audio = signal.detrend(audio, axis=0, type="constant")
    fade = min(
        round(layer.crossfade_ms * sample_rate / 1000.0),
        max(1, len(audio) // 4),
    )
    if fade >= 2:
        phase = np.linspace(0.0, math.pi / 2.0, fade, endpoint=True)[:, None]
        fade_in = np.sin(phase) ** 2
        fade_out = np.cos(phase) ** 2
        loop = np.array(audio[:-fade], copy=True)
        loop[:fade] = audio[-fade:] * fade_out + audio[:fade] * fade_in
    else:
        loop = np.array(audio, copy=True)

    rms = float(np.sqrt(np.mean(np.square(loop))))
    if rms < 1e-8:
        raise ValueError(f"loop region is effectively silent in {layer.path}")
    target_rms = 10.0 ** (target_rms_db / 20.0)
    gain = target_rms / rms * 10.0 ** (layer.gain_db / 20.0)
    return sample_rate, loop * gain


def analyze_recording(
    path: str | Path, cylinders: int, start: float = 0.0, end: float | None = None
) -> dict[str, float | int | str]:
    """Estimate the strongest steady firing frequency and corresponding RPM."""

    if not 1 <= cylinders <= 16:
        raise ValueError("cylinders must be between 1 and 16")
    sample_rate, audio = read_wav(path)
    mono = np.mean(audio, axis=1)
    first = max(0, round(start * sample_rate))
    last = len(mono) if end is None else min(len(mono), round(end * sample_rate))
    mono = mono[first:last]
    if len(mono) < round(0.2 * sample_rate):
        raise ValueError("analysis region must be at least 0.2 seconds")
    if len(mono) > round(8.0 * sample_rate):
        middle = len(mono) // 2
        radius = round(4.0 * sample_rate)
        mono = mono[middle - radius:middle + radius]
    mono = signal.detrend(mono, type="constant")
    frequencies, power = signal.welch(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=min(len(mono), 16_384),
        noverlap=min(len(mono) // 2, 8_192),
        scaling="spectrum",
    )
    band = (frequencies >= 20.0) & (frequencies <= 700.0)
    if not np.any(band):
        raise ValueError("recording has no analyzable engine-frequency band")
    local = np.flatnonzero(band)
    peak_index = int(local[np.argmax(power[band])])
    firing_hz = float(frequencies[peak_index])
    rpm = firing_hz * 120.0 / cylinders
    rms = float(np.sqrt(np.mean(np.square(mono))))
    return {
        "file": str(Path(path).resolve()),
        "sample_rate": sample_rate,
        "channels": int(audio.shape[1]),
        "duration": len(audio) / sample_rate,
        "firing_hz": firing_hz,
        "estimated_rpm": rpm,
        "rms_dbfs": 20.0 * math.log10(max(rms, 1e-12)),
    }


def _match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return np.mean(audio, axis=1, keepdims=True)
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio[:, :2]


def _sample_variable_loop(
    loop: np.ndarray,
    source_rate: int,
    output_rate: int,
    target_rpms: np.ndarray,
    anchor_rpm: float,
    initial_phase: float,
) -> np.ndarray:
    increment = target_rpms / anchor_rpm * (source_rate / output_rate)
    positions = np.cumsum(increment, dtype=np.float64) + initial_phase
    positions = np.mod(positions, len(loop))
    left = np.floor(positions).astype(np.int64)
    right = (left + 1) % len(loop)
    fraction = (positions - left)[:, None]
    return loop[left] * (1.0 - fraction) + loop[right] * fraction


def _transient_layer(
    rpms: np.ndarray,
    loads: np.ndarray,
    mode: str,
    sample_rate: int,
    seed: int | None,
) -> np.ndarray:
    """Create sparse shift thumps and overrun pops, not a synthetic engine bed."""

    rng = np.random.default_rng(seed)
    count = len(rpms)
    raw = np.zeros(count, dtype=np.float64)

    if mode == "accelerate":
        shift_candidates, _ = signal.find_peaks(
            -loads,
            distance=max(1, round(0.25 * sample_rate)),
            prominence=0.12,
        )
        for onset in shift_candidates:
            length = min(count - onset, round(0.11 * sample_rate))
            if length <= 1:
                continue
            time = np.arange(length) / sample_rate
            thump = (
                np.sin(2.0 * math.pi * 72.0 * time)
                + 0.32 * rng.standard_normal(length)
            ) * np.exp(-time / 0.030)
            raw[onset:onset + length] += thump * 0.10

    if mode in ("decelerate", "rev"):
        eligible = (loads < 0.25) & (rpms > 1_400.0)
        event_probability = 4.0 / sample_rate
        events = np.flatnonzero(eligible & (rng.random(count) < event_probability))
        last = -round(0.07 * sample_rate)
        for onset in events:
            if onset - last < round(0.07 * sample_rate):
                continue
            last = int(onset)
            length = min(count - onset, round(0.09 * sample_rate))
            time = np.arange(length) / sample_rate
            frequency = rng.uniform(62.0, 105.0)
            pop = (
                np.sin(2.0 * math.pi * frequency * time + rng.uniform(0, math.pi))
                + 0.45 * rng.standard_normal(length)
            ) * np.exp(-time / 0.022)
            raw[onset:onset + length] += pop * rng.uniform(0.10, 0.18)

    sos = signal.butter(3, [35.0, min(1_400.0, sample_rate * 0.44)],
                        btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfilt(sos, raw)


def _oversampled_saturate(
    audio: np.ndarray, amount: float, factor: int
) -> np.ndarray:
    if amount <= 0.0:
        return audio
    factor = max(1, int(factor))
    upsampled = (
        signal.resample_poly(audio, factor, 1, axis=0)
        if factor > 1
        else audio
    )
    drive = 1.0 + 2.2 * amount
    shaped = np.tanh(upsampled * drive) / math.tanh(drive)
    if factor > 1:
        shaped = signal.resample_poly(shaped, 1, factor, axis=0)
    return np.asarray(shaped[:len(audio)], dtype=np.float64)


def generate_hybrid_engine_sound(
    bank: str | Path | SampleBank,
    mode: str = "accelerate",
    duration: float = 6.0,
    start_rpm: float | None = None,
    end_rpm: float | None = None,
    throttle: float = 0.82,
    variation: float = 0.30,
    gear_shifts: int = 2,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
    stereo: bool = False,
    procedural_mix: float = 0.04,
    transient_mix: float = 0.55,
    saturation: float = 0.12,
    oversample: int = 2,
    peak_db: float = -1.0,
) -> AudioBuffer:
    """Render a recording-driven engine performance from a sample bank."""

    sample_bank = bank if isinstance(bank, SampleBank) else load_sample_bank(bank)
    profile = replace(
        ENGINE_PROFILES[sample_bank.engine], cylinders=sample_bank.cylinders
    )
    resolved_start, resolved_end = _resolve_rpms(
        profile, mode, start_rpm, end_rpm
    )
    _validate_settings(
        sample_bank.engine,
        mode,
        duration,
        resolved_start,
        resolved_end,
        throttle,
        variation,
        gear_shifts,
        0.5,
        sample_rate,
        peak_db,
    )
    if not 0.0 <= procedural_mix <= 1.0:
        raise ValueError("procedural_mix must be between 0 and 1")
    if not 0.0 <= transient_mix <= 2.0:
        raise ValueError("transient_mix must be between 0 and 2")
    if not 0.0 <= saturation <= 1.0:
        raise ValueError("saturation must be between 0 and 1")
    if oversample not in (1, 2, 4):
        raise ValueError("oversample must be 1, 2, or 4")

    control_rng = random.Random(seed)
    rpm_values, load_values = _rpm_and_load_curves(
        profile,
        mode,
        duration,
        resolved_start,
        resolved_end,
        throttle,
        variation,
        gear_shifts,
        sample_rate,
        control_rng,
    )
    rpms = np.asarray(rpm_values, dtype=np.float64)
    loads = np.asarray(load_values, dtype=np.float64)
    channels = 2 if stereo else 1
    mix = np.zeros((len(rpms), channels), dtype=np.float64)
    weight_power = np.zeros(len(rpms), dtype=np.float64)
    phase_rng = np.random.default_rng(seed)
    source_cache: dict[Path, tuple[int, np.ndarray]] = {}

    log_width = sample_bank.rpm_blend_octaves
    load_width = sample_bank.load_blend_width
    for layer in sample_bank.layers:
        source_rate, loop = _prepare_loop(
            layer, sample_bank.target_rms_db, source_cache
        )
        loop = _match_channels(loop, channels)
        sampled = _sample_variable_loop(
            loop,
            source_rate,
            sample_rate,
            rpms,
            layer.rpm,
            float(phase_rng.uniform(0.0, len(loop))),
        )
        rpm_distance = np.log2(np.maximum(rpms, 1.0) / layer.rpm) / log_width
        load_distance = (loads - layer.load) / load_width
        exponent = np.maximum(-60.0, -0.5 * (rpm_distance ** 2 + load_distance ** 2))
        weight = np.exp(exponent)
        mix += sampled * weight[:, None]
        weight_power += weight ** 2

    # The loop phases are intentionally independent, so use equal-power rather
    # than linear normalization to avoid a level dip halfway between anchors.
    mix /= np.maximum(np.sqrt(weight_power[:, None]), 1e-12)
    amplitude = 0.56 + 0.44 * np.sqrt(np.clip(loads, 0.0, 1.0))
    mix *= amplitude[:, None]

    if procedural_mix > 0.0:
        _, intake, mechanical, bass = _synthesize_layers(
            profile,
            mode,
            rpm_values,
            load_values,
            sample_rate,
            control_rng,
        )
        detail = (
            0.48 * np.asarray(intake)
            + 0.34 * np.asarray(mechanical)
            + 0.18 * np.asarray(bass)
        )
        if channels == 1:
            mix[:, 0] += detail * procedural_mix
        else:
            mix[:, 0] += detail * procedural_mix * 0.92
            mix[:, 1] += detail * procedural_mix * 1.08

    transients = _transient_layer(
        rpms,
        loads,
        mode,
        sample_rate,
        None
        if seed is None
        else derive_seed(
            seed,
            provider=__package__,
            cue="hybrid-transients",
            filename=bank.name,
        ),
    )
    mix += transients[:, None] * transient_mix

    highpass = signal.butter(3, 24.0, btype="highpass", fs=sample_rate, output="sos")
    mix = signal.sosfilt(highpass, mix, axis=0)
    if sample_rate > 36_000:
        lowpass = signal.butter(
            5,
            min(17_500.0, sample_rate * 0.44),
            btype="lowpass",
            fs=sample_rate,
            output="sos",
        )
        mix = signal.sosfilt(lowpass, mix, axis=0)
    mix = _oversampled_saturate(mix, saturation, oversample)

    mix -= np.mean(mix, axis=0, keepdims=True)
    fade_in = min(len(mix), max(2, round(0.018 * sample_rate)))
    fade_out = min(len(mix), max(2, round(0.070 * sample_rate)))
    mix[:fade_in] *= np.linspace(0.0, 1.0, fade_in)[:, None]
    mix[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)[:, None]
    peak = float(np.max(np.abs(mix)))
    if peak > 1e-12:
        mix *= 10.0 ** (peak_db / 20.0) / peak
    mix = np.clip(mix, -1.0, 1.0)
    return AudioBuffer(
        sample_rate,
        tuple(mix[:, channel].tolist() for channel in range(channels)),
    )


def _render_command(arguments: argparse.Namespace) -> int:
    try:
        bank = load_sample_bank(arguments.bank)
        audio = generate_hybrid_engine_sound(
            bank,
            mode=arguments.mode,
            duration=arguments.duration,
            start_rpm=arguments.start_rpm,
            end_rpm=arguments.end_rpm,
            throttle=arguments.throttle,
            variation=arguments.variation,
            gear_shifts=arguments.gear_shifts,
            sample_rate=arguments.sample_rate,
            seed=arguments.seed,
            stereo=arguments.stereo,
            procedural_mix=arguments.procedural_mix,
            transient_mix=arguments.transient_mix,
            saturation=arguments.saturation,
            oversample=arguments.oversample,
            peak_db=arguments.peak_db,
        )
        output = write_wav(arguments.output, audio)
    except (OSError, ValueError) as error:
        raise SystemExit(f"soundgen car-engine: {error}") from error
    channel_word = "stereo" if audio.channel_count == 2 else "mono"
    print(
        f"Wrote {bank.name} {arguments.mode} "
        f"({audio.duration:.2f}s, {audio.sample_rate} Hz, {channel_word}) to {output}"
    )
    return 0


def _analyze_command(arguments: argparse.Namespace) -> int:
    try:
        result = analyze_recording(
            arguments.recording,
            arguments.cylinders,
            arguments.start,
            arguments.end,
        )
    except ValueError as error:
        raise SystemExit(f"soundgen car-engine: {error}") from error
    print(json.dumps(result, indent=2))
    return 0


def _inspect_command(arguments: argparse.Namespace) -> int:
    try:
        bank = load_sample_bank(arguments.bank)
    except ValueError as error:
        raise SystemExit(f"soundgen car-engine: {error}") from error
    print(f"{bank.name} ({bank.engine}, {bank.cylinders} cylinders)")
    if bank.license:
        print(f"license: {bank.license}")
    if bank.source:
        print(f"source: {bank.source}")
    print("RPM     LOAD  RECORDING")
    for layer in sorted(bank.layers, key=lambda item: (item.rpm, item.load)):
        print(f"{layer.rpm:7.0f}  {layer.load:4.2f}  {layer.path.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen car-engine",
        description="Analyze recordings and render sample-driven car engines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="render from a sample bank")
    render.add_argument("bank", help="manifest JSON or bank directory")
    render.add_argument("output", help="output WAV path")
    render.add_argument("--mode", choices=MODES, default="accelerate")
    render.add_argument("--duration", type=float, default=6.0)
    render.add_argument("--start-rpm", type=float)
    render.add_argument("--end-rpm", type=float)
    render.add_argument("--throttle", type=float, default=0.82)
    render.add_argument("--variation", type=float, default=0.30)
    render.add_argument("--gear-shifts", type=int, default=2)
    render.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    render.add_argument("--seed", type=int)
    render.add_argument("--stereo", action="store_true")
    render.add_argument("--procedural-mix", type=float, default=0.04)
    render.add_argument("--transient-mix", type=float, default=0.55)
    render.add_argument("--saturation", type=float, default=0.12)
    render.add_argument("--oversample", type=int, choices=(1, 2, 4), default=2)
    render.add_argument("--peak-db", type=float, default=-1.0)
    render.set_defaults(func=_render_command)

    analyze = subparsers.add_parser(
        "analyze", help="estimate firing frequency and RPM of a steady recording"
    )
    analyze.add_argument("recording")
    analyze.add_argument("--cylinders", type=int, required=True)
    analyze.add_argument("--start", type=float, default=0.0)
    analyze.add_argument("--end", type=float)
    analyze.set_defaults(func=_analyze_command)

    inspect = subparsers.add_parser("inspect", help="list a sample bank")
    inspect.add_argument("bank")
    inspect.set_defaults(func=_inspect_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.func(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
