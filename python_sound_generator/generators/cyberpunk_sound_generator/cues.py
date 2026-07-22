"""Cue and pack registries for the procedural cyberpunk sound library."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from ..._audio import dsp
from ..._audio.models import RenderResult, VariationFilenames
from ..._audio.validation import bounded_float
from . import engine
from .engine import STYLES, Style


Renderer = Callable[[Style, random.Random, float], list[float]]


@dataclass(frozen=True, slots=True)
class Options:
    style: Style = STYLES["electric"]
    intensity: float = 0.85
    foley: float = 0.0


@dataclass(frozen=True, slots=True)
class Cue:
    name: str
    group: str
    description: str
    renderer: Renderer
    peak_db: float
    seconds: float
    layers: tuple[str, ...]
    foley_kinds: tuple[str, ...] = ()


def _canvas(seconds: float) -> list[float]:
    return dsp.silence(seconds)


def _add(
    output: list[float],
    onset: float,
    signal: Sequence[float],
    gain: float = 1.0,
) -> None:
    dsp.add_at(output, onset, list(signal), gain)


def _noise_swell(
    seconds: float,
    style: Style,
    rng: random.Random,
    *,
    peak: float,
) -> list[float]:
    noise = engine.radio_noise(seconds, style, rng)
    envelope = dsp.bell_env(seconds, peak, max(0.02, peak * 0.75), 0.16)
    return dsp.mul(noise, envelope)


# Interface -----------------------------------------------------------------

def _terminal_key(style: Style, rng: random.Random,
                  intensity: float) -> list[float]:
    output = _canvas(0.12)
    tone = engine.digital_tone(
        78,
        0.085,
        style,
        rng,
        decay=0.018,
        attack=0.0002,
    )
    _add(output, 0.0, engine.glitch(tone, style, rng, 0.50), 0.62)
    _add(output, 0.0, engine.static_burst(0.025, style, rng),
         0.18 + 0.12 * intensity)
    return output


def _terminal_confirm(style: Style, rng: random.Random,
                      intensity: float) -> list[float]:
    output = engine.sequence(
        (67, 74), 0.32, style, rng, gap=0.075, tone_seconds=0.23, fm=True
    )
    _add(output, 0.0, engine.static_burst(0.032, style, rng), 0.20)
    return engine.glitch(output, style, rng, 0.18 + 0.12 * intensity)


def _terminal_error(style: Style, rng: random.Random,
                    intensity: float) -> list[float]:
    output = _canvas(0.36)
    _add(
        output,
        0.0,
        engine.sequence(
            (66, 65, 59),
            0.31,
            style,
            rng,
            gap=0.065,
            tone_seconds=0.13,
        ),
        0.72,
    )
    buzz = engine.static_burst(
        0.22, style, rng, low=180.0, high=2_400.0, decay=0.14
    )
    _add(output, 0.08, engine.glitch(buzz, style, rng, 1.0),
         0.28 + 0.22 * intensity)
    return output


def _hologram_open(style: Style, rng: random.Random,
                   intensity: float) -> list[float]:
    output = _canvas(0.55)
    _add(output, 0.0, _noise_swell(0.46, style, rng, peak=0.24), 0.30)
    _add(output, 0.02, engine.sweep(43, 79, 0.44, style, rng, curve=0.72),
         0.54 + 0.16 * intensity)
    _add(output, 0.31, engine.fm_tone(84, 0.22, style, rng, ratio=2.7), 0.28)
    return engine.glitch(output, style, rng, 0.20)


def _hologram_close(style: Style, rng: random.Random,
                    intensity: float) -> list[float]:
    output = _canvas(0.48)
    _add(output, 0.0, engine.sweep(79, 40, 0.38, style, rng, curve=1.35),
         0.56 + 0.14 * intensity)
    _add(output, 0.03, _noise_swell(0.36, style, rng, peak=0.12), 0.26)
    _add(output, 0.31, engine.static_burst(0.09, style, rng), 0.32)
    return engine.glitch(output, style, rng, 0.32)


# Network -------------------------------------------------------------------

def _connection_open(style: Style, rng: random.Random,
                     intensity: float) -> list[float]:
    output = _canvas(0.70)
    _add(output, 0.0, engine.radio_noise(0.48, style, rng), 0.12)
    _add(
        output,
        0.03,
        engine.pulse_train(62, 5, 0.068, 0.42, style, rng, rising=True),
        0.58,
    )
    _add(output, 0.34, engine.sequence((69, 76), 0.32, style, rng,
                                       gap=0.075, fm=True),
         0.48 + 0.16 * intensity)
    return output


def _connection_lost(style: Style, rng: random.Random,
                     intensity: float) -> list[float]:
    output = _canvas(0.58)
    fall = engine.sweep(74, 38, 0.44, style, rng, curve=1.4)
    _add(output, 0.0, engine.glitch(fall, style, rng, 1.0), 0.64)
    _add(output, 0.16, engine.radio_noise(0.34, style, rng),
         0.18 + 0.18 * intensity)
    _add(output, 0.42, engine.static_burst(0.10, style, rng), 0.40)
    return output


def _data_upload(style: Style, rng: random.Random,
                 intensity: float) -> list[float]:
    output = _canvas(0.84)
    _add(
        output,
        0.0,
        engine.pulse_train(64, 9, 0.062, 0.62, style, rng, rising=True),
        0.48 + 0.18 * intensity,
    )
    _add(output, 0.50, engine.sweep(67, 91, 0.25, style, rng, curve=0.7), 0.42)
    _add(output, 0.66, engine.fm_tone(91, 0.15, style, rng), 0.34)
    return engine.glitch(output, style, rng, 0.34)


def _data_download(style: Style, rng: random.Random,
                   intensity: float) -> list[float]:
    output = _canvas(0.80)
    for index in range(8):
        note = 84 - index * 1.6
        packet = engine.digital_tone(
            note, 0.055, style, rng, decay=0.014, attack=0.0003
        )
        _add(output, 0.025 + index * 0.065, packet,
             0.44 + 0.16 * intensity)
    _add(output, 0.55, engine.sequence((60, 67), 0.22, style, rng,
                                       gap=0.06, fm=True), 0.38)
    return engine.glitch(output, style, rng, 0.42)


def _scan_ping(style: Style, rng: random.Random,
               intensity: float) -> list[float]:
    output = _canvas(0.76)
    ping = engine.fm_tone(91, 0.44, style, rng, ratio=3.07, index=2.4,
                          decay=0.18)
    _add(output, 0.0, ping, 0.68 + 0.16 * intensity)
    _add(output, 0.16, ping[:dsp.samples(0.30)], 0.22)
    _add(output, 0.34, ping[:dsp.samples(0.22)], 0.10)
    _add(output, 0.0, engine.static_burst(0.045, style, rng), 0.22)
    return output


# Security ------------------------------------------------------------------

def _hack_start(style: Style, rng: random.Random,
                intensity: float) -> list[float]:
    output = _canvas(0.66)
    _add(output, 0.0, engine.sub_hit(0.30, style, rng), 0.36)
    _add(output, 0.035, engine.pulse_train(55, 6, 0.070, 0.45, style, rng,
                                           rising=True),
         0.50 + 0.18 * intensity)
    _add(output, 0.38, engine.sweep(52, 76, 0.22, style, rng), 0.38)
    return engine.glitch(output, style, rng, 0.36)


def _hack_progress(style: Style, rng: random.Random,
                   intensity: float) -> list[float]:
    output = _canvas(0.40)
    _add(output, 0.0, engine.pulse_train(72, 5, 0.071, 0.37, style, rng,
                                         rising=True),
         0.48 + 0.22 * intensity)
    _add(output, 0.0, engine.radio_noise(0.34, style, rng), 0.10)
    return engine.glitch(output, style, rng, 0.72)


def _hack_success(style: Style, rng: random.Random,
                  intensity: float) -> list[float]:
    output = _canvas(0.92)
    _add(output, 0.0, engine.sequence((60, 67, 72, 79), 0.72, style, rng,
                                      gap=0.105, tone_seconds=0.34, fm=True),
         0.58 + 0.16 * intensity)
    _add(output, 0.26, _noise_swell(0.54, style, rng, peak=0.30), 0.20)
    _add(output, 0.54, engine.fm_tone(84, 0.30, style, rng, ratio=2.51), 0.28)
    return engine.glitch(output, style, rng, 0.22)


def _hack_failure(style: Style, rng: random.Random,
                  intensity: float) -> list[float]:
    output = _canvas(0.72)
    _add(output, 0.0, engine.sequence((70, 64, 58), 0.48, style, rng,
                                      gap=0.09, tone_seconds=0.21), 0.58)
    _add(output, 0.10, engine.static_burst(0.46, style, rng, low=260.0,
                                           high=5_600.0, decay=0.26),
         0.28 + 0.18 * intensity)
    _add(output, 0.35, engine.sub_hit(0.30, style, rng), 0.42)
    return engine.glitch(output, style, rng, 1.0)


def _trace_warning(style: Style, rng: random.Random,
                   intensity: float) -> list[float]:
    output = _canvas(0.88)
    for index in range(6):
        note = 82 if index % 2 == 0 else 76
        _add(output, index * 0.115,
             engine.fm_tone(note, 0.085, style, rng, decay=0.025),
             0.52 + 0.20 * intensity)
    _add(output, 0.0, engine.radio_noise(0.78, style, rng), 0.13)
    _add(output, 0.58, engine.sub_hit(0.25, style, rng), 0.30)
    return engine.glitch(output, style, rng, 0.46)


# Cyberware -----------------------------------------------------------------

def _implant_boot(style: Style, rng: random.Random,
                  intensity: float) -> list[float]:
    output = _canvas(1.05)
    _add(output, 0.0, engine.sub_hit(0.34, style, rng), 0.32)
    _add(output, 0.08, engine.pulse_train(48, 7, 0.075, 0.56, style, rng,
                                          rising=True), 0.40)
    _add(output, 0.36, engine.sweep(45, 72, 0.44, style, rng, curve=0.68),
         0.42 + 0.18 * intensity)
    _add(output, 0.73, engine.sequence((60, 67, 72), 0.28, style, rng,
                                       gap=0.06, fm=True), 0.44)
    return engine.glitch(output, style, rng, 0.20)


def _implant_activate(style: Style, rng: random.Random,
                      intensity: float) -> list[float]:
    output = _canvas(0.62)
    _add(output, 0.0, engine.sub_hit(0.28, style, rng, start_hz=190.0), 0.42)
    _add(output, 0.025, engine.sweep(50, 82, 0.42, style, rng, curve=0.62),
         0.54 + 0.22 * intensity)
    _add(output, 0.34, engine.fm_tone(82, 0.22, style, rng), 0.30)
    return output


def _implant_malfunction(style: Style, rng: random.Random,
                         intensity: float) -> list[float]:
    output = _canvas(0.78)
    for index, onset in enumerate((0.0, 0.09, 0.16, 0.34, 0.39, 0.57)):
        note = 70 + (index % 3) * 5 - rng.uniform(0.0, 3.0)
        _add(output, onset,
             engine.digital_tone(note, 0.10, style, rng, decay=0.025),
             0.42 + 0.18 * intensity)
    _add(output, 0.04, engine.radio_noise(0.64, style, rng), 0.22)
    return engine.glitch(output, style, rng, 1.25)


def _overclock(style: Style, rng: random.Random,
               intensity: float) -> list[float]:
    output = _canvas(1.10)
    _add(output, 0.0, engine.sub_hit(0.38, style, rng), 0.34)
    for index in range(4):
        onset = index * 0.16
        _add(output, onset,
             engine.sweep(45 + index * 5, 67 + index * 6, 0.42,
                          style, rng, curve=0.72),
             (0.24 + index * 0.07) * (0.75 + 0.25 * intensity))
    _add(output, 0.62, _noise_swell(0.42, style, rng, peak=0.24), 0.24)
    _add(output, 0.78, engine.fm_tone(88, 0.26, style, rng), 0.30)
    return engine.glitch(output, style, rng, 0.24)


def _emp_blast(style: Style, rng: random.Random,
               intensity: float) -> list[float]:
    output = _canvas(0.95)
    _add(output, 0.0, engine.static_burst(0.54, style, rng, low=80.0,
                                          high=15_000.0, decay=0.15),
         0.46 + 0.26 * intensity)
    _add(output, 0.0, engine.sub_hit(0.62, style, rng, start_hz=230.0,
                                     end_hz=28.0), 0.78)
    power_down = engine.sweep(84, 30, 0.72, style, rng, curve=1.8,
                              decay=0.32)
    _add(output, 0.04, engine.glitch(power_down, style, rng, 0.80), 0.42)
    return output


# World ---------------------------------------------------------------------

def _comms_incoming(style: Style, rng: random.Random,
                    intensity: float) -> list[float]:
    output = _canvas(0.86)
    _add(output, 0.0, engine.radio_noise(0.78, style, rng), 0.14)
    for onset in (0.04, 0.36):
        _add(output, onset,
             engine.sequence((76, 83), 0.25, style, rng, gap=0.07,
                             tone_seconds=0.16, fm=True),
             0.48 + 0.16 * intensity)
    _add(output, 0.68, engine.static_burst(0.10, style, rng), 0.20)
    return engine.glitch(output, style, rng, 0.20)


def _credits_transfer(style: Style, rng: random.Random,
                      intensity: float) -> list[float]:
    output = _canvas(0.60)
    for index in range(5):
        note = 79 + index * 2
        _add(output, index * 0.052,
             engine.fm_tone(note, 0.095, style, rng, ratio=2.73,
                            decay=0.030),
             0.38 + 0.10 * intensity)
    _add(output, 0.28, engine.sequence((67, 74), 0.27, style, rng,
                                       gap=0.065, fm=True), 0.44)
    return output


def _objective_update(style: Style, rng: random.Random,
                      intensity: float) -> list[float]:
    output = _canvas(0.72)
    _add(output, 0.0, engine.pulse_train(65, 4, 0.062, 0.32, style, rng,
                                         rising=True), 0.32)
    _add(output, 0.18, engine.sequence((62, 69, 74), 0.47, style, rng,
                                       gap=0.09, tone_seconds=0.25, fm=True),
         0.48 + 0.15 * intensity)
    _add(output, 0.15, _noise_swell(0.42, style, rng, peak=0.20), 0.14)
    return engine.glitch(output, style, rng, 0.16)


def _drone_alert(style: Style, rng: random.Random,
                 intensity: float) -> list[float]:
    output = _canvas(0.95)
    for index in range(5):
        onset = index * 0.155
        note = 76 if index % 2 == 0 else 69
        _add(output, onset,
             engine.digital_tone(note, 0.13, style, rng, decay=0.055,
                                 vibrato=0.012),
             0.50 + 0.18 * intensity)
    rotor = [
        math.sin(dsp.TAU * 34.0 * frame / engine.SR)
        * (0.45 + 0.55 * math.sin(dsp.TAU * 4.1 * frame / engine.SR) ** 2)
        for frame in range(dsp.samples(0.88))
    ]
    _add(output, 0.0, rotor, 0.10 * style.body)
    _add(output, 0.0, engine.radio_noise(0.86, style, rng), 0.10)
    return engine.glitch(output, style, rng, 0.24)


def _production_renderer(kind: str, signature: int,
                         seconds: float) -> Renderer:
    """Build a compact, style-aware renderer for production game semantics.

    These cues deliberately share the cyberpunk palette while using distinct
    pitch, cadence, bandwidth, and transient signatures.  They remain fully
    procedural so physical recordings are only introduced through the
    explicit, provenance-aware Foley path.
    """

    def render(style: Style, rng: random.Random,
               intensity: float) -> list[float]:
        output = _canvas(seconds)
        base = 40 + signature % 31
        bright = 74 + signature % 18
        weight = 0.72 + 0.28 * intensity
        if kind == "ui":
            notes = (base + 12, base + 19) if signature % 2 == 0 else (
                base + 19, base + 12
            )
            _add(output, 0.0, engine.sequence(
                notes, min(seconds, 0.34), style, rng,
                gap=0.065, tone_seconds=min(0.18, seconds), fm=True,
            ), 0.55 * weight)
            _add(output, 0.0, engine.static_burst(
                min(seconds, 0.035), style, rng,
            ), 0.15)
        elif kind == "weapon":
            _add(output, 0.0, engine.sub_hit(
                min(seconds, 0.34), style, rng,
                start_hz=145.0 + signature * 5.0,
                end_hz=38.0 + signature % 9,
            ), 0.58 * weight)
            _add(output, 0.0, engine.static_burst(
                min(seconds, 0.24), style, rng,
                low=110.0 + signature * 13.0,
                high=5_400.0 + signature * 170.0,
                decay=0.025 + (signature % 5) * 0.012,
            ), 0.48 * weight)
            _add(output, min(seconds * 0.30, 0.12), engine.fm_tone(
                base + 10, min(seconds * 0.55, 0.24), style, rng,
                ratio=1.8 + (signature % 5) * 0.31,
            ), 0.18)
        elif kind == "impact":
            _add(output, 0.0, engine.sub_hit(
                min(seconds, 0.42), style, rng,
                start_hz=120.0 + signature * 4.0,
            ), 0.64 * weight)
            _add(output, 0.0, engine.static_burst(
                min(seconds, 0.32), style, rng,
                low=180.0 + signature * 19.0,
                high=3_600.0 + signature * 230.0,
            ), 0.34 + 0.16 * intensity)
            _add(output, 0.015, engine.sweep(
                bright, base, min(seconds, 0.34), style, rng,
                curve=1.1 + (signature % 4) * 0.22,
            ), 0.21)
        elif kind == "motion":
            direction = -1 if signature % 2 else 1
            start = base + (24 if direction < 0 else 0)
            end = start + direction * (18 + signature % 12)
            _add(output, 0.0, engine.sweep(
                start, end, min(seconds * 0.82, 0.70), style, rng,
                curve=0.62 if direction > 0 else 1.42,
            ), 0.52 * weight)
            _add(output, 0.0, _noise_swell(
                min(seconds, 0.66), style, rng,
                peak=min(seconds * 0.42, 0.24),
            ), 0.19)
        elif kind == "alert":
            count = 3 + signature % 4
            period = min(0.13, seconds / (count + 1))
            for index in range(count):
                note = bright if index % 2 == 0 else bright - 7 - signature % 3
                _add(output, index * period, engine.digital_tone(
                    note, min(0.10, seconds - index * period), style, rng,
                    decay=0.030, vibrato=0.010 + signature % 4 * 0.003,
                ), 0.48 * weight)
            _add(output, 0.0, engine.radio_noise(
                min(seconds, 0.82), style, rng,
            ), 0.10 + 0.08 * intensity)
        elif kind == "positive":
            notes = (base, base + 7, base + 12, base + 19)
            _add(output, 0.0, engine.sequence(
                notes, min(seconds, 0.82), style, rng,
                gap=max(0.055, min(0.12, seconds / 7.0)),
                tone_seconds=min(0.30, seconds * 0.55), fm=True,
            ), 0.55 * weight)
            _add(output, min(seconds * 0.28, 0.22), _noise_swell(
                min(seconds * 0.62, 0.52), style, rng,
                peak=min(seconds * 0.20, 0.15),
            ), 0.14)
        elif kind == "negative":
            _add(output, 0.0, engine.sequence(
                (bright, bright - 6, bright - 13), min(seconds, 0.62),
                style, rng, gap=0.085, tone_seconds=min(0.22, seconds),
            ), 0.52 * weight)
            _add(output, min(0.08, seconds * 0.18), engine.static_burst(
                min(seconds * 0.72, 0.48), style, rng,
                low=180.0, high=4_200.0 + signature * 90.0,
            ), 0.24 + 0.12 * intensity)
        elif kind == "matrix":
            _add(output, 0.0, engine.pulse_train(
                base + 12, 4 + signature % 5, 0.058 + signature % 3 * 0.011,
                min(seconds, 0.68), style, rng, rising=signature % 2 == 0,
            ), 0.48 * weight)
            _add(output, min(seconds * 0.32, 0.20), engine.fm_tone(
                bright, min(seconds * 0.60, 0.42), style, rng,
                ratio=2.1 + signature % 6 * 0.27,
                index=1.4 + signature % 4 * 0.45,
            ), 0.32)
            _add(output, 0.0, engine.radio_noise(
                min(seconds, 0.72), style, rng,
            ), 0.09)
        elif kind == "magic":
            _add(output, 0.0, engine.sweep(
                base, bright, min(seconds * 0.74, 0.64), style, rng,
                curve=0.54 + signature % 4 * 0.10,
            ), 0.48 * weight)
            _add(output, min(seconds * 0.18, 0.14), engine.fm_tone(
                bright + signature % 5, min(seconds * 0.58, 0.44),
                style, rng, ratio=2.6 + signature % 3 * 0.53,
            ), 0.34)
            _add(output, 0.0, _noise_swell(
                min(seconds, 0.66), style, rng,
                peak=min(seconds * 0.44, 0.28),
            ), 0.16)
        else:
            raise ValueError(f"unknown production renderer kind: {kind}")
        return engine.glitch(output, style, rng, 0.12 + signature * 0.013)

    return render


GROUPS = (
    "interface", "network", "security", "cyberware", "world",
    "combat", "stealth", "matrix", "magic",
)


def _cue(
    name: str,
    group: str,
    description: str,
    renderer: Renderer,
    peak_db: float,
    seconds: float,
    *layers: str,
    foley_kinds: tuple[str, ...] = (),
) -> Cue:
    return Cue(
        name,
        group,
        description,
        renderer,
        peak_db,
        seconds,
        layers,
        foley_kinds,
    )


CUES: dict[str, Cue] = {
    cue.name: cue
    for cue in (
        _cue("terminal_key", "interface", "terminal keypress or compact cursor move",
             _terminal_key, -9.0, 0.12, "quantised key tone", "contact static",
             foley_kinds=("mechanical key",)),
        _cue("terminal_confirm", "interface", "terminal command accepted",
             _terminal_confirm, -6.0, 0.32, "rising FM pair", "digital edge"),
        _cue("terminal_error", "interface", "invalid command or rejected operation",
             _terminal_error, -5.0, 0.36, "falling code", "corrupted buzz"),
        _cue("hologram_open", "interface", "project a holographic display",
             _hologram_open, -5.0, 0.55, "spectral rise", "projection static", "lock tone"),
        _cue("hologram_close", "interface", "collapse a holographic display",
             _hologram_close, -6.0, 0.48, "spectral fall", "static collapse"),
        _cue("connection_open", "network", "establish a remote data connection",
             _connection_open, -5.0, 0.70, "packet handshake", "carrier noise", "connected code"),
        _cue("connection_lost", "network", "remote connection interrupted",
             _connection_lost, -4.5, 0.58, "broken carrier", "falling link", "static cutoff"),
        _cue("data_upload", "network", "upload or transmit a data payload",
             _data_upload, -5.0, 0.84, "ascending packets", "completion chirp"),
        _cue("data_download", "network", "download or receive a data payload",
             _data_download, -5.5, 0.80, "descending packets", "receipt code"),
        _cue("scan_ping", "network", "active scanner ping and electronic return",
             _scan_ping, -4.0, 0.76, "metallic ping", "two returns", "transmit click"),
        _cue("hack_start", "security", "begin an intrusion or bypass attempt",
             _hack_start, -4.5, 0.66, "system wake", "clock packets", "rising probe"),
        _cue("hack_progress", "security", "one repeatable hacking progress tick",
             _hack_progress, -7.0, 0.40, "ascending data ticks", "carrier bed"),
        _cue("hack_success", "security", "security bypass completed",
             _hack_success, -3.5, 0.92, "access arpeggio", "spectral bloom", "credential tone"),
        _cue("hack_failure", "security", "intrusion rejected or countermeasure triggered",
             _hack_failure, -3.0, 0.72, "falling rejection", "corruption", "low lockout"),
        _cue("trace_warning", "security", "hostile trace is closing on the player",
             _trace_warning, -2.5, 0.88, "alternating alarm", "network noise", "threat pulse"),
        _cue("implant_boot", "cyberware", "cybernetic implant initialization sequence",
             _implant_boot, -4.0, 1.05, "power rail", "diagnostic clock", "boot sweep", "ready code"),
        _cue("implant_activate", "cyberware", "cyberware engages successfully",
             _implant_activate, -3.0, 0.62, "mechanical impulse", "energy rise", "active tone",
             foley_kinds=("servo movement",)),
        _cue("implant_malfunction", "cyberware", "cyberware skips, faults, or destabilizes",
             _implant_malfunction, -2.5, 0.78, "broken clock", "radio interference", "bit failure"),
        _cue("overclock", "cyberware", "push cyberware beyond its nominal limit",
             _overclock, -2.5, 1.10, "power stages", "rising rails", "limit tone"),
        _cue("emp_blast", "cyberware", "electromagnetic pulse and electronics shutdown",
             _emp_blast, -1.5, 0.95, "wideband discharge", "sub impact", "power collapse"),
        _cue("comms_incoming", "world", "incoming encrypted communication",
             _comms_incoming, -5.0, 0.86, "radio bed", "double call code", "squelch"),
        _cue("credits_transfer", "world", "currency or credits transfer confirmed",
             _credits_transfer, -5.0, 0.60, "credit ticks", "ledger confirmation"),
        _cue("objective_update", "world", "mission objective or dossier updated",
             _objective_update, -5.0, 0.72, "data preamble", "objective code", "display swell"),
        _cue("drone_alert", "world", "nearby surveillance or combat drone warning",
             _drone_alert, -2.5, 0.95, "alternating siren", "rotor modulation", "radio telemetry"),
        _cue("cursor_move", "interface", "move through a terminal or tactical menu",
             _production_renderer("ui", 1, 0.10), -10.0, 0.10, "dry data chirp"),
        _cue("menu_cancel", "interface", "cancel or back out of an interface",
             _production_renderer("ui", 2, 0.24), -8.0, 0.24, "falling command pair"),
        _cue("dialogue_blip", "interface", "repeatable encrypted dialogue character blip",
             _production_renderer("ui", 3, 0.08), -13.0, 0.08, "short radio syllable"),
        _cue("door_open", "world", "powered security door opening",
             _production_renderer("motion", 4, 0.72), -5.0, 0.72, "motor sweep", "latch edge"),
        _cue("footstep", "world", "armored street footstep",
             _production_renderer("impact", 5, 0.24), -9.0, 0.24, "boot contact", "gear tick"),
        _cue("transport_arrive", "world", "cab, lift, or district transport arrival",
             _production_renderer("motion", 6, 0.92), -5.0, 0.92, "braking sweep", "arrival code"),
        _cue("purchase", "world", "equipment or service purchase confirmed",
             _production_renderer("positive", 7, 0.58), -6.0, 0.58, "ledger ticks", "receipt code"),
        _cue("clinic_heal", "world", "medical treatment or stabilization",
             _production_renderer("positive", 8, 0.90), -5.0, 0.90, "monitor rise", "recovery field"),
        _cue("runner_hire", "world", "runner joins under contract",
             _production_renderer("positive", 9, 0.96), -4.0, 0.96, "contract handshake", "crew code"),
        _cue("safehouse_rest", "world", "safehouse rest and recovery",
             _production_renderer("positive", 10, 1.10), -8.0, 1.10, "systems settle", "safe cadence"),
        _cue("save_game", "world", "campaign state committed safely",
             _production_renderer("positive", 11, 0.82), -7.0, 0.82, "write pulses", "commit tone"),
        _cue("combat_start", "combat", "tactical encounter begins",
             _production_renderer("alert", 12, 0.82), -2.5, 0.82, "threat pulses", "initiative lock"),
        _cue("pistol_shot", "combat", "compact firearm discharge",
             _production_renderer("weapon", 13, 0.42), -1.5, 0.42, "muzzle crack", "action ring"),
        _cue("smg_burst", "combat", "short automatic burst",
             _production_renderer("weapon", 14, 0.62), -1.0, 0.62, "burst reports", "mechanism chatter"),
        _cue("rifle_shot", "combat", "high-pressure rifle discharge",
             _production_renderer("weapon", 15, 0.56), -1.0, 0.56, "supersonic crack", "pressure tail"),
        _cue("shotgun_blast", "combat", "broad close-range firearm blast",
             _production_renderer("weapon", 16, 0.72), -0.8, 0.72, "broad report", "low pressure tail"),
        _cue("reload", "combat", "magazine and action reload sequence",
             _production_renderer("motion", 17, 0.76), -6.0, 0.76, "magazine movement", "action cycle"),
        _cue("dry_fire", "combat", "empty weapon trigger and striker",
             _production_renderer("ui", 18, 0.18), -8.0, 0.18, "dry mechanism click"),
        _cue("melee_swing", "combat", "close-combat weapon swing",
             _production_renderer("motion", 19, 0.38), -4.5, 0.38, "fast air cut"),
        _cue("melee_hit", "combat", "close-combat strike lands",
             _production_renderer("impact", 20, 0.46), -1.5, 0.46, "body impact", "weapon edge"),
        _cue("armor_hit", "combat", "attack stopped or reduced by armor",
             _production_renderer("impact", 21, 0.48), -2.0, 0.48, "plate contact", "deflection ring"),
        _cue("grenade_throw", "combat", "grenade leaves the hand",
             _production_renderer("motion", 22, 0.42), -5.0, 0.42, "throw whoosh", "arming tick"),
        _cue("grenade_explosion", "combat", "grenade detonation",
             _production_renderer("weapon", 23, 1.05), -0.5, 1.05, "shock transient", "debris tail"),
        _cue("ricochet", "combat", "projectile deflects from hard cover",
             _production_renderer("impact", 24, 0.58), -3.0, 0.58, "cover strike", "metallic departure"),
        _cue("runner_downed", "combat", "crew member becomes incapacitated",
             _production_renderer("negative", 25, 0.86), -2.5, 0.86, "condition fall", "flatline code"),
        _cue("combat_victory", "combat", "hostile force neutralized",
             _production_renderer("positive", 26, 1.06), -2.5, 1.06, "clear signal", "crew cadence"),
        _cue("mission_failure", "combat", "run fails or team is forced out",
             _production_renderer("negative", 27, 1.08), -2.0, 1.08, "objective collapse", "failure code"),
        _cue("stealth_detect", "stealth", "guard or sensor notices suspicious activity",
             _production_renderer("alert", 28, 0.52), -4.0, 0.52, "awareness pulse"),
        _cue("alarm_raise", "stealth", "facility alarm escalates",
             _production_renderer("alert", 29, 0.96), -1.5, 0.96, "security siren", "reinforcement code"),
        _cue("alarm_clear", "stealth", "facility returns to a lower alert state",
             _production_renderer("positive", 30, 0.68), -6.0, 0.68, "alarm release", "clear code"),
        _cue("lockpick", "stealth", "physical lock or maglock bypassed",
             _production_renderer("ui", 31, 0.54), -6.0, 0.54, "mechanism ticks", "unlock code"),
        _cue("overwatch_ready", "stealth", "overwatch reaction is armed",
             _production_renderer("alert", 32, 0.46), -5.0, 0.46, "target lock", "ready pulse"),
        _cue("escape", "stealth", "team reaches extraction or breaks pursuit",
             _production_renderer("positive", 33, 0.92), -3.5, 0.92, "pursuit release", "extraction cadence"),
        _cue("jack_in", "matrix", "operator enters the Matrix",
             _production_renderer("matrix", 34, 0.94), -3.0, 0.94, "persona boot", "node handshake"),
        _cue("program_attack", "matrix", "offensive program executes",
             _production_renderer("matrix", 35, 0.62), -2.0, 0.62, "program packets", "ICE impact"),
        _cue("ice_spawn", "matrix", "hostile countermeasure activates",
             _production_renderer("matrix", 36, 0.78), -2.0, 0.78, "ICE materialization", "threat clock"),
        _cue("dumpshock", "matrix", "operator is forcibly disconnected",
             _production_renderer("negative", 37, 1.02), -1.0, 1.02, "link rupture", "feedback impact"),
        _cue("magic_cast", "magic", "spell force gathers and releases",
             _production_renderer("magic", 38, 0.82), -3.0, 0.82, "force rise", "cast release"),
        _cue("magic_impact", "magic", "spell strikes or is resisted",
             _production_renderer("magic", 39, 0.72), -1.5, 0.72, "arcane impact", "resistance shimmer"),
        _cue("magic_drain", "magic", "spell backlash drains the caster",
             _production_renderer("negative", 40, 0.88), -2.5, 0.88, "energy collapse", "mental strain"),
    )
}


def _render_result(cue_name: str, options: Options,
                   rng: random.Random) -> RenderResult[list[float]]:
    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    if options.style.name not in STYLES or STYLES[options.style.name] != options.style:
        raise ValueError(f"unregistered style {options.style.name!r}")
    bounded_float(options.intensity, "intensity", 0.05, 1.0)
    bounded_float(options.foley, "foley", 0.0, 1.0)

    signal = cue.renderer(options.style, rng, options.intensity)
    sources: tuple[str, ...] = ()
    if options.foley > 0.0 and cue.foley_kinds:
        from . import foley

        signal, sources = foley.apply(
            cue_name,
            signal,
            options.style,
            options.intensity,
            options.foley,
            rng,
        )
    frames = dsp.samples(cue.seconds)
    signal = (signal + [0.0] * max(0, frames - len(signal)))[:frames]
    return RenderResult(
        engine.master(signal, cue.peak_db, options.style),
        sources,
    )


def render(cue_name: str, options: Options,
           rng: random.Random) -> list[float]:
    """Render one cue while preserving the historical list return type."""

    return _render_result(cue_name, options, rng).samples


def render_with_sources(cue_name: str, options: Options,
                        rng: random.Random) -> RenderResult[list[float]]:
    return _render_result(cue_name, options, rng)


def generate(
    cue_name: str,
    *,
    style: str = "electric",
    intensity: float = 0.85,
    foley_mix: float = 0.0,
    seed: int = 20260721,
) -> list[float]:
    """Return deterministic mono floating-point samples for one cue."""

    options = Options(
        style=engine.style_named(style),
        intensity=intensity,
        foley=foley_mix,
    )
    return render(cue_name, options, random.Random(seed))


def generate_with_sources(
    cue_name: str,
    *,
    style: str = "electric",
    intensity: float = 0.85,
    foley_mix: float = 0.0,
    seed: int = 20260721,
) -> RenderResult[list[float]]:
    options = Options(
        style=engine.style_named(style),
        intensity=intensity,
        foley=foley_mix,
    )
    return render_with_sources(cue_name, options, random.Random(seed))


def stem_for(cue_name: str, options: Options) -> str:
    if cue_name not in CUES:
        raise ValueError(f"unknown cue {cue_name!r}")
    return f"{cue_name}_{options.style.name}"


@dataclass(frozen=True, slots=True)
class PackEntry:
    filename: str
    cue: str
    count: int = 1


@dataclass(frozen=True, slots=True)
class Pack(VariationFilenames):
    name: str
    description: str
    entries: tuple[PackEntry, ...]


CORE = Pack(
    "core",
    "one organised render of every cyberpunk cue",
    tuple(PackEntry(f"{cue.group}/{cue.name}", cue.name) for cue in CUES.values()),
)

NETRUNNER = Pack(
    "netrunner",
    "three variations of network and intrusion feedback",
    tuple(
        PackEntry(name, name, 3)
        for name in (
            "connection_open",
            "connection_lost",
            "data_upload",
            "data_download",
            "scan_ping",
            "hack_start",
            "hack_progress",
            "hack_success",
            "hack_failure",
            "trace_warning",
        )
    ),
)

STREET_TECH = Pack(
    "street-tech",
    "two variations of interfaces, cyberware, and street-world feedback",
    tuple(
        PackEntry(name, name, 2)
        for name in (
            "terminal_key",
            "terminal_confirm",
            "terminal_error",
            "hologram_open",
            "hologram_close",
            "implant_boot",
            "implant_activate",
            "implant_malfunction",
            "overclock",
            "emp_blast",
            "comms_incoming",
            "credits_transfer",
            "objective_update",
            "drone_alert",
        )
    ),
)

PACKS = {pack.name: pack for pack in (CORE, NETRUNNER, STREET_TECH)}


__all__ = [
    "CUES",
    "GROUPS",
    "PACKS",
    "Cue",
    "Options",
    "Pack",
    "PackEntry",
    "generate",
    "generate_with_sources",
    "render",
    "render_with_sources",
    "stem_for",
]
