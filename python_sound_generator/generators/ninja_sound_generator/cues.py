"""Cue and pack registries for the hybrid ninja game sound library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..._audio.models import RenderResult, VariationFilenames
from ..._audio.validation import bounded_float
from . import engine, foley
from .engine import STYLES, Style


@dataclass(frozen=True, slots=True)
class Options:
    style: Style = STYLES["shadow"]
    intensity: float = 0.85
    foley: float = 1.0


Renderer = Callable[[Options, np.random.Generator], tuple[np.ndarray, list[str]]]


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


def _recorded(
    output: np.ndarray,
    onset: float,
    pool: Sequence[foley.FoleySource],
    options: Options,
    rng: np.random.Generator,
    sources: list[str],
    *,
    gain: float,
    speed: float = 1.0,
    low_hz: float = 35.0,
    high_hz: float = 18_000.0,
    decay_seconds: float | None = None,
    maximum_seconds: float | None = None,
) -> None:
    if options.foley <= 0.0:
        return
    signal, source = engine.recorded(
        pool,
        options.style,
        rng,
        speed=speed,
        low_hz=low_hz,
        high_hz=high_hz,
        decay_seconds=decay_seconds,
        maximum_seconds=maximum_seconds,
    )
    engine.add_at(output, onset, signal, gain * options.foley)
    sources.append(source)


def _silent_step(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.28)
    sources: list[str] = []
    _recorded(
        output, 0.01, foley.CLOTH, options, rng, sources,
        gain=0.28, speed=1.38, low_hz=120.0, high_hz=4_200.0,
        decay_seconds=0.10, maximum_seconds=0.23,
    )
    engine.add_at(
        output, 0.08,
        engine.noise_hit(0.13, options.style, rng, low=70.0, high=1_100.0,
                         decay_seconds=0.035),
        0.10 + 0.07 * options.intensity,
    )
    return output, sources


def _rooftop_run(options: Options, rng: np.random.Generator):
    output = engine.canvas(1.10)
    sources: list[str] = []
    for index, onset in enumerate((0.02, 0.23, 0.45, 0.68, 0.91)):
        _recorded(
            output, onset, foley.WOOD_HIT, options, rng, sources,
            gain=(0.24 + 0.10 * options.intensity) * (1.0 - index * 0.035),
            speed=1.72 + index * 0.035, low_hz=90.0, high_hz=4_800.0,
            decay_seconds=0.048, maximum_seconds=0.15,
        )
        engine.add_at(
            output, onset,
            engine.noise_hit(0.11, options.style, rng, low=80.0,
                             high=1_500.0, decay_seconds=0.025),
            0.08,
        )
    _recorded(
        output, 0.02, foley.CLOTH, options, rng, sources,
        gain=0.17, speed=1.20, low_hz=180.0, high_hz=5_500.0,
        decay_seconds=0.52, maximum_seconds=0.96,
    )
    return output, sources


def _cloth_rustle(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.58)
    sources: list[str] = []
    _recorded(
        output, 0.01, foley.CLOTH, options, rng, sources,
        gain=0.52 + 0.20 * options.intensity, speed=1.04,
        low_hz=100.0, high_hz=7_200.0, decay_seconds=0.31,
        maximum_seconds=0.54,
    )
    engine.add_at(
        output, 0.03,
        engine.whoosh(0.42, options.style, rng, low=350.0, high=4_500.0),
        0.08,
    )
    return output, sources


def _soft_land(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.52)
    sources: list[str] = []
    _recorded(
        output, 0.04, foley.BODY, options, rng, sources,
        gain=0.31 + 0.15 * options.intensity, speed=1.28,
        low_hz=45.0, high_hz=2_100.0, decay_seconds=0.095,
        maximum_seconds=0.30,
    )
    _recorded(
        output, 0.0, foley.CLOTH, options, rng, sources,
        gain=0.28, speed=1.16, low_hz=150.0, high_hz=5_400.0,
        decay_seconds=0.22, maximum_seconds=0.42,
    )
    engine.add_at(output, 0.035, engine.sub_hit(
        0.26, options.style, rng, frequency=88.0, decay_seconds=0.08
    ), 0.12)
    return output, sources


def _hard_land(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.78)
    sources: list[str] = []
    _recorded(
        output, 0.015, foley.BODY, options, rng, sources,
        gain=0.52 + 0.20 * options.intensity, speed=0.82,
        low_hz=35.0, high_hz=3_600.0, decay_seconds=0.18,
        maximum_seconds=0.46,
    )
    _recorded(
        output, 0.025, foley.STONE_HIT, options, rng, sources,
        gain=0.34, speed=0.90, low_hz=90.0, high_hz=7_500.0,
        decay_seconds=0.13, maximum_seconds=0.34,
    )
    engine.add_at(output, 0.0, engine.sub_hit(
        0.56, options.style, rng, frequency=58.0, decay_seconds=0.17
    ), 0.44 + 0.18 * options.intensity)
    engine.add_at(output, 0.005, engine.noise_hit(
        0.30, options.style, rng, low=90.0, high=5_500.0,
        decay_seconds=0.065
    ), 0.20)
    return output, sources


def _wall_kick(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.48)
    sources: list[str] = []
    engine.add_at(output, 0.0, engine.whoosh(
        0.29, options.style, rng, low=240.0, high=6_800.0
    ), 0.34 + 0.15 * options.intensity)
    _recorded(
        output, 0.18, foley.WOOD_HIT, options, rng, sources,
        gain=0.40 + 0.18 * options.intensity, speed=1.15,
        low_hz=80.0, high_hz=6_500.0, decay_seconds=0.075,
        maximum_seconds=0.22,
    )
    engine.add_at(output, 0.18, engine.sub_hit(
        0.23, options.style, rng, frequency=102.0, decay_seconds=0.06
    ), 0.16)
    return output, sources


def _katana_draw(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.72)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.GEAR, options, rng, sources,
        gain=0.22, speed=1.22, low_hz=350.0, high_hz=8_500.0,
        decay_seconds=0.10, maximum_seconds=0.20,
    )
    engine.add_at(output, 0.04, engine.pitch_gesture(
        410.0, 3_900.0, 0.54, options.style,
        harmonics=((1.0, 0.48), (1.52, 0.24), (2.06, 0.10)),
    ), 0.52 + 0.16 * options.intensity)
    engine.add_at(output, 0.05, engine.whoosh(
        0.48, options.style, rng, low=900.0, high=11_000.0
    ), 0.19)
    _recorded(
        output, 0.47, foley.METAL_HIT, options, rng, sources,
        gain=0.16, speed=1.55, low_hz=850.0, high_hz=13_000.0,
        decay_seconds=0.08, maximum_seconds=0.16,
    )
    return output, sources


def _katana_sheathe(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.62)
    sources: list[str] = []
    engine.add_at(output, 0.0, engine.pitch_gesture(
        3_200.0, 440.0, 0.43, options.style,
        harmonics=((1.0, 0.42), (1.47, 0.19)),
    ), 0.43)
    engine.add_at(output, 0.0, engine.whoosh(
        0.39, options.style, rng, low=850.0, high=9_500.0, reverse=True
    ), 0.17)
    _recorded(
        output, 0.39, foley.LOCK, options, rng, sources,
        gain=0.30 + 0.12 * options.intensity, speed=1.45,
        low_hz=220.0, high_hz=8_800.0, decay_seconds=0.07,
        maximum_seconds=0.18,
    )
    return output, sources


def _katana_swing(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.45)
    sources: list[str] = []
    engine.add_at(output, 0.0, engine.whoosh(
        0.42, options.style, rng, low=280.0, high=12_500.0
    ), 0.58 + 0.20 * options.intensity)
    engine.add_at(output, 0.10, engine.pitch_gesture(
        560.0, 2_500.0, 0.24, options.style,
        harmonics=((1.0, 0.20), (2.08, 0.10)),
    ), 0.24)
    _recorded(
        output, 0.0, foley.CLOTH, options, rng, sources,
        gain=0.16, speed=1.42, low_hz=240.0, high_hz=6_800.0,
        decay_seconds=0.14, maximum_seconds=0.30,
    )
    return output, sources


def _katana_clash(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.70)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.METAL_HIT, options, rng, sources,
        gain=0.66 + 0.22 * options.intensity, speed=1.02,
        low_hz=160.0, high_hz=14_000.0, decay_seconds=0.18,
        maximum_seconds=0.34,
    )
    _recorded(
        output, 0.007, foley.METAL_HIT, options, rng, sources,
        gain=0.27, speed=1.34, low_hz=700.0, high_hz=15_000.0,
        decay_seconds=0.10, maximum_seconds=0.22,
    )
    engine.add_at(output, 0.003, engine.blade_ring(
        0.62, options.style, rng, base_hz=790.0
    ), 0.40)
    engine.add_at(output, 0.0, engine.noise_hit(
        0.18, options.style, rng, low=500.0, high=15_000.0,
        decay_seconds=0.035
    ), 0.25)
    return output, sources


def _parry(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.42)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.METAL_HIT, options, rng, sources,
        gain=0.52 + 0.18 * options.intensity, speed=1.36,
        low_hz=480.0, high_hz=15_000.0, decay_seconds=0.075,
        maximum_seconds=0.18,
    )
    engine.add_at(output, 0.0, engine.blade_ring(
        0.36, options.style, rng, base_hz=1_180.0
    ), 0.33)
    engine.add_at(output, 0.025, engine.whoosh(
        0.24, options.style, rng, low=650.0, high=9_500.0, reverse=True
    ), 0.16)
    return output, sources


def _blade_hit(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.58)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.BODY, options, rng, sources,
        gain=0.56 + 0.20 * options.intensity, speed=0.96,
        low_hz=45.0, high_hz=4_200.0, decay_seconds=0.12,
        maximum_seconds=0.34,
    )
    _recorded(
        output, 0.004, foley.METAL_HIT, options, rng, sources,
        gain=0.18, speed=1.62, low_hz=900.0, high_hz=14_000.0,
        decay_seconds=0.05, maximum_seconds=0.13,
    )
    engine.add_at(output, 0.0, engine.sub_hit(
        0.36, options.style, rng, frequency=76.0, decay_seconds=0.10
    ), 0.27)
    return output, sources


def _kunai_throw(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.42)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.GEAR, options, rng, sources,
        gain=0.11, speed=1.65, low_hz=600.0, high_hz=10_000.0,
        decay_seconds=0.045, maximum_seconds=0.10,
    )
    engine.add_at(output, 0.02, engine.whoosh(
        0.36, options.style, rng, low=520.0, high=11_000.0
    ), 0.45 + 0.18 * options.intensity)
    engine.add_at(output, 0.10, engine.pitch_gesture(
        1_900.0, 680.0, 0.22, options.style, decay_seconds=0.10,
        harmonics=((1.0, 0.26), (1.90, 0.08)),
    ), 0.30)
    return output, sources


def _kunai_impact_wood(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.48)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.WOOD_HIT, options, rng, sources,
        gain=0.62 + 0.18 * options.intensity, speed=1.30,
        low_hz=90.0, high_hz=8_000.0, decay_seconds=0.10,
        maximum_seconds=0.25,
    )
    engine.add_at(output, 0.0, engine.blade_ring(
        0.31, options.style, rng, base_hz=1_320.0
    ), 0.16)
    engine.add_at(output, 0.004, engine.noise_hit(
        0.18, options.style, rng, low=300.0, high=7_000.0,
        decay_seconds=0.035
    ), 0.17)
    return output, sources


def _kunai_impact_stone(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.52)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.STONE_HIT, options, rng, sources,
        gain=0.55 + 0.18 * options.intensity, speed=1.23,
        low_hz=120.0, high_hz=10_000.0, decay_seconds=0.11,
        maximum_seconds=0.28,
    )
    _recorded(
        output, 0.002, foley.METAL_HIT, options, rng, sources,
        gain=0.25, speed=1.48, low_hz=650.0, high_hz=14_000.0,
        decay_seconds=0.07, maximum_seconds=0.16,
    )
    engine.add_at(output, 0.0, engine.blade_ring(
        0.42, options.style, rng, base_hz=1_040.0
    ), 0.22)
    return output, sources


def _shuriken_throw(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.58)
    sources: list[str] = []
    engine.add_at(output, 0.0, engine.whoosh(
        0.54, options.style, rng, low=700.0, high=14_000.0
    ), 0.42 + 0.16 * options.intensity)
    for index in range(4):
        engine.add_at(output, 0.035 + index * 0.085, engine.pitch_gesture(
            1_600.0 + index * 130.0,
            2_250.0 + index * 170.0,
            0.13,
            options.style,
            decay_seconds=0.055,
            harmonics=((1.0, 0.18), (2.04, 0.07)),
        ), 0.32)
    return output, sources


def _shuriken_impact_metal(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.62)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.METAL_HIT, options, rng, sources,
        gain=0.72 + 0.18 * options.intensity, speed=1.42,
        low_hz=450.0, high_hz=15_000.0, decay_seconds=0.11,
        maximum_seconds=0.25,
    )
    _recorded(
        output, 0.042, foley.METAL_HIT, options, rng, sources,
        gain=0.27, speed=1.78, low_hz=850.0, high_hz=16_000.0,
        decay_seconds=0.06, maximum_seconds=0.16,
    )
    engine.add_at(output, 0.002, engine.blade_ring(
        0.56, options.style, rng, base_hz=1_240.0
    ), 0.34)
    return output, sources


def _poison_dart(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.34)
    sources: list[str] = []
    engine.add_at(output, 0.0, engine.whoosh(
        0.26, options.style, rng, low=1_200.0, high=15_000.0
    ), 0.34 + 0.16 * options.intensity)
    engine.add_at(output, 0.015, engine.pitch_gesture(
        2_800.0, 1_100.0, 0.20, options.style, decay_seconds=0.07,
        harmonics=((1.0, 0.18),),
    ), 0.30)
    _recorded(
        output, 0.21, foley.WOOD_HIT, options, rng, sources,
        gain=0.12, speed=2.20, low_hz=1_200.0, high_hz=12_000.0,
        decay_seconds=0.025, maximum_seconds=0.07,
    )
    return output, sources


def _smoke_bomb(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.88)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.METAL_HIT, options, rng, sources,
        gain=0.18, speed=1.46, low_hz=400.0, high_hz=10_000.0,
        decay_seconds=0.045, maximum_seconds=0.11,
    )
    engine.add_at(output, 0.015, engine.sub_hit(
        0.44, options.style, rng, frequency=66.0, decay_seconds=0.12
    ), 0.26)
    engine.add_at(output, 0.025, engine.noise_hit(
        0.78, options.style, rng, low=80.0, high=11_000.0,
        decay_seconds=0.24
    ), 0.58 + 0.20 * options.intensity)
    engine.add_at(output, 0.12, engine.whoosh(
        0.60, options.style, rng, low=100.0, high=4_800.0, reverse=True
    ), 0.19)
    return output, sources


def _flash_powder(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.52)
    sources: list[str] = []
    engine.add_at(output, 0.0, engine.noise_hit(
        0.46, options.style, rng, low=600.0, high=18_000.0,
        decay_seconds=0.075
    ), 0.72 + 0.18 * options.intensity)
    engine.add_at(output, 0.0, engine.pitch_gesture(
        5_200.0, 1_700.0, 0.34, options.style, decay_seconds=0.085,
        harmonics=((1.0, 0.24), (1.73, 0.08)),
    ), 0.36)
    return output, sources


def _grappling_hook_throw(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.76)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.GEAR, options, rng, sources,
        gain=0.20, speed=1.26, low_hz=300.0, high_hz=8_500.0,
        decay_seconds=0.08, maximum_seconds=0.17,
    )
    engine.add_at(output, 0.03, engine.whoosh(
        0.62, options.style, rng, low=240.0, high=9_000.0
    ), 0.45 + 0.16 * options.intensity)
    engine.add_at(output, 0.16, engine.pitch_gesture(
        420.0, 1_650.0, 0.40, options.style,
        harmonics=((1.0, 0.22), (2.20, 0.09)),
    ), 0.31)
    return output, sources


def _grappling_hook_latch(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.72)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.METAL_HIT, options, rng, sources,
        gain=0.64 + 0.16 * options.intensity, speed=0.96,
        low_hz=120.0, high_hz=13_000.0, decay_seconds=0.15,
        maximum_seconds=0.30,
    )
    _recorded(
        output, 0.018, foley.WOOD_HIT, options, rng, sources,
        gain=0.25, speed=1.08, low_hz=90.0, high_hz=6_500.0,
        decay_seconds=0.08, maximum_seconds=0.20,
    )
    engine.add_at(output, 0.0, engine.blade_ring(
        0.62, options.style, rng, base_hz=640.0
    ), 0.25)
    engine.add_at(output, 0.24, engine.pitch_gesture(
        280.0, 210.0, 0.32, options.style, decay_seconds=0.18,
        harmonics=((1.0, 0.18), (2.0, 0.08)),
    ), 0.32)
    return output, sources


def _rope_tension(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.92)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.CLOTH, options, rng, sources,
        gain=0.38, speed=0.82, low_hz=80.0, high_hz=3_800.0,
        decay_seconds=0.48, maximum_seconds=0.80,
    )
    _recorded(
        output, 0.08, foley.WOOD_SCRAPE, options, rng, sources,
        gain=0.18, speed=1.18, low_hz=180.0, high_hz=4_500.0,
        decay_seconds=0.38, maximum_seconds=0.64,
    )
    for index, onset in enumerate((0.18, 0.43, 0.67)):
        engine.add_at(output, onset, engine.pitch_gesture(
            190.0 + index * 35.0,
            245.0 + index * 42.0,
            0.19,
            options.style,
            decay_seconds=0.10,
            harmonics=((1.0, 0.20), (2.7, 0.08)),
        ), 0.38 + 0.10 * options.intensity)
    return output, sources


def _bamboo_break(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.90)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.WOOD_BREAK, options, rng, sources,
        gain=0.64 + 0.18 * options.intensity, speed=1.38,
        low_hz=100.0, high_hz=11_000.0, decay_seconds=0.23,
        maximum_seconds=0.55,
    )
    _recorded(
        output, 0.045, foley.WOOD_HIT, options, rng, sources,
        gain=0.25, speed=1.72, low_hz=250.0, high_hz=12_000.0,
        decay_seconds=0.08, maximum_seconds=0.18,
    )
    engine.add_at(output, 0.0, engine.noise_hit(
        0.54, options.style, rng, low=180.0, high=10_000.0,
        decay_seconds=0.10
    ), 0.22)
    return output, sources


def _conceal(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.74)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.CLOTH, options, rng, sources,
        gain=0.46 + 0.12 * options.intensity, speed=1.18,
        low_hz=120.0, high_hz=6_200.0, decay_seconds=0.30,
        maximum_seconds=0.58,
    )
    engine.add_at(output, 0.02, engine.whoosh(
        0.61, options.style, rng, low=120.0, high=5_500.0, reverse=True
    ), 0.25)
    engine.add_at(output, 0.11, engine.pitch_gesture(
        620.0, 145.0, 0.45, options.style, decay_seconds=0.20,
        harmonics=((1.0, 0.20), (1.50, 0.07)),
    ), 0.30)
    return output, sources


def _detected(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.66)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.METAL_HIT, options, rng, sources,
        gain=0.28, speed=1.18, low_hz=350.0, high_hz=12_000.0,
        decay_seconds=0.09, maximum_seconds=0.18,
    )
    engine.add_at(output, 0.02, engine.alert_tone(
        (740.0, 990.0, 740.0), 0.58, options.style
    ), 0.72 + 0.15 * options.intensity)
    engine.add_at(output, 0.0, engine.noise_hit(
        0.24, options.style, rng, low=250.0, high=8_000.0,
        decay_seconds=0.055
    ), 0.20)
    return output, sources


def _takedown(options: Options, rng: np.random.Generator):
    output = engine.canvas(0.92)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.CLOTH, options, rng, sources,
        gain=0.34, speed=0.98, low_hz=90.0, high_hz=5_500.0,
        decay_seconds=0.28, maximum_seconds=0.55,
    )
    _recorded(
        output, 0.24, foley.BODY, options, rng, sources,
        gain=0.68 + 0.17 * options.intensity, speed=0.74,
        low_hz=35.0, high_hz=3_400.0, decay_seconds=0.18,
        maximum_seconds=0.44,
    )
    engine.add_at(output, 0.22, engine.sub_hit(
        0.52, options.style, rng, frequency=54.0, decay_seconds=0.18
    ), 0.40)
    engine.add_at(output, 0.0, engine.whoosh(
        0.32, options.style, rng, low=180.0, high=4_800.0
    ), 0.23)
    return output, sources


def _sliding_door(options: Options, rng: np.random.Generator):
    output = engine.canvas(1.22)
    sources: list[str] = []
    engine.add_at(output, 0.05, engine.whoosh(
        0.98, options.style, rng, low=90.0, high=2_200.0, reverse=True
    ), 0.075)
    engine.add_at(output, 0.13, engine.pitch_gesture(
        118.0, 74.0, 0.82, options.style,
        harmonics=((1.0, 0.10), (2.64, 0.035)),
    ), 0.16)
    _recorded(
        output, 0.0, foley.WOOD_SCRAPE, options, rng, sources,
        gain=0.52 + 0.14 * options.intensity, speed=1.05,
        low_hz=80.0, high_hz=6_800.0, decay_seconds=0.72,
        maximum_seconds=1.08,
    )
    _recorded(
        output, 0.95, foley.LOCK, options, rng, sources,
        gain=0.24, speed=1.32, low_hz=180.0, high_hz=7_500.0,
        decay_seconds=0.08, maximum_seconds=0.18,
    )
    return output, sources


def _secret_panel(options: Options, rng: np.random.Generator):
    output = engine.canvas(1.35)
    sources: list[str] = []
    _recorded(
        output, 0.0, foley.LOCK, options, rng, sources,
        gain=0.28, speed=0.96, low_hz=120.0, high_hz=6_800.0,
        decay_seconds=0.12, maximum_seconds=0.24,
    )
    _recorded(
        output, 0.17, foley.WOOD_SCRAPE, options, rng, sources,
        gain=0.46, speed=0.82, low_hz=55.0, high_hz=4_200.0,
        decay_seconds=0.78, maximum_seconds=1.02,
    )
    _recorded(
        output, 1.03, foley.STONE_HIT, options, rng, sources,
        gain=0.26, speed=0.82, low_hz=70.0, high_hz=5_000.0,
        decay_seconds=0.12, maximum_seconds=0.24,
    )
    engine.add_at(output, 0.12, engine.sub_hit(
        0.92, options.style, rng, frequency=43.0, decay_seconds=0.38
    ), 0.18)
    return output, sources


def _mission_complete(options: Options, rng: np.random.Generator):
    output = engine.canvas(1.20)
    sources: list[str] = []
    notes = (392.0, 494.0, 587.0, 659.0, 784.0)
    engine.add_at(output, 0.0, engine.alert_tone(notes, 0.90, options.style), 0.64)
    for index, frequency in enumerate((392.0, 494.0, 659.0)):
        engine.add_at(output, 0.63, engine.pitch_gesture(
            frequency,
            frequency * 1.006,
            0.48,
            options.style,
            decay_seconds=0.30,
            harmonics=((1.0, 0.18), (2.0, 0.05)),
        ), 0.36 + index * 0.04)
    engine.add_at(output, 0.67, engine.whoosh(
        0.42, options.style, rng, low=1_000.0, high=12_000.0
    ), 0.12 + 0.10 * options.intensity)
    return output, sources


GROUPS = ("movement", "blades", "projectiles", "tools", "stealth-world")


def _cue(
    name: str,
    group: str,
    description: str,
    renderer: Renderer,
    peak_db: float,
    seconds: float,
    layers: tuple[str, ...],
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
        _cue("silent_step", "movement", "one deliberately quiet foot placement",
             _silent_step, -10.0, 0.28,
             ("muted contact", "garment shift"), ("cloth",)),
        _cue("rooftop_run", "movement", "five quick steps over a wooden roof",
             _rooftop_run, -4.5, 1.10,
             ("filtered roof contacts", "continuous clothing"),
             ("wood hit", "cloth")),
        _cue("cloth_rustle", "movement", "stance change beneath layered clothing",
             _cloth_rustle, -11.0, 0.58,
             ("close garment movement", "small air gesture"), ("cloth",)),
        _cue("soft_land", "movement", "controlled crouched landing",
             _soft_land, -6.0, 0.52,
             ("soft body contact", "cloth fold", "restrained weight"),
             ("body impact", "cloth")),
        _cue("hard_land", "movement", "forceful landing on a masonry surface",
             _hard_land, -2.0, 0.78,
             ("body impact", "stone contact", "low weight"),
             ("body impact", "stone hit")),
        _cue("wall_kick", "movement", "push away from a wooden wall",
             _wall_kick, -4.0, 0.48,
             ("approach air", "wood contact", "compact body"), ("wood hit",)),
        _cue("katana_draw", "blades", "blade clears its scabbard and settles ready",
             _katana_draw, -5.0, 0.72,
             ("gear shift", "rising blade scrape", "release click"),
             ("light gear", "metal contact")),
        _cue("katana_sheathe", "blades", "blade returns and locks into the scabbard",
             _katana_sheathe, -6.0, 0.62,
             ("falling blade scrape", "scabbard lock"), ("lock",)),
        _cue("katana_swing", "blades", "fast committed sword cut through open air",
             _katana_swing, -4.0, 0.45,
             ("broad air cut", "blade whistle", "sleeve movement"), ("cloth",)),
        _cue("katana_clash", "blades", "two blades meet with full force",
             _katana_clash, -1.5, 0.70,
             ("dual metal contact", "inharmonic blade ring", "hard edge"),
             ("metal impact",)),
        _cue("parry", "blades", "short deflection that redirects a blade",
             _parry, -3.0, 0.42,
             ("bright metal contact", "short ring", "redirected air"),
             ("metal impact",)),
        _cue("blade_hit", "blades", "blade strike with physical target contact",
             _blade_hit, -2.0, 0.58,
             ("body impact", "blade edge", "low consequence"),
             ("body impact", "metal contact")),
        _cue("kunai_throw", "projectiles", "compact knife released through the air",
             _kunai_throw, -5.0, 0.42,
             ("gear release", "fast flyby", "falling whistle"), ("light gear",)),
        _cue("kunai_impact_wood", "projectiles", "thrown knife embeds in timber",
             _kunai_impact_wood, -3.0, 0.48,
             ("wood contact", "small blade ring", "fiber crack"), ("wood hit",)),
        _cue("kunai_impact_stone", "projectiles", "thrown knife skips off masonry",
             _kunai_impact_stone, -2.5, 0.52,
             ("stone contact", "metal skip", "blade ring"),
             ("stone hit", "metal impact")),
        _cue("shuriken_throw", "projectiles", "spinning throwing star in flight",
             _shuriken_throw, -5.0, 0.58,
             ("high air pass", "four rotating whistles")),
        _cue("shuriken_impact_metal", "projectiles", "throwing star ricochets on metal",
             _shuriken_impact_metal, -1.5, 0.62,
             ("two metal contacts", "high spinning ring"), ("metal impact",)),
        _cue("poison_dart", "projectiles", "small dart launches and punctures wood",
             _poison_dart, -8.0, 0.34,
             ("fine air line", "dart whistle", "tiny puncture"), ("wood hit",)),
        _cue("smoke_bomb", "tools", "pellet strikes and releases a dense smoke cloud",
             _smoke_bomb, -2.0, 0.88,
             ("shell contact", "low pop", "filtered smoke expansion"),
             ("metal contact",)),
        _cue("flash_powder", "tools", "instant bright powder burst without physical weight",
             _flash_powder, -3.0, 0.52,
             ("wide high burst", "falling glare tone")),
        _cue("grappling_hook_throw", "tools", "hook and rope launch upward",
             _grappling_hook_throw, -4.0, 0.76,
             ("gear release", "heavy air arc", "rising rope tone"), ("light gear",)),
        _cue("grappling_hook_latch", "tools", "hook catches a wooden ledge",
             _grappling_hook_latch, -2.0, 0.72,
             ("metal bite", "wood contact", "rope tension"),
             ("metal impact", "wood hit")),
        _cue("rope_tension", "tools", "rope takes weight and fibers settle",
             _rope_tension, -7.0, 0.92,
             ("fiber movement", "surface rub", "three strain squeaks"),
             ("cloth", "wood scrape")),
        _cue("bamboo_break", "tools", "bamboo or light timber snaps under force",
             _bamboo_break, -2.0, 0.90,
             ("recorded wood fracture", "sharp secondary split", "fiber noise"),
             ("wood break", "wood hit")),
        _cue("conceal", "stealth-world", "slip into cover and settle out of sight",
             _conceal, -8.0, 0.74,
             ("garment fold", "inward air", "descending conceal tone"), ("cloth",)),
        _cue("detected", "stealth-world", "enemy awareness snaps into an alert state",
             _detected, -3.0, 0.66,
             ("metal alarm edge", "three-note alert", "noise snap"),
             ("metal impact",)),
        _cue("takedown", "stealth-world", "close silent takedown and controlled fall",
             _takedown, -2.5, 0.92,
             ("cloth struggle", "body fall", "restrained low weight"),
             ("cloth", "body impact")),
        _cue("sliding_door", "stealth-world", "wooden sliding door opens and seats",
             _sliding_door, -7.0, 1.22,
             ("recorded panel scrape", "end-stop latch"), ("wood scrape", "lock")),
        _cue("secret_panel", "stealth-world", "concealed wall panel unlocks and moves",
             _secret_panel, -5.0, 1.35,
             ("hidden latch", "heavy panel scrape", "stone stop", "low movement"),
             ("lock", "wood scrape", "stone hit")),
        _cue("mission_complete", "stealth-world", "original restrained success cadence",
             _mission_complete, -3.5, 1.20,
             ("five-note ascent", "resolved triad", "small air flourish")),
    )
}


def _render_result(
    cue_name: str,
    options: Options,
    rng: np.random.Generator,
) -> RenderResult[list[float]]:
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

    signal, sources = cue.renderer(options, rng)
    expected = engine.frames(cue.seconds)
    fixed = np.zeros(expected, dtype=np.float64)
    fixed[:min(expected, len(signal))] = signal[:expected]
    samples = engine.master(fixed, cue.peak_db, options.style)
    return RenderResult(samples, tuple(dict.fromkeys(sources)))


def render(
    cue_name: str,
    options: Options,
    rng: np.random.Generator,
) -> list[float]:
    """Render one cue while preserving the common plain-sample contract."""

    return _render_result(cue_name, options, rng).samples


def render_with_sources(
    cue_name: str,
    options: Options,
    rng: np.random.Generator,
) -> RenderResult[list[float]]:
    """Render one cue together with every logical recording it consumed."""

    return _render_result(cue_name, options, rng)


def generate(
    cue_name: str,
    *,
    style: str = "shadow",
    intensity: float = 0.85,
    foley_mix: float = 1.0,
    seed: int = 20260721,
) -> list[float]:
    """Return deterministic mono samples for one ninja cue."""

    options = Options(
        style=engine.style_named(style),
        intensity=intensity,
        foley=foley_mix,
    )
    return render(cue_name, options, np.random.default_rng(seed))


def generate_with_sources(
    cue_name: str,
    *,
    style: str = "shadow",
    intensity: float = 0.85,
    foley_mix: float = 1.0,
    seed: int = 20260721,
) -> RenderResult[list[float]]:
    """Return samples together with every logical Foley recording consumed."""

    options = Options(
        style=engine.style_named(style),
        intensity=intensity,
        foley=foley_mix,
    )
    return render_with_sources(cue_name, options, np.random.default_rng(seed))


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
    "one organised render of every ninja movement, weapon, tool, and stealth cue",
    tuple(PackEntry(f"{cue.group}/{cue.name}", cue.name) for cue in CUES.values()),
)

INFILTRATION_NAMES = (
    "silent_step",
    "rooftop_run",
    "cloth_rustle",
    "soft_land",
    "hard_land",
    "wall_kick",
    "conceal",
    "detected",
    "takedown",
    "sliding_door",
    "secret_panel",
)
INFILTRATION = Pack(
    "infiltration",
    "three variations of movement, concealment, detection, and entry cues",
    tuple(PackEntry(name, name, 3) for name in INFILTRATION_NAMES),
)

ARSENAL_NAMES = tuple(
    cue.name for cue in CUES.values() if cue.group in {"blades", "projectiles"}
)
ARSENAL = Pack(
    "arsenal",
    "three variations of every blade and thrown-weapon cue",
    tuple(PackEntry(name, name, 3) for name in ARSENAL_NAMES),
)

TOOLS = Pack(
    "tools",
    "three variations of smoke, flash, grappling, rope, and bamboo cues",
    tuple(
        PackEntry(cue.name, cue.name, 3)
        for cue in CUES.values()
        if cue.group == "tools"
    ),
)

PACKS = {pack.name: pack for pack in (CORE, INFILTRATION, ARSENAL, TOOLS)}


__all__ = [
    "ARSENAL_NAMES",
    "CUES",
    "GROUPS",
    "INFILTRATION_NAMES",
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
