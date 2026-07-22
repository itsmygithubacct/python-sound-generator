"""Cue, profile, and consumer-pack registry for computer sounds."""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from ..._audio.models import VariationFilenames

from ..._audio.models import RenderResult as Render

import random
from dataclasses import dataclass
from typing import Callable

from ..._audio import dsp
from ..._audio import loops as loopdsp

from . import engine


@dataclass(frozen=True)
class Options:
    size: engine.Size = engine.SIZES["desktop"]
    region: engine.Region = engine.REGIONS["ntsc"]
    seconds: float = 4.0
    intensity: float = 0.7




@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    renderer: Callable
    peak_db: float
    loop: bool = False
    sourced: bool = False
    honours: Honours | str | tuple[str, ...] = "size, region, intensity"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


CUES: dict[str, Cue] = {
    "fan": Cue(
        "fan", "loop", "cooling-fan blade-pass loop, by machine size",
        engine.fan, -13.0, loop=True,
        honours="size, seconds, intensity",
    ),
    "drive": Cue(
        "drive", "loop", "hard-drive spindle and bearing loop, by RPM",
        engine.drive, -16.0, loop=True, sourced=True,
        honours="size, seconds, intensity",
    ),
    "psu": Cue(
        "psu", "loop", "PSU transformer, mains harmonics, and coil whine",
        engine.psu, -17.0, loop=True,
        honours="size, region, seconds, intensity",
    ),
    "key": Cue(
        "key", "mechanism", "physical terminal or keyboard key contact",
        engine.key, -8.0, sourced=True,
    ),
    "switch": Cue(
        "switch", "mechanism", "physical latching power switch",
        engine.switch, -7.0, sourced=True,
    ),
    "seek": Cue(
        "seek", "mechanism", "hard-drive actuator seek chatter",
        engine.seek, -10.0, sourced=True,
    ),
    "dial": Cue(
        "dial", "signal", "regional telephone dial tone",
        engine.dial, -12.0,
    ),
    "modem": Cue(
        "modem", "signal", "real dial-up modem connect handshake",
        engine.modem, -8.0, sourced=True,
    ),
    "power_on": Cue(
        "power_on", "transition", "real PC switch-on and mechanism run-up",
        engine.power_on, -5.0, sourced=True,
    ),
    "boot": Cue(
        "boot", "transition", "real PC startup and disk-drive check",
        engine.boot, -7.0, sourced=True,
    ),
    "spin_up": Cue(
        "spin_up", "transition", "drive motor accelerating to spindle rate",
        engine.spin_up, -10.0, sourced=True,
    ),
    "shutdown": Cue(
        "shutdown", "transition", "hard-drive spindle coast-down",
        engine.shutdown, -9.0, sourced=True,
    ),
}


GROUPS = ("loop", "mechanism", "signal", "transition")
LOOPS = tuple(name for name, cue in CUES.items() if cue.loop)
ONESHOTS = tuple(name for name, cue in CUES.items() if not cue.loop)


def render(cue_name: str, options: Options, rng: random.Random) -> Render:
    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    if not 0.05 <= options.intensity <= 1.0:
        raise ValueError("intensity must be between 0.05 and 1")
    if cue.loop:
        raw = cue.renderer(
            options.size, options.region, options.seconds, rng,
            options.intensity,
        )
        if cue.sourced:
            samples, sources = raw
        else:
            samples, sources = raw, ()
        return Render(
            loopdsp.master(samples, cue.peak_db), tuple(sources)
        )
    raw = cue.renderer(
        options.size, options.region, rng, options.intensity
    )
    if cue.sourced:
        samples, sources = raw
    else:
        samples, sources = raw, ()
    return Render(dsp.master(samples, cue.peak_db), tuple(sources))


@dataclass(frozen=True)
class PackEntry:
    filename: str
    cue: str
    count: int = 1
    options: Options = Options()


@dataclass(frozen=True)
class Pack(VariationFilenames):
    name: str
    description: str
    entries: tuple[PackEntry, ...]



SHOWCASE = Pack(
    "showcase",
    "one representative render of every computer cue",
    tuple(PackEntry(name, name) for name in CUES),
)


ZCOM = Pack(
    "zcom",
    "machine-world audio bank for the named X-COM-like consumer",
    (
        PackEntry("fan_loop", "fan", options=Options(seconds=4.0)),
        PackEntry("drive_loop", "drive", options=Options(seconds=4.0)),
        PackEntry("psu_loop", "psu", options=Options(seconds=4.0)),
        PackEntry("terminal_key", "key", count=3),
        PackEntry("power_switch", "switch"),
        PackEntry("drive_seek", "seek", count=3),
        PackEntry("dial_tone", "dial"),
        PackEntry("modem_connect", "modem"),
        PackEntry("power_on", "power_on"),
        PackEntry("boot", "boot"),
        PackEntry("drive_spin_up", "spin_up"),
        PackEntry("shutdown", "shutdown"),
    ),
)


PACKS = {pack.name: pack for pack in (SHOWCASE, ZCOM)}
