"""Game-facing pack registry for the door Foley generator."""

from __future__ import annotations

from ..._audio.models import VariationFilenames

from dataclasses import dataclass


@dataclass(frozen=True)
class Options:
    action: str = "close"
    material: str = "heavy_wood"
    intensity: float = 0.85
    age: float = 0.65
    duration: float | None = None
    room: float = 0.08
    stereo: bool = False


@dataclass(frozen=True)
class PackEntry:
    filename: str
    count: int
    options: Options


@dataclass(frozen=True)
class Pack(VariationFilenames):
    name: str
    description: str
    entries: tuple[PackEntry, ...]



FLIGHT = Pack(
    "flight",
    "the dry metal cockpit toggle Flight Simulator loads as switch.wav",
    (
        PackEntry(
            "switch",
            1,
            Options(
                action="latch",
                material="metal",
                intensity=0.82,
                age=0.0,
                duration=0.12,
                room=0.0,
            ),
        ),
    ),
)


PACKS = {pack.name: pack for pack in (FLIGHT,)}
