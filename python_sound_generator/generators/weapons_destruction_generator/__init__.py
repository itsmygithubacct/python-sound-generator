"""Dependency-free procedural weapons and destruction sound effects."""

from .cues import CUES, GROUPS, PACKS, Cue, Options, render
from .engine import (
    EXPLOSIVES,
    FIREARMS,
    MATERIALS,
    PROJECTILES,
    SR,
    ExplosiveProfile,
    FirearmProfile,
    ImpactMaterial,
    ProjectileProfile,
    write_wav,
)

__all__ = [
    "CUES",
    "EXPLOSIVES",
    "FIREARMS",
    "GROUPS",
    "MATERIALS",
    "PACKS",
    "PROJECTILES",
    "SR",
    "Cue",
    "ExplosiveProfile",
    "FirearmProfile",
    "ImpactMaterial",
    "Options",
    "ProjectileProfile",
    "render",
    "write_wav",
]
