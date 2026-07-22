"""Stone -- struck, cracked, shattered, ground and falling -- from CC0 recordings.

The four layer functions are exported alongside the cues on purpose. This
material provider is a foundation for event providers such as `crash_generator`
and `projectile_sounds`: a material generator owns what stone is, and an event
generator composes it. `contact`, `fracture`, `debris`, and `sub` form that
contract.
"""

from .cues import CUES, GROUPS, PACKS, Cue, Options, Pack, Render, render
from .engine import (
    SCALES,
    SR,
    STONES,
    Scale,
    Stone,
    bundled_sources,
    contact,
    debris,
    fracture,
    scale_named,
    stone_named,
    sub,
    write_wav,
)

__all__ = [
    "CUES",
    "GROUPS",
    "PACKS",
    "SCALES",
    "SR",
    "STONES",
    "Cue",
    "Options",
    "Pack",
    "Render",
    "Scale",
    "Stone",
    "bundled_sources",
    "contact",
    "debris",
    "fracture",
    "render",
    "scale_named",
    "stone_named",
    "sub",
    "write_wav",
]
