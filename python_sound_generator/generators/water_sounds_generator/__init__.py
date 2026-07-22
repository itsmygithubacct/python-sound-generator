"""Water -- entries, drips, bubbles, rain, tanks and wakes -- from CC0 recordings.

The layer functions are exported alongside the cues, following
`stone_noise_generator`: a material generator owns what a material *is* -- its
entry, its spray, its body -- and an event generator composes them.
"""

from .cues import CUES, GROUPS, NOT_MADE, PACKS, Cue, Options, Pack, Render, render
from .engine import (
    MASSES,
    SR,
    Mass,
    body,
    bundled_sources,
    consumed_sources,
    entry,
    local_sources,
    mass_named,
    spray,
    sub,
    write_wav,
)

__all__ = [
    "CUES",
    "GROUPS",
    "MASSES",
    "NOT_MADE",
    "PACKS",
    "SR",
    "Cue",
    "Mass",
    "Options",
    "Pack",
    "Render",
    "body",
    "bundled_sources",
    "consumed_sources",
    "entry",
    "local_sources",
    "mass_named",
    "render",
    "spray",
    "sub",
    "write_wav",
]
