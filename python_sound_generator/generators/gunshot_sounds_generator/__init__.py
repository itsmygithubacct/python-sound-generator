"""Firearm context and mechanism Foley composed over existing generators."""

from .cues import CUES, GROUPS, PACKS, Cue, Options, Render, render, stem_for
from .engine import (
    FIREARMS,
    SPACES,
    SUPPRESSIONS,
    SURFACES,
    SR,
    FirearmProfile,
    SpaceProfile,
    SuppressionProfile,
    SurfaceProfile,
    firearm_named,
    space_named,
    suppression_named,
    surface_named,
    write_wav,
)

__all__ = [
    "CUES", "FIREARMS", "GROUPS", "PACKS", "SPACES", "SUPPRESSIONS",
    "SURFACES", "SR", "Cue", "FirearmProfile", "Options", "Render",
    "SpaceProfile", "SuppressionProfile", "SurfaceProfile", "firearm_named",
    "render", "space_named", "stem_for", "suppression_named", "surface_named",
    "write_wav",
]
