"""Public Python API for the role-playing game sound library.

Importing this module opts into the RPG synthesis engine while the package's
top-level provider catalog remains lazy.
"""

from .generators.rpg_sound_generator import (
    CUES,
    GROUPS,
    PACKS,
    SR,
    THEMES,
    Cue,
    Options,
    Pack,
    PackEntry,
    Theme,
    generate,
    generate_with_sources,
    missing_sources,
    render,
    render_with_sources,
    source_inventory,
    stem_for,
    theme_named,
    write_wav,
)


__all__ = [
    "CUES",
    "GROUPS",
    "PACKS",
    "SR",
    "THEMES",
    "Cue",
    "Options",
    "Pack",
    "PackEntry",
    "Theme",
    "generate",
    "generate_with_sources",
    "missing_sources",
    "render",
    "render_with_sources",
    "source_inventory",
    "stem_for",
    "theme_named",
    "write_wav",
]
