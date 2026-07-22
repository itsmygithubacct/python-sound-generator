"""Public Python API for the cyberpunk game sound library.

Importing this module opts into the cyberpunk synthesis engine while the
top-level provider catalog remains lazy.
"""

from .generators.cyberpunk_sound_generator import (
    CUES,
    GROUPS,
    PACKS,
    SR,
    STYLES,
    Cue,
    Options,
    Pack,
    PackEntry,
    Style,
    generate,
    generate_with_sources,
    missing_sources,
    render,
    render_with_sources,
    source_inventory,
    stem_for,
    style_named,
    write_wav,
)


__all__ = [
    "CUES",
    "GROUPS",
    "PACKS",
    "SR",
    "STYLES",
    "Cue",
    "Options",
    "Pack",
    "PackEntry",
    "Style",
    "generate",
    "generate_with_sources",
    "missing_sources",
    "render",
    "render_with_sources",
    "source_inventory",
    "stem_for",
    "style_named",
    "write_wav",
]
