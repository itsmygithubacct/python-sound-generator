"""Deterministic procedural and optional-Foley library for role-playing games.

This provider covers the RPG interactions that sit between the project's
specialised combat, magic, movement, and ambience domains: inventory and loot,
economy and items, quests and progression, dialogue, saving, and rest.
"""

from .cues import (
    CUES,
    GROUPS,
    PACKS,
    Cue,
    Options,
    Pack,
    PackEntry,
    generate,
    generate_with_sources,
    render,
    render_with_sources,
    stem_for,
)
from .engine import SR, THEMES, Theme, theme_named, write_wav


def source_inventory():
    """Load and return the optional Foley inventory only when requested."""

    from .foley import source_inventory as implementation

    return implementation()


def missing_sources() -> tuple[str, ...]:
    from .foley import missing_sources as implementation

    return implementation()


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
