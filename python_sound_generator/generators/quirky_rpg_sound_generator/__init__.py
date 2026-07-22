"""Original procedural and optional cartridge-Foley quirky RPG library."""

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
from .engine import SR, STYLES, Style, style_named, write_wav


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
