"""Hybrid procedural and CC0-Foley sound library for ninja games."""

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
from .foley import missing_sources, source_inventory


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
