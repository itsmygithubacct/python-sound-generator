"""Sample-free procedural powerup and item-acquisition synthesis.

Renders pickup, powerup, progression, and denial cues in three timbre styles
(arcade / acoustic / sci-fi) as mono 44.1 kHz 16-bit WAV.

    soundgen powerups list
    soundgen powerups render coin --style arcade
    soundgen powerups pack showcase --style acoustic
"""

from .cues import CUES, PACKS, Cue, render
from .engine import SR, midi_to_freq, write_wav
from .styles import DEFAULT_STYLE, STYLES, Style, get_style

__all__ = [
    "CUES",
    "PACKS",
    "SR",
    "Cue",
    "DEFAULT_STYLE",
    "STYLES",
    "Style",
    "get_style",
    "midi_to_freq",
    "render",
    "write_wav",
]
