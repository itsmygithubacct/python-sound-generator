"""Sample-free procedural UI and game-state synthesis.

An extension of :mod:`powerup_generator`: it imports that package's engine,
gestures and styles unchanged, and supplies its own cue registry. The DSP and
the three timbre houses (arcade / acoustic / sci-fi) are shared, so a game using
both banks sounds like one bank.

What is *not* shared is the pitch language. A pickup means reward and rises; a
menu keypress means nothing in particular and must not. See :mod:`.cues`.

Renders menu, state, feedback, round and terminal cues as mono 44.1 kHz 16-bit
WAV.

    soundgen ui list
    soundgen ui render menu_accept --style arcade
    soundgen ui render combo_step --ladder 8
    soundgen ui pack showcase --style acoustic
"""

from ..powerup_generator.engine import SR, midi_to_freq, write_wav
from ..powerup_generator.styles import DEFAULT_STYLE, STYLES, Style, get_style

from .cues import CUES, GROUPS, MAX_STEP, PACKS, Cue, render, scale_step

__all__ = [
    "CUES",
    "GROUPS",
    "MAX_STEP",
    "PACKS",
    "SR",
    "Cue",
    "DEFAULT_STYLE",
    "STYLES",
    "Style",
    "get_style",
    "midi_to_freq",
    "render",
    "scale_step",
    "write_wav",
]
