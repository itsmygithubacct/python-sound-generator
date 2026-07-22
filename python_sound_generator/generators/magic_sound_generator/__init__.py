"""Dependency-free procedural magic and energy sound effects.

Cast, projectile and state cues for a game's magic bank -- a beam, a bolt, a
teleport; a cast, a charge, a heal, sparkle; a shield ward (with a seamless
loopable hum), a dissolve, a curse -- each an element (arcane, holy, shadow,
nature) away from four variants of itself.

Everything is synthesized from the standard-library DSP core in
``combat_generators`` and the seam-safe loop extensions in
``environment_ambience_generator``. There are no recordings, so nothing here
carries a licence. Magic is the pure case of what procedural DSP wins on -- a
rate the ear can follow -- which is why it is synthesized and fire is not; see
the module docstrings for that argument in full.

    soundgen magic list
    soundgen magic render beam --count 5
    soundgen magic render charge --style holy --play
    soundgen magic render shield_loop --seconds 12
    soundgen magic verify
    soundgen magic pack fantasy-adventure
"""

from .cues import (
    CUES,
    GROUPS,
    LOOPS,
    MAX_SECONDS,
    MIN_SECONDS,
    ONESHOTS,
    PACKS,
    Cue,
    Options,
    Pack,
    PackEntry,
    Render,
    render,
    stem_for,
)
from .engine import (
    SR,
    STYLES,
    Style,
    master,
    style_named,
    write_wav,
)

__all__ = [
    "CUES",
    "GROUPS",
    "LOOPS",
    "MAX_SECONDS",
    "MIN_SECONDS",
    "ONESHOTS",
    "PACKS",
    "SR",
    "STYLES",
    "Cue",
    "Options",
    "Pack",
    "PackEntry",
    "Render",
    "Style",
    "master",
    "render",
    "stem_for",
    "style_named",
    "write_wav",
]
