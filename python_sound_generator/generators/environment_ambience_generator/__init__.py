"""Dependency-free procedural environmental ambience.

Seamless loops for wind, ocean surf, caves, dungeons, forests, crowds,
machinery and space, plus thunder, rockfalls, church bells and fog horns.

Rain, aquarium ambience, bubbles, splashes and standalone drips live in
``water_sounds_generator`` so those water families have one owner.

Fire and lava are deliberately absent: the synthesized fire here was rejected by
ear on 2026-07-15 and both moved to ``fire_sound_generator``, which is built on
real recordings. Fire has no rate, and procedural DSP loses on chaotic
transients -- see that package's README.

Ocean, wind, birdsong, bells and the horn are ports of earlier researched
prototypes, moved to the standard library and made to loop. :mod:`.syrinx` is the one
model rebuilt rather than copied -- the original's had lost its gamma^2 terms
and never oscillated.

The loops are seamless by construction rather than by crossfading the ends
together. See :mod:`.loopdsp` for what that takes and :func:`.loopdsp.seam` for
how the result is measured; `verify` reports it from the command line.

    soundgen ambience list
    soundgen ambience render wind --intensity 1
    soundgen ambience render cave --seconds 20
    soundgen ambience verify
    soundgen ambience render ocean --seconds 90
    soundgen ambience pack town_shore
"""

from .cues import CUES, GROUPS, LOOPS, ONESHOTS, PACKS, Cue, Options, render
from .engine import MAX_SECONDS, MIN_SECONDS, SR, write_wav
from .loopdsp import Seam, seam

__all__ = [
    "CUES",
    "GROUPS",
    "LOOPS",
    "MAX_SECONDS",
    "MIN_SECONDS",
    "ONESHOTS",
    "PACKS",
    "SR",
    "Cue",
    "Options",
    "Seam",
    "render",
    "seam",
    "write_wav",
]
