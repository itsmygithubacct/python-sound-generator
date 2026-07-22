"""Vehicle foley, with crash events delegated to the central composer.

The foley half of a driving soundscape. ``car_engine_generator`` owns the
engine; this owns everything the engine is bolted to and everything the tires
are touching. A game layers the two.

Every sound is the cross product of a vehicle and a surface, so a monster truck
on mud and a kart on concrete come out of the same code and share no timbre.
Continuous foley renders as a seamless loop, built on the shared loop core in
:mod:`environment_ambience_generator.loopdsp`; events render as one-shots.

    soundgen vehicles list
    soundgen vehicles render roll --surface gravel --speed 0.8
    soundgen vehicles render crash --vehicle truck
    soundgen vehicles pack offroad
"""

from .cues import (
    CUES,
    GROUPS,
    LOOPS,
    PACKS,
    Cue,
    Options,
    Render,
    render,
    render_with_sources,
    stem_for,
)
from .engine import (
    MAX_LOOP,
    MIN_LOOP,
    SR,
    SURFACES,
    VEHICLES,
    SurfaceProfile,
    VehicleProfile,
    surface_named,
    vehicle_named,
    write_wav,
)

__all__ = [
    "CUES",
    "GROUPS",
    "LOOPS",
    "MAX_LOOP",
    "MIN_LOOP",
    "PACKS",
    "SR",
    "SURFACES",
    "VEHICLES",
    "Cue",
    "Options",
    "Render",
    "SurfaceProfile",
    "VehicleProfile",
    "render",
    "render_with_sources",
    "stem_for",
    "surface_named",
    "vehicle_named",
    "write_wav",
]
