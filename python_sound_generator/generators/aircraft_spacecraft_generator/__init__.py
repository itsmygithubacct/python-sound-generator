"""Procedural aircraft and spacecraft sound, with delegated crash events.

Propulsion, airframe, alarms, and arrivals for anything that leaves the ground.
``car_engine_generator`` models a crankshaft and ``vehicle_foley_generator``
models a contact patch; neither is much use once the wheels are up, so this
models the things an aeroplane has instead -- a propeller, a compressor, a
plume, and a whole structure being shaken by the air going past it.

Every sound is the cross product of a powerplant and an airframe, so a radial
in a warbird and an ion drive in a freighter come out of the same code and
share no timbre. Running is one verb, so ``engine`` is one cue and the profile
decides whether that means pistons, a turbine, a rocket, or a thruster.

Nine of the eleven cues are synthesised here and need only the standard library.
They are the ones with a *rate* -- a firing order, a blade, a shaft, a flow --
and modelling the rate is what makes them convincing. The two that have no
rate, ``touchdown`` and ``crash``, are compatibility delegates to
:mod:`crash_generator`; synthesising them here produced a bell over a beep,
which is what those primitives are.

Continuous cues are seamless by construction rather than by crossfading, and
``verify`` measures the loop points rather than asserting them.

    soundgen aircraft list
    soundgen aircraft render engine --powerplant radial
    soundgen aircraft render cockpit --craft airliner \
        --powerplant turbofan --speed 0.8
    soundgen aircraft verify
    soundgen aircraft pack lander
"""

from .cues import (
    CUES,
    GROUPS,
    LOOPS,
    ONESHOTS,
    PACKS,
    Cue,
    Options,
    Render,
    render,
    render_with_sources,
    stem_for,
)
from .engine import (
    CRAFT,
    MAX_LOOP,
    MIN_LOOP,
    POWERPLANTS,
    SR,
    CraftProfile,
    PowerplantProfile,
    craft_named,
    powerplant_named,
    write_wav,
)

__all__ = [
    "CRAFT",
    "CUES",
    "GROUPS",
    "LOOPS",
    "MAX_LOOP",
    "MIN_LOOP",
    "ONESHOTS",
    "PACKS",
    "POWERPLANTS",
    "SR",
    "CraftProfile",
    "Cue",
    "Options",
    "Render",
    "PowerplantProfile",
    "craft_named",
    "powerplant_named",
    "render",
    "render_with_sources",
    "stem_for",
    "write_wav",
]
