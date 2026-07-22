"""Projectile impacts and debris, built from CC0 recordings.

``weapons_destruction_generator`` fires the projectile; this is what happens
where it lands. The split is not tidiness, it is what each technique is good at.
A report is a pressure event and a flyby is a Doppler rate, and synthesis models
a rate well. An impact has no rate — it is contact, deformation and debris
arriving milliseconds apart — and eleven of that generator's twelve impact
renders were rejected by ear on 2026-07-15 because a sum of decaying sinusoids
is a bell however it is labelled.

Every sound is the cross product of a projectile and a material, so a bullet in
dirt and a bomb on plate come out of the same code and share no timbre. Weight
is varispeed: pitch and decay move together, so a heavy round is a bigger event
rather than a slowed-down small one.

Needs NumPy and SciPy (``requirements.txt``) and the bundled recordings in
``sources/``, whose licensing is recorded in ``sources/provenance.json``.

    soundgen projectiles list
    soundgen projectiles render impact --projectile shell --material dirt
    soundgen projectiles render debris --material stone --count 4
    soundgen projectiles pack arena-fps
"""

from .cues import (
    CUES,
    GROUPS,
    PACKS,
    Cue,
    Options,
    Render,
    render,
    stem_for,
    weapons_render,
    weapons_seed,
)
from .engine import (
    MATERIALS,
    PROJECTILES,
    SR,
    MaterialProfile,
    ProjectileProfile,
    bundled_sources,
    material_contact,
    material_named,
    projectile_named,
    write_wav,
)

__all__ = [
    "CUES",
    "GROUPS",
    "MATERIALS",
    "PACKS",
    "PROJECTILES",
    "SR",
    "Cue",
    "MaterialProfile",
    "Options",
    "ProjectileProfile",
    "Render",
    "bundled_sources",
    "material_named",
    "material_contact",
    "projectile_named",
    "render",
    "stem_for",
    "weapons_render",
    "weapons_seed",
    "write_wav",
]
