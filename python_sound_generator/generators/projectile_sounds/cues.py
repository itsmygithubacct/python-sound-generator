"""Cue and pack registries for the projectile-impact generator.

Two cues, because there are only two things here that a recording does better
than a synthesizer: an impact and the debris it throws. Everything else a
projectile does -- the report that launched it, the flyby, the blast -- stays in
``weapons_destruction_generator``, and the packs below *borrow* it rather than
reimplement it, so a game's bank is whole from one command and there is still
exactly one implementation of a rifle shot.
"""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from ..._audio.models import VariationFilenames

from ..._audio.models import RenderResult as Render
from ..._audio.runtime import derive_seed

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..weapons_destruction_generator import cues as weapons

from . import engine


@dataclass(frozen=True)
class Options:
    """Every supported render axis; cues ignore fields they do not honour."""

    projectile: engine.ProjectileProfile = engine.PROJECTILES["bullet"]
    material: engine.MaterialProfile = engine.MATERIALS["metal"]
    intensity: float = 0.85




@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Callable[[random.Random, Options], tuple[np.ndarray, list[str]]]
    peak_db: float
    honours: Honours | str | tuple[str, ...] = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


CUES: dict[str, Cue] = {
    "impact": Cue(
        "impact", "impacts",
        "a projectile striking a target material",
        lambda rng, opts: engine.impact(
            opts.projectile, opts.material, rng, opts.intensity
        ),
        -2.0, "projectile, material, intensity",
    ),
    "debris": Cue(
        "debris", "impacts",
        "a decelerating shower of material fragments",
        lambda rng, opts: engine.debris(opts.material, rng, opts.intensity),
        -5.0, "material, intensity",
    ),
}

GROUPS = ("impacts",)


def render(cue_name: str, options: Options, rng: random.Random) -> Render:
    """Render and master a named cue, keeping the sources it consumed."""

    try:
        cue = CUES[cue_name]
    except KeyError:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        ) from None
    signal, used = cue.render(rng, options)
    return Render(engine.master(signal, cue.peak_db), tuple(used))


def stem_for(cue_name: str, options: Options) -> str:
    """Build a descriptive default artifact name from honoured options."""

    if cue_name == "impact":
        return f"impact_{options.projectile.name}_{options.material.name}"
    if cue_name == "debris":
        return f"debris_{options.material.name}"
    return cue_name


@dataclass(frozen=True)
class PackEntry:
    filename: str
    cue: str
    count: int
    options: Options


@dataclass(frozen=True)
class Pack(VariationFilenames):
    """A game's bank, which does not respect the seam between the generators.

    ``borrows`` names files from a ``weapons_destruction_generator`` pack and
    they are rendered through *its* registry with the same stable seed identity,
    so a borrowed report is byte-identical to the one its own CLI writes. There
    is one implementation of a rifle shot and it is not here.
    """

    name: str
    description: str
    entries: tuple[PackEntry, ...]
    borrows: tuple[str, ...] = ()
    weapons_pack: str | None = None


    def borrowed_entries(self) -> tuple[weapons.PackEntry, ...]:
        if not self.borrows:
            return ()
        source = weapons.PACKS[self.weapons_pack]
        by_name = {entry.filename: entry for entry in source.entries}
        missing = [name for name in self.borrows if name not in by_name]
        if missing:
            raise KeyError(
                f"{self.name} borrows {missing} from weapons_destruction_"
                f"generator pack {self.weapons_pack!r}, which no longer ships "
                f"them"
            )
        return tuple(by_name[name] for name in self.borrows)


def _at(material: str = "metal", projectile: str = "bullet",
        **kwargs) -> Options:
    return Options(
        projectile=engine.PROJECTILES[projectile],
        material=engine.MATERIALS[material],
        **kwargs,
    )


SHOWCASE = Pack(
    "showcase",
    "one bullet impact against every material, plus every debris field",
    tuple(
        PackEntry(f"impact_bullet_{name}", "impact", 1, _at(name))
        for name in engine.MATERIALS
    )
    + tuple(
        PackEntry(f"debris_{name}", "debris", 1, _at(name))
        for name, material in engine.MATERIALS.items()
        if material.debris
    ),
)


# The full cross product, which is the point of the generator: the same bullet
# against six materials is six different sounds, and the same material under a
# bomb and a laser is two more.
MATRIX = Pack(
    "matrix",
    "every projectile against every material, and every debris field",
    tuple(
        PackEntry(f"impact_{pname}_{mname}", "impact", 2, _at(mname, pname))
        for pname in engine.PROJECTILES
        for mname in engine.MATERIALS
    )
    + tuple(
        PackEntry(f"debris_{name}", "debris", 3, _at(name))
        for name, material in engine.MATERIALS.items()
        if material.debris
    ),
)


# Artillery Terrain is artillery onto terrain. The reports and the blasts are
# weapons_destruction_generator's and are borrowed; what is here is what the
# shells land on.
ARTILLERY_TERRAIN = Pack(
    "artillery-terrain",
    "artillery landing on terrain, with the guns borrowed from weapons",
    entries=(
        PackEntry("impact_dirt", "impact", 3, _at("dirt", "shell")),
        PackEntry("impact_stone", "impact", 3, _at("stone", "shell")),
        PackEntry("impact_water", "impact", 2, _at("water", "shell")),
        PackEntry("impact_metal", "impact", 2, _at("metal", "shell")),
        PackEntry("debris_dirt", "debris", 2, _at("dirt")),
        PackEntry("debris_stone", "debris", 2, _at("stone")),
    ),
    borrows=("flyby_shell", "explosion_artillery"),
    weapons_pack="showcase",
)


# The FPS wants a bullet hitting things, and the thing it hits most is people.
ARENA_FPS = Pack(
    "arena-fps",
    "bullet impacts for the FPS, with the firing borrowed from weapons",
    entries=(
        PackEntry("impact_flesh", "impact", 4, _at("flesh", "bullet")),
        PackEntry("impact_metal", "impact", 3, _at("metal", "bullet")),
        PackEntry("impact_wood", "impact", 3, _at("wood", "bullet")),
        PackEntry("impact_stone", "impact", 3, _at("stone", "bullet")),
        PackEntry("impact_dirt", "impact", 3, _at("dirt", "bullet")),
        PackEntry("impact_water", "impact", 2, _at("water", "bullet")),
    ),
    borrows=("fire_rifle", "ricochet_bullet_metal"),
    weapons_pack="showcase",
)


PACKS = {pack.name: pack
         for pack in (SHOWCASE, MATRIX, ARTILLERY_TERRAIN, ARENA_FPS)}


def weapons_seed(entry: weapons.PackEntry, index: int, seed: int) -> int:
    """Return the source provider's stable seed for a borrowed artifact."""

    return derive_seed(
        seed,
        provider=weapons.__package__,
        cue=entry.cue,
        filename=entry.filename,
        variation=index + 1,
    )


def weapons_render(entry: weapons.PackEntry, index: int,
                   seed: int) -> list[float]:
    """Render one borrowed file through weapons_destruction_generator's registry.

    The stable seed identity matches that project's ``cmd_pack``, so a borrowed
    report is the same report, to the byte, as the one its own CLI writes.
    """

    rng = random.Random(weapons_seed(entry, index, seed))
    return weapons.render(entry.cue, entry.options, rng)
