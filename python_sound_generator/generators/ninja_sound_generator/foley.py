"""Curated CC0 Foley used by the ninja sound library.

The recordings remain owned by their material providers.  This module selects
and reshapes them by logical path, so manifests retain the original filename
and provenance while the package stores no duplicate WAV bytes.
"""

from __future__ import annotations

from ..._audio.foley_bank import (
    PROVIDER_ROOT as PROVIDER_ROOT,
    FoleySource,
    choose,
    inventory,
    sources as _sources,
    take,
    unresolved,
)


CLOTH = _sources(
    "creature_foley_generator",
    "artisticdude_cloth.wav",
    "artisticdude_cloth_heavy.wav",
)

GEAR = _sources(
    "creature_foley_generator",
    "artisticdude_armor_light.wav",
    "artisticdude_chainmail.wav",
)

BODY = _sources(
    "unarmed_foley_generator",
    *(f"independent_nu_hit_{index:02d}.wav" for index in (2, 7, 8, 20, 29, 34)),
)

WOOD_HIT = _sources(
    "wood_noise_generator",
    *(f"rubberduck_wood_hit_{index:02d}.wav" for index in range(1, 10)),
    "rubberduck_break_wood_hit_01.wav",
    "rubberduck_break_wood_hit_02.wav",
)

WOOD_BREAK = _sources(
    "wood_noise_generator",
    "rubberduck_break_wood_breaking_01.wav",
    "rubberduck_break_wood_breaking_02.wav",
    "rubberduck_break_wood_breaking_03.wav",
)

STONE_HIT = _sources(
    "stone_noise_generator",
    "rubberduck_sfx2_stones_01.wav",
    "rubberduck_sfx2_stones_02.wav",
    "rubberduck_sfx2_stones_03.wav",
)

METAL_HIT = _sources(
    "aircraft_spacecraft_generator",
    *(f"kenney_impact_metal_light_{index:03d}.wav" for index in range(5)),
)

WOOD_SCRAPE = _sources(
    "wood_noise_generator",
    "lampeight_wood_panel_scrape.wav",
    "rutgermuller_wood_plank_scraping.wav",
)

LOCK = _sources("door_sound_generator", "laleksic_lock.wav")

POOLS: dict[str, tuple[FoleySource, ...]] = {
    "cloth": CLOTH,
    "gear": GEAR,
    "body": BODY,
    "wood_hit": WOOD_HIT,
    "wood_break": WOOD_BREAK,
    "stone_hit": STONE_HIT,
    "metal_hit": METAL_HIT,
    "wood_scrape": WOOD_SCRAPE,
    "lock": LOCK,
}


def source_inventory() -> tuple[FoleySource, ...]:
    """Return every approved source this provider may consume."""

    return inventory(POOLS)


def missing_sources() -> tuple[str, ...]:
    """Return unresolved logical recordings, suitable for diagnostics."""

    return unresolved(source_inventory())


__all__ = [
    "BODY",
    "CLOTH",
    "GEAR",
    "LOCK",
    "METAL_HIT",
    "POOLS",
    "STONE_HIT",
    "WOOD_BREAK",
    "WOOD_HIT",
    "WOOD_SCRAPE",
    "FoleySource",
    "choose",
    "missing_sources",
    "source_inventory",
    "take",
]
