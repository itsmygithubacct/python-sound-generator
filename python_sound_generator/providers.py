"""Lazy registry for the unified sound-generator providers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, Sequence

from ._audio.assets import AssetCatalog, resolve_asset
from .generators import PROVIDER_ROOT


ProviderMain = Callable[[Sequence[str] | None], int | None]
_ASSET_SUFFIXES = frozenset({".html", ".json", ".mjs", ".wav"})


@dataclass(frozen=True, slots=True)
class Provider:
    """One isolated sound domain behind the public dispatcher."""

    slug: str
    entrypoint: str
    description: str
    aliases: tuple[str, ...] = ()
    dependency: str = "stdlib"
    required_files: tuple[str, ...] = ()

    @property
    def module_name(self) -> str:
        return self.entrypoint.partition(":")[0]

    @property
    def package_name(self) -> str:
        return self.module_name.partition(".")[0]

    @property
    def directory(self) -> Path:
        return PROVIDER_ROOT / self.package_name

    @property
    def source_directory(self) -> Path:
        return self.directory / "sources"

    @property
    def entrypoint_file(self) -> Path:
        module_name = self.module_name
        return PROVIDER_ROOT.joinpath(*module_name.split(".")).with_suffix(".py")

    def load(self) -> ProviderMain:
        """Import this provider inside the package's private namespace."""
        module_name, separator, attribute = self.entrypoint.partition(":")
        if not separator:
            raise RuntimeError(f"invalid provider entrypoint: {self.entrypoint}")
        module = import_module(f"{__package__}.generators.{module_name}")
        main = getattr(module, attribute)
        if not callable(main):
            raise TypeError(f"provider entrypoint is not callable: {self.entrypoint}")
        return main

    def invoke(self, args: Sequence[str] = ()) -> int:
        """Run this provider and normalize its process-style return code."""
        try:
            result = self.load()(list(args))
        except SystemExit as error:
            if error.code is None:
                return 0
            if isinstance(error.code, int):
                return error.code
            print(error.code, file=sys.stderr)
            return 1
        return int(result or 0)

    def source_file_count(self) -> int:
        """Return the number of files in the conventional ``sources`` bank."""
        local = (
            sum(path.is_file() for path in self.source_directory.rglob("*"))
            if self.source_directory.is_dir()
            else 0
        )
        shared = sum(
            logical.parent == self.source_directory
            for logical, _ in AssetCatalog.default().aliases_for_provider(
                self.package_name
            )
        )
        return local + shared

    def asset_entries(self) -> tuple[tuple[Path, Path], ...]:
        """Return logical provider paths paired with their physical storage."""

        local: dict[Path, Path] = {}
        if self.directory.is_dir():
            local = {
                path: path
                for path in self.directory.rglob("*")
                if path.is_file() and path.suffix.lower() in _ASSET_SUFFIXES
            }
        for logical, physical in AssetCatalog.default().aliases_for_provider(
            self.package_name
        ):
            local.setdefault(logical, physical)
        return tuple(sorted(local.items(), key=lambda item: item[0].as_posix()))

    def asset_files(self) -> tuple[Path, ...]:
        """Return packaged data owned by this provider, in stable order."""
        return tuple(physical for _, physical in self.asset_entries())

    def audio_file_count(self) -> int:
        return sum(path.suffix.lower() == ".wav" for path in self.asset_files())

    def asset_size(self) -> int:
        return sum(path.stat().st_size for path in self.asset_files())

    def provenance_files(self) -> tuple[Path, ...]:
        return tuple(
            physical
            for logical, physical in self.asset_entries()
            if logical.name in {"manifest.json", "provenance.json"}
        )

    def missing_runtime_files(self) -> tuple[str, ...]:
        return tuple(
            relative for relative in self.required_files
            if not resolve_asset(self.directory / relative).is_file()
        )

    def validation_errors(self) -> tuple[str, ...]:
        """Check entrypoint and packaged-data integrity without rendering audio."""
        errors: list[str] = []
        if not self.directory.is_dir():
            return ("provider directory is missing",)
        if not (self.directory / "__init__.py").is_file():
            errors.append("provider package is missing __init__.py")
        if not self.entrypoint_file.is_file():
            errors.append(
                f"entrypoint module is missing: "
                f"{self.entrypoint_file.relative_to(PROVIDER_ROOT)}"
            )

        for relative in self.required_files:
            required = Path(relative)
            if required.is_absolute() or ".." in required.parts:
                errors.append(f"invalid required-file path: {relative}")
            elif not resolve_asset(self.directory / required).is_file():
                errors.append(f"missing runtime file: {relative}")

        for logical, path in self.asset_entries():
            relative = logical.relative_to(self.directory)
            if logical.suffix.lower() == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    errors.append(f"invalid JSON {relative}: {error}")
                else:
                    if not isinstance(payload, dict):
                        errors.append(f"JSON root is not an object: {relative}")
                    elif path.name == "provenance.json" and not isinstance(
                        payload.get("source_audio_policy"), str
                    ):
                        errors.append(
                            f"provenance has no source_audio_policy: {relative}"
                        )
                    elif path.name == "manifest.json" and "layers" in payload:
                        if not payload.get("source") or not payload.get("license"):
                            errors.append(
                                f"sample manifest has no source or license: {relative}"
                            )
                        layers = payload.get("layers")
                        if not isinstance(layers, list):
                            errors.append(
                                f"sample manifest layers is not an array: {relative}"
                            )
                        else:
                            for index, layer in enumerate(layers):
                                filename = (
                                    layer.get("file")
                                    if isinstance(layer, dict)
                                    else None
                                )
                                candidate = (
                                    Path(filename)
                                    if isinstance(filename, str)
                                    else None
                                )
                                if (
                                    candidate is None
                                    or candidate.is_absolute()
                                    or ".." in candidate.parts
                                    or not resolve_asset(
                                        logical.parent / candidate
                                    ).is_file()
                                ):
                                    errors.append(
                                        f"invalid sample layer {index} in {relative}"
                                    )
            elif logical.suffix.lower() == ".wav":
                try:
                    with path.open("rb") as handle:
                        header = handle.read(12)
                except OSError as error:
                    errors.append(f"unreadable WAV {relative}: {error}")
                else:
                    if (
                        len(header) != 12
                        or header[:4] not in {b"RIFF", b"RF64"}
                        or header[8:] != b"WAVE"
                    ):
                        errors.append(f"invalid WAV container: {relative}")
        return tuple(errors)

    def as_dict(self) -> dict[str, object]:
        assets = self.asset_files()
        provenance = self.provenance_files()
        return {
            "name": self.slug,
            "module": self.package_name,
            "description": self.description,
            "aliases": list(self.aliases),
            "dependency": self.dependency,
            "source_files": self.source_file_count(),
            "audio_files": sum(
                logical.suffix.lower() == ".wav"
                for logical, _ in self.asset_entries()
            ),
            "asset_files": len(assets),
            "asset_bytes": sum(path.stat().st_size for path in assets),
            "provenance": bool(provenance),
            "provenance_files": len(provenance),
            "missing_runtime_files": list(self.missing_runtime_files()),
        }


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        "aircraft",
        "aircraft_spacecraft_generator.cli:main",
        "aircraft and spacecraft engines, motion, touchdown, and crash cues",
        ("aircraft-spacecraft", "aircraft_spacecraft_generator"),
        "stdlib; crash cues use numpy/scipy",
    ),
    Provider(
        "ambience",
        "environment_ambience_generator.cli:main",
        "seamless environmental beds and environmental one-shots",
        ("environment", "environment-ambience", "environment_ambience_generator"),
    ),
    Provider(
        "appliances",
        "household_appliance_generator.cli:main",
        "recording-driven household appliance loops and one-shots",
        ("household", "household-appliance", "household_appliance_generator"),
        "numpy/scipy",
    ),
    Provider(
        "car-engine",
        "car_engine_generator.cli:main",
        "procedural and recording-driven combustion-engine performances",
        ("car", "car_engine_generator"),
        "stdlib procedural; numpy/scipy sample-based",
        ("sample_banks/mini_cooper_s/manifest.json",),
    ),
    Provider(
        "combat",
        "combat_generators.combat_sfx:main",
        "melee weapons, armor contacts, combat movement, and falls",
        ("combat_generators",),
    ),
    Provider(
        "computer",
        "computer_sounds_generator.cli:main",
        "computer machinery, terminal input, modem, and power transitions",
        ("computers", "computer-sounds", "computer_sounds_generator"),
    ),
    Provider(
        "crash",
        "crash_generator.cli:main",
        "recording-driven collision, crash, and debris composition",
        ("crashes", "crash_generator"),
        "numpy/scipy",
    ),
    Provider(
        "creature",
        "creature_foley_generator.cli:main",
        "creature movement, contact, vocal, and lifecycle Foley",
        ("creatures", "creature-foley", "creature_foley_generator"),
        "numpy/scipy",
    ),
    Provider(
        "cyberpunk",
        "cyberpunk_sound_generator.cli:main",
        "cyberpunk interfaces, networks, hacking, cyberware, and world cues",
        (
            "cyberpunk-game",
            "cyberpunk-sounds",
            "cyberpunk_sound_generator",
        ),
    ),
    Provider(
        "doors",
        "door_sound_generator.cli:main",
        "door, hinge, latch, lock, and slam Foley",
        ("door", "door-sounds", "door_sound_generator"),
        "numpy/scipy",
    ),
    Provider(
        "fire",
        "fire_sound_generator.cli:main",
        "recording-driven flames, fireballs, explosions, and lava flow",
        ("fire-sounds", "fire_sound_generator"),
        "numpy/scipy",
    ),
    Provider(
        "footsteps",
        "footstep_generator.engine:main",
        "footsteps parameterized by footwear and surface",
        ("footstep", "footstep_generator"),
        "numpy/scipy",
    ),
    Provider(
        "gunshots",
        "gunshot_sounds_generator.cli:main",
        "firearm reports, mechanisms, reloads, contexts, and suppression",
        ("guns", "gunshot-sounds", "gunshot_sounds_generator"),
        "numpy/scipy",
    ),
    Provider(
        "human-voice",
        "human_voice_generator.cli:main",
        "human exertion, pain, death, and reaction voices",
        ("human", "human_voice_generator"),
        "numpy/scipy",
    ),
    Provider(
        "lava",
        "lava_sound_generator.cli:main",
        "lava impacts, splashes, bubbles, and molten motion",
        ("lava-sounds", "lava_sound_generator"),
        "numpy/scipy",
    ),
    Provider(
        "magic",
        "magic_sound_generator.cli:main",
        "procedural spell, portal, enchantment, and magical-state cues",
        ("magic-sounds", "magic_sound_generator"),
    ),
    Provider(
        "magic-energy",
        "magic_energy_generator.cli:main",
        "fantasy energy beams and related game packs",
        ("energy", "magic_energy_generator"),
    ),
    Provider(
        "music",
        "music_sound_generator.cli:main",
        "short musical cues, stingers, and transitions",
        ("music-sounds", "music_sound_generator"),
        "numpy/scipy",
        ("recorder/record_cue.mjs", "recorder/record_template.html"),
    ),
    Provider(
        "ninja",
        "ninja_sound_generator.cli:main",
        "hybrid Foley and procedural ninja movement, weapons, tools, and stealth cues",
        (
            "ninja-game",
            "ninja-sounds",
            "shinobi-sounds",
            "ninja_sound_generator",
        ),
        "numpy/scipy; reused CC0 Foley",
        ("sources/provenance.json",),
    ),
    Provider(
        "power-tools",
        "power_tool_generator.cli:main",
        "recording-driven powered-tool Foley",
        ("power-tool", "power_tool_generator"),
        "numpy/scipy",
    ),
    Provider(
        "powerups",
        "powerup_generator.cli:main",
        "pickup, reward, denial, warning, and quest-completion cues",
        ("powerup", "powerup_generator"),
        "numpy/scipy",
    ),
    Provider(
        "projectiles",
        "projectile_sounds.cli:main",
        "projectile impacts across projectile and material families",
        ("projectile", "projectile_sounds"),
        "numpy/scipy",
    ),
    Provider(
        "quirky-rpg",
        "quirky_rpg_sound_generator.cli:main",
        "original quirky 16-bit RPG interfaces, dialogue, psychic, battle, and world cues",
        (
            "16-bit-rpg",
            "retro-jrpg",
            "surreal-rpg",
            "quirky_rpg_sound_generator",
        ),
    ),
    Provider(
        "rpg",
        "rpg_sound_generator.cli:main",
        "RPG inventory, loot, economy, quest, progression, and world cues",
        (
            "role-playing",
            "role-playing-game",
            "rpg-sounds",
            "rpg_sound_generator",
        ),
    ),
    Provider(
        "stone",
        "stone_noise_generator.cli:main",
        "stone contact, fracture, grinding, falling, and debris",
        ("stone-noise", "stone_noise_generator"),
        "numpy/scipy",
    ),
    Provider(
        "ui",
        "ui_game_state_generator.cli:main",
        "menu, feedback, state, round, and terminal game cues",
        ("game-state", "ui-game-state", "ui_game_state_generator"),
        "numpy/scipy",
    ),
    Provider(
        "unarmed",
        "unarmed_foley_generator.cli:main",
        "punch, kick, swing, block, body-impact, and fall Foley",
        ("unarmed-foley", "unarmed_foley_generator"),
        "numpy/scipy",
    ),
    Provider(
        "vehicles",
        "vehicle_foley_generator.cli:main",
        "tire, suspension, chassis, boost, and vehicle-motion Foley",
        ("vehicle", "vehicle-foley", "vehicle_foley_generator"),
        "stdlib; crash cues use numpy/scipy",
    ),
    Provider(
        "voice-foley",
        "voice_foley_generator.cli:main",
        "composed voice-plus-body game-event Foley",
        ("voice", "voice_foley_generator"),
        "numpy/scipy",
    ),
    Provider(
        "water",
        "water_sounds_generator.cli:main",
        "water splashes, pours, movement, rain, bubbles, and aquarium beds",
        ("water-sounds", "water_sounds_generator"),
        "numpy/scipy",
    ),
    Provider(
        "weapons",
        "weapons_destruction_generator.cli:main",
        "firearms, artillery, energy weapons, flybys, ricochets, and explosions",
        ("weapons-destruction", "weapons_destruction_generator"),
    ),
    Provider(
        "wood",
        "wood_noise_generator.cli:main",
        "wood contact, fracture, stress, scrape, and debris Foley",
        ("wood-noise", "wood_noise_generator"),
        "numpy/scipy",
    ),
)


def _build_lookup() -> dict[str, Provider]:
    lookup: dict[str, Provider] = {}
    for provider in PROVIDERS:
        for name in (provider.slug, *provider.aliases):
            normalized = name.strip().lower().replace("_", "-")
            existing = lookup.get(normalized)
            if existing is not None and existing is not provider:
                raise RuntimeError(f"provider alias collision: {name}")
            lookup[normalized] = provider
    return lookup


_LOOKUP = _build_lookup()


def provider_named(name: str) -> Provider:
    """Resolve a canonical provider name or compatibility alias."""
    normalized = name.strip().lower().replace("_", "-")
    try:
        return _LOOKUP[normalized]
    except KeyError as error:
        raise KeyError(name) from error


def provider_names() -> tuple[str, ...]:
    return tuple(provider.slug for provider in PROVIDERS)


__all__ = ["PROVIDERS", "Provider", "provider_named", "provider_names"]
