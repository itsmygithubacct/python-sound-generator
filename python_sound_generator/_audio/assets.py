"""Content-addressed audio assets with provider-level logical aliases."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROVIDER_ROOT = PACKAGE_ROOT / "generators"
ASSET_ROOT = PACKAGE_ROOT / "assets"
CATALOG_PATH = ASSET_ROOT / "catalog.json"


@dataclass(frozen=True, slots=True)
class AssetAlias:
    sha256: str
    file: str
    bytes: int
    provenance: str | None = None

    @property
    def path(self) -> Path:
        return ASSET_ROOT / self.file


@dataclass(frozen=True)
class AssetCatalog:
    """Immutable mapping from old provider paths to shared physical assets."""

    aliases: Mapping[str, AssetAlias]
    schema_version: int = 1

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "AssetCatalog":
        if not CATALOG_PATH.is_file():
            return cls(MappingProxyType({}))
        payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        aliases = {
            key: AssetAlias(
                sha256=value["sha256"],
                file=value["file"],
                bytes=int(value["bytes"]),
                provenance=value.get("provenance"),
            )
            for key, value in payload.get("aliases", {}).items()
        }
        return cls(
            MappingProxyType(aliases),
            int(payload.get("schema_version", 0)),
        )

    @staticmethod
    def key_for(path: str | Path) -> str | None:
        candidate = Path(path).expanduser().resolve(strict=False)
        try:
            return candidate.relative_to(PROVIDER_ROOT.resolve()).as_posix()
        except ValueError:
            return None

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        key = self.key_for(candidate)
        alias = self.aliases.get(key or "")
        return alias.path.resolve() if alias is not None else candidate

    def logical_paths(
        self,
        directory: str | Path,
        pattern: str = "*.wav",
    ) -> tuple[Path, ...]:
        """List local and aliased logical files as if the bank were unchanged."""

        root = Path(directory).expanduser().resolve(strict=False)
        paths = {path.resolve() for path in root.glob(pattern) if path.is_file()}
        for key in self.aliases:
            logical = PROVIDER_ROOT / key
            if logical.parent.resolve(strict=False) == root and fnmatch.fnmatch(
                logical.name, pattern
            ):
                paths.add(logical)
        return tuple(sorted(paths, key=lambda path: path.name))

    def logical_names(
        self,
        directory: str | Path,
        pattern: str = "*.wav",
    ) -> tuple[str, ...]:
        return tuple(path.name for path in self.logical_paths(directory, pattern))

    def aliases_for_provider(self, package_name: str) -> tuple[tuple[Path, Path], ...]:
        prefix = f"{package_name}/"
        return tuple(
            (PROVIDER_ROOT / key, alias.path.resolve())
            for key, alias in sorted(self.aliases.items())
            if key.startswith(prefix)
        )

    def validation_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append(f"unsupported asset catalog schema {self.schema_version}")
        for key, alias in self.aliases.items():
            logical = Path(key)
            if logical.is_absolute() or ".." in logical.parts or len(logical.parts) < 2:
                errors.append(f"invalid asset alias {key!r}")
                continue
            if len(alias.sha256) != 64 or any(
                character not in "0123456789abcdef" for character in alias.sha256
            ):
                errors.append(f"invalid SHA-256 for {key}")
                continue
            if not alias.path.is_file():
                errors.append(f"missing shared asset for {key}: {alias.file}")
                continue
            if alias.path.stat().st_size != alias.bytes:
                errors.append(f"size mismatch for {key}: {alias.file}")
                continue
            digest = hashlib.sha256(alias.path.read_bytes()).hexdigest()
            if digest != alias.sha256:
                errors.append(f"checksum mismatch for {key}: {alias.file}")
        return tuple(errors)


def resolve_asset(path: str | Path) -> Path:
    """Resolve a provider-local path through the default shared catalog."""

    return AssetCatalog.default().resolve(path)


def asset_paths(
    directory: str | Path,
    pattern: str = "*.wav",
) -> tuple[Path, ...]:
    """Return logical paths for local and shared aliases in stable order."""

    return AssetCatalog.default().logical_paths(directory, pattern)


def asset_names(
    directory: str | Path,
    pattern: str = "*.wav",
) -> tuple[str, ...]:
    return AssetCatalog.default().logical_names(directory, pattern)


__all__ = [
    "ASSET_ROOT",
    "AssetAlias",
    "AssetCatalog",
    "asset_names",
    "asset_paths",
    "resolve_asset",
]
