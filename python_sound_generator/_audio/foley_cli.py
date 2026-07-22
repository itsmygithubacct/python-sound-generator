"""Shared provenance reporting for recording-backed command-line providers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .assets import resolve_asset
from .runtime import file_sha256


class LogicalSource(Protocol):
    """The inventory field needed by the source-report command."""

    @property
    def logical_name(self) -> str: ...


def _payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"provenance root is not an object: {path}")
    if not isinstance(value.get("collections"), list):
        raise ValueError(f"provenance collections are not an array: {path}")
    return value


def provenance_errors(
    path: Path,
    payload: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Verify every provider-owned file carrying a provenance checksum."""

    document = dict(payload) if payload is not None else _payload(path)
    errors: list[str] = []
    for collection_index, collection in enumerate(document.get("collections", [])):
        if not isinstance(collection, Mapping):
            errors.append(f"collection {collection_index} is not an object")
            continue
        if "files" not in collection:
            continue
        records = collection["files"]
        if records is None:
            continue
        if not isinstance(records, list):
            errors.append(f"collection {collection_index} files are not an array")
            continue
        for record_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                continue
            filename = record.get("file")
            expected = record.get("sha256")
            if filename is None and expected is None:
                continue
            label = f"collection {collection_index} file {record_index}"
            candidate = Path(filename) if isinstance(filename, str) else None
            if (
                candidate is None
                or candidate.is_absolute()
                or ".." in candidate.parts
            ):
                errors.append(f"{label} has an invalid relative path")
                continue
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
            ):
                errors.append(f"{label} has an invalid SHA-256")
                continue
            source = resolve_asset(path.parent / candidate)
            if not source.is_file():
                errors.append(f"missing checksummed source: {candidate.as_posix()}")
                continue
            if file_sha256(source) != expected:
                errors.append(f"checksum mismatch: {candidate.as_posix()}")
    return tuple(errors)


def report_sources(
    provenance: Path,
    *,
    source_inventory: Callable[[], Sequence[LogicalSource]],
    missing_sources: Callable[[], Sequence[str]],
    use_key: str,
    json_output: bool = False,
) -> int:
    """Print one consistent source inventory and return a diagnostic status."""

    payload = _payload(provenance)
    inventory = tuple(source_inventory())
    inventory_names = tuple(source.logical_name for source in inventory)
    if len(set(inventory_names)) != len(inventory_names):
        raise ValueError("Foley inventory contains duplicate logical paths")
    missing = tuple(missing_sources())
    invalid = provenance_errors(provenance, payload)

    if json_output:
        payload["inventory"] = list(inventory_names)
        payload["missing"] = list(missing)
        payload["invalid"] = list(invalid)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if not missing and not invalid else 2

    print(f"\nPolicy: {payload.get('source_audio_policy', 'unspecified')}")
    print(f"Storage: {payload.get('storage_policy', 'unspecified')}")
    for index, collection in enumerate(payload["collections"]):
        if not isinstance(collection, Mapping):
            raise ValueError(f"provenance collection {index} is not an object")
        try:
            name = collection["name"]
            license_name = collection["license"]
            author = collection["author"]
            use = collection[use_key]
        except KeyError as error:
            raise ValueError(
                f"provenance collection {index} has no {error.args[0]!r}"
            ) from None
        print(f"\n  {name} — {license_name}")
        print(f"    author: {author}")
        owner = collection.get("owner_provenance")
        if owner:
            print(f"    owner:  {owner}")
        print(f"    use:    {use}")
    print(f"\n  {len(inventory)} approved logical recordings")
    if missing:
        print("  missing:")
        for name in missing:
            print(f"    {name}")
    if invalid:
        print("  invalid provenance:")
        for error in invalid:
            print(f"    {error}")
    if missing or invalid:
        print()
        return 2
    print("  all sources resolved; checksummed excerpts verified\n")
    return 0


__all__ = ["LogicalSource", "provenance_errors", "report_sources"]
