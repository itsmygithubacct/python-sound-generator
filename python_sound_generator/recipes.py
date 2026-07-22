"""Validated, reproducible multi-cue recipes for game sound banks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from .providers import provider_named


SCHEMA_VERSION = "soundgen.recipe.v1"
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STYLE_AXES = {"rpg": "theme", "cyberpunk": "style", "quirky-rpg": "style"}
_SOURCE_LICENSES = frozenset({"CC0-1.0", "Public-Domain"})
_ACCESS_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MAX_RECIPE_BYTES = 1024 * 1024
_MAX_GAMES = 64
_MAX_CUES_PER_GAME = 256
_MAX_SOURCES = 1024


class RecipeError(ValueError):
    """A recipe is malformed, unsafe, or cannot be reproduced."""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    title: str
    creator: str
    source_url: str
    license: str
    license_evidence_url: str
    accessed: str
    sha256: str
    attribution: str


@dataclass(frozen=True, slots=True)
class CueRecipe:
    cue_id: str
    provider_cue: str
    seed: int
    count: int
    output: str


@dataclass(frozen=True, slots=True)
class GameRecipe:
    game_id: str
    provider: str
    style_axis: str
    style: str
    intensity: float
    cues: tuple[CueRecipe, ...]


@dataclass(frozen=True, slots=True)
class RecipeBook:
    games: tuple[GameRecipe, ...]
    sources: tuple[SourceRecord, ...]

    def require_game(self, game_id: str) -> GameRecipe:
        for game in self.games:
            if game.game_id == game_id:
                return game
        raise RecipeError(f"unknown recipe game: {game_id!r}")


def packaged_recipe_path() -> Path:
    return Path(__file__).with_name("assets") / "recipes" / "game-recipes.json"


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecipeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecipeError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise RecipeError(f"{label} has unknown field(s): {', '.join(unknown)}")
    if missing:
        raise RecipeError(f"{label} is missing field(s): {', '.join(missing)}")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise RecipeError(f"{label} must be a portable identifier")
    return value


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecipeError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise RecipeError(f"{label} must stay beneath the output directory")
    normalized = path.as_posix()
    if normalized in {".", ""}:
        raise RecipeError(f"{label} must name an output subdirectory")
    if any(not _ID.fullmatch(part) for part in path.parts):
        raise RecipeError(f"{label} must contain only portable path components")
    return normalized


def _http_url(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeError(f"{label} must be a non-empty HTTP(S) URL")
    if value != value.strip() or any(character.isspace() for character in value):
        raise RecipeError(f"{label} must be a valid HTTP(S) URL")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as error:
        raise RecipeError(f"{label} must be a valid HTTP(S) URL") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RecipeError(f"{label} must be a valid HTTP(S) URL without credentials")
    return value


def _access_date(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ACCESS_DATE.fullmatch(value):
        raise RecipeError(f"{label} must be an ISO 8601 calendar date (YYYY-MM-DD)")
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise RecipeError(
            f"{label} must be an ISO 8601 calendar date (YYYY-MM-DD)"
        ) from error
    return value


def _provider_contract(provider: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return cue and style names for the deliberately small recipe adapters."""

    if provider == "rpg":
        from .generators.rpg_sound_generator.cues import CUES
        from .generators.rpg_sound_generator.engine import THEMES

        return frozenset(CUES), frozenset(THEMES)
    if provider == "cyberpunk":
        from .generators.cyberpunk_sound_generator.cues import CUES
        from .generators.cyberpunk_sound_generator.engine import STYLES

        return frozenset(CUES), frozenset(STYLES)
    if provider == "quirky-rpg":
        from .generators.quirky_rpg_sound_generator.cues import CUES
        from .generators.quirky_rpg_sound_generator.engine import STYLES

        return frozenset(CUES), frozenset(STYLES)
    raise RecipeError(f"provider {provider!r} has no recipe adapter")


def _source(value: Any, index: int) -> SourceRecord:
    item = _mapping(value, f"sources[{index}]")
    expected = {
        "id", "title", "creator", "source_url", "license",
        "license_evidence_url", "accessed", "sha256", "attribution",
    }
    _exact_keys(item, expected, f"sources[{index}]")
    source_id = _identifier(item["id"], f"sources[{index}].id")
    license_name = item["license"]
    if license_name not in _SOURCE_LICENSES:
        raise RecipeError(f"{source_id}: sampled sources must be CC0 or public domain")
    for key in ("title", "creator"):
        if not isinstance(item[key], str) or not item[key].strip():
            raise RecipeError(f"{source_id}.{key} must be a non-empty string")
    source_url = _http_url(item["source_url"], f"{source_id}.source_url")
    evidence_url = _http_url(
        item["license_evidence_url"], f"{source_id}.license_evidence_url"
    )
    accessed = _access_date(item["accessed"], f"{source_id}.accessed")
    digest = item["sha256"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise RecipeError(f"{source_id}.sha256 must be lowercase SHA-256")
    attribution = item["attribution"]
    if not isinstance(attribution, str) or not attribution.strip():
        raise RecipeError(f"{source_id}.attribution must be a non-empty string")
    return SourceRecord(
        source_id, item["title"], item["creator"], source_url,
        license_name, evidence_url, accessed, digest,
        attribution,
    )


def _cue(value: Any, game_id: str, index: int) -> CueRecipe:
    item = _mapping(value, f"games.{game_id}.cues[{index}]")
    _exact_keys(item, {"id", "cue", "seed", "count", "output"},
                f"games.{game_id}.cues[{index}]")
    cue_id = _identifier(item["id"], f"games.{game_id}.cues[{index}].id")
    provider_cue = _identifier(item["cue"], f"games.{game_id}.{cue_id}.cue")
    seed = item["seed"]
    count = item["count"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise RecipeError(f"games.{game_id}.{cue_id}.seed must be a non-negative int64")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 32:
        raise RecipeError(f"games.{game_id}.{cue_id}.count must be in 1..32")
    return CueRecipe(
        cue_id, provider_cue, seed, count,
        _relative_path(item["output"], f"games.{game_id}.{cue_id}.output"),
    )


def _game(game_id: str, value: Any) -> GameRecipe:
    game_id = _identifier(game_id, "game id")
    item = _mapping(value, f"games.{game_id}")
    _exact_keys(item, {"provider", "style_axis", "style", "intensity", "cues"},
                f"games.{game_id}")
    provider = _identifier(item["provider"], f"games.{game_id}.provider")
    try:
        provider_named(provider)
    except KeyError as error:
        raise RecipeError(f"games.{game_id} names unknown provider {provider!r}") from error
    expected_axis = _STYLE_AXES.get(provider)
    if expected_axis is None:
        raise RecipeError(f"games.{game_id}: provider {provider!r} has no recipe adapter")
    style_axis = item["style_axis"]
    if style_axis != expected_axis:
        raise RecipeError(f"games.{game_id}.style_axis must be {expected_axis!r}")
    style = _identifier(item["style"], f"games.{game_id}.style")
    provider_cues, provider_styles = _provider_contract(provider)
    if style not in provider_styles:
        raise RecipeError(
            f"games.{game_id}.style names unknown {provider} style {style!r}"
        )
    intensity = item["intensity"]
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float)):
        raise RecipeError(f"games.{game_id}.intensity must be numeric")
    intensity = float(intensity)
    if not 0.05 <= intensity <= 1.0:
        raise RecipeError(f"games.{game_id}.intensity must be in 0.05..1")
    raw_cues = item["cues"]
    if not isinstance(raw_cues, list) or not raw_cues:
        raise RecipeError(f"games.{game_id}.cues must be a non-empty array")
    if len(raw_cues) > _MAX_CUES_PER_GAME:
        raise RecipeError(
            f"games.{game_id}.cues exceeds the {_MAX_CUES_PER_GAME}-cue limit"
        )
    cues = tuple(_cue(raw, game_id, index) for index, raw in enumerate(raw_cues))
    cue_ids = [cue.cue_id for cue in cues]
    outputs = [cue.output for cue in cues]
    if len(set(cue_ids)) != len(cue_ids):
        raise RecipeError(f"games.{game_id} has duplicate cue IDs")
    if len(set(outputs)) != len(outputs):
        raise RecipeError(f"games.{game_id} has duplicate output paths")
    unknown_cues = sorted({cue.provider_cue for cue in cues} - provider_cues)
    if unknown_cues:
        raise RecipeError(
            f"games.{game_id} names unknown {provider} cue(s): "
            f"{', '.join(unknown_cues)}"
        )
    return GameRecipe(game_id, provider, style_axis, style, intensity, cues)


def load_recipe(path: str | Path) -> RecipeBook:
    source = Path(path)
    try:
        if source.stat().st_size > _MAX_RECIPE_BYTES:
            raise RecipeError(
                f"recipe exceeds the {_MAX_RECIPE_BYTES}-byte safety limit"
            )
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RecipeError(f"non-finite JSON value: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise RecipeError(f"could not load recipe: {error}") from error
    root = _mapping(payload, "recipe")
    _exact_keys(root, {"schema_version", "games", "sources"}, "recipe")
    if root["schema_version"] != SCHEMA_VERSION:
        raise RecipeError(f"unsupported recipe schema: {root['schema_version']!r}")
    raw_games = _mapping(root["games"], "games")
    if not raw_games:
        raise RecipeError("games must not be empty")
    if len(raw_games) > _MAX_GAMES:
        raise RecipeError(f"games exceeds the {_MAX_GAMES}-game limit")
    games = tuple(_game(game_id, raw_games[game_id]) for game_id in sorted(raw_games))
    raw_sources = root["sources"]
    if not isinstance(raw_sources, list):
        raise RecipeError("sources must be an array")
    if len(raw_sources) > _MAX_SOURCES:
        raise RecipeError(f"sources exceeds the {_MAX_SOURCES}-source limit")
    sources = tuple(_source(item, index) for index, item in enumerate(raw_sources))
    source_ids = [item.source_id for item in sources]
    if len(set(source_ids)) != len(source_ids):
        raise RecipeError("sources has duplicate IDs")
    return RecipeBook(games, sources)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained_path(root: Path, relative: str, label: str) -> Path:
    target = root.joinpath(*PurePosixPath(relative).parts)
    try:
        target.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise RecipeError(f"{label} escapes the output directory through a symlink") from error
    return target


def _provider_artifacts(manifest: Path, cue: CueRecipe) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(
            manifest.read_text(encoding="utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RecipeError(f"non-finite provider manifest value: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise RecipeError(f"could not load provider manifest: {error}") from error
    root = _mapping(payload, f"{cue.cue_id} provider manifest")
    rendered = root.get("artifacts")
    if not isinstance(rendered, list) or len(rendered) != cue.count:
        raise RecipeError(f"{cue.cue_id} provider manifest has wrong artifact count")

    destination = manifest.parent
    variations: set[int] = set()
    for index, raw in enumerate(rendered):
        artifact = _mapping(raw, f"{cue.cue_id} artifacts[{index}]")
        filename = _relative_path(
            artifact.get("file"), f"{cue.cue_id} artifacts[{index}].file"
        )
        output = _contained_path(
            destination, filename, f"{cue.cue_id} artifacts[{index}].file"
        )
        if not output.is_file():
            raise RecipeError(f"{cue.cue_id} provider manifest names a missing artifact")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise RecipeError(f"{cue.cue_id} artifact has invalid SHA-256")
        if _sha256(output) != digest:
            raise RecipeError(f"{cue.cue_id} artifact checksum does not match")
        if artifact.get("cue") != cue.provider_cue:
            raise RecipeError(f"{cue.cue_id} provider manifest names the wrong cue")
        variation = artifact.get("variation")
        if (
            isinstance(variation, bool)
            or not isinstance(variation, int)
            or not 1 <= variation <= cue.count
            or variation in variations
        ):
            raise RecipeError(f"{cue.cue_id} provider manifest has invalid variations")
        variations.add(variation)
        seed = artifact.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise RecipeError(f"{cue.cue_id} artifact has invalid derived seed")
        for metric in ("duration_seconds", "peak_dbfs", "rms_dbfs"):
            value = artifact.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RecipeError(f"{cue.cue_id} artifact has invalid {metric}")
        if float(artifact["duration_seconds"]) <= 0.0:
            raise RecipeError(f"{cue.cue_id} artifact has non-positive duration")
    return rendered


def _reject_output_symlinks(directory: Path, label: str) -> None:
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise RecipeError(f"{label} contains a pre-existing symlink")


def render_game(
    book: RecipeBook,
    game_id: str,
    output_root: str | Path,
    cue_ids: Iterable[str] | None = None,
) -> Path:
    game = book.require_game(game_id)
    selected = None if cue_ids is None else frozenset(cue_ids)
    if selected is not None:
        if not selected:
            raise RecipeError("cue selection must not be empty")
        known = {cue.cue_id for cue in game.cues}
        unknown = sorted(selected - known)
        if unknown:
            raise RecipeError(f"unknown {game_id} cue(s): {', '.join(unknown)}")
    output_base = Path(output_root)
    output_base.mkdir(parents=True, exist_ok=True)
    root = _contained_path(output_base, game.game_id, "game output")
    root.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    provider = provider_named(game.provider)
    for cue in game.cues:
        if selected is not None and cue.cue_id not in selected:
            continue
        destination = _contained_path(root, cue.output, f"{cue.cue_id} output")
        destination.mkdir(parents=True, exist_ok=True)
        _reject_output_symlinks(destination, f"{cue.cue_id} output")
        manifest = _contained_path(
            destination, "manifest.json", f"{cue.cue_id} provider manifest"
        )
        argv = [
            "render", cue.provider_cue,
            f"--{game.style_axis}", game.style,
            "--intensity", format(game.intensity, ".8g"),
            "--seed", str(cue.seed),
            "--count", str(cue.count),
            "--out", str(destination),
        ]
        result = provider.invoke(argv)
        if result != 0:
            raise RecipeError(f"{game.game_id}.{cue.cue_id} render failed with {result}")
        if not manifest.is_file():
            raise RecipeError(f"{game.game_id}.{cue.cue_id} produced no manifest")
        rendered = _provider_artifacts(manifest, cue)
        artifacts.append({
            "cue_id": cue.cue_id,
            "provider_cue": cue.provider_cue,
            "seed": cue.seed,
            "count": cue.count,
            "manifest": manifest.relative_to(root).as_posix(),
            "manifest_sha256": _sha256(manifest),
            "output_sha256": [artifact["sha256"] for artifact in rendered],
        })
    aggregate = {
        "artifacts": artifacts,
        "game": game.game_id,
        "provider": game.provider,
        "recipe_schema": SCHEMA_VERSION,
        "sources": [item.source_id for item in book.sources],
        "style": game.style,
        "style_axis": game.style_axis,
    }
    target = _contained_path(
        root, "soundgen-recipe-manifest.json", "aggregate manifest"
    )
    target.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def recipe_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="soundgen recipe")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("recipe", type=Path, nargs="?", default=packaged_recipe_path())
    validate.add_argument("--json", action="store_true")
    render = subparsers.add_parser("render")
    render.add_argument("game")
    render.add_argument("--recipe", type=Path, default=packaged_recipe_path())
    render.add_argument("--cue", action="append", default=[])
    render.add_argument("--out", type=Path, required=True)
    try:
        args = parser.parse_args(list(argv))
        book = load_recipe(args.recipe)
        if args.command == "validate":
            payload = {
                "games": [game.game_id for game in book.games],
                "schema_version": SCHEMA_VERSION,
                "source_count": len(book.sources),
                "status": "ok",
            }
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(
                    f"valid {SCHEMA_VERSION}: {len(book.games)} games, "
                    f"{len(book.sources)} sampled sources"
                )
            return 0
        manifest = render_game(book, args.game, args.out, args.cue or None)
        print(manifest)
        return 0
    except (OSError, RecipeError) as error:
        print(f"soundgen recipe: {error}", file=sys.stderr)
        return 2


__all__ = [
    "CueRecipe", "GameRecipe", "RecipeBook", "RecipeError", "SCHEMA_VERSION",
    "SourceRecord", "load_recipe", "packaged_recipe_path", "recipe_main",
    "render_game",
]
