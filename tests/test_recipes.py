from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from python_sound_generator.recipes import (
    RecipeError,
    SCHEMA_VERSION,
    load_recipe,
    packaged_recipe_path,
    recipe_main,
    render_game,
)


def test_packaged_recipe_is_complete_and_procedural() -> None:
    book = load_recipe(packaged_recipe_path())
    assert [game.game_id for game in book.games] == [
        "cyber-runner", "fantasy-adventure", "fantasy-quest", "suburban-odyssey"
    ]
    assert book.sources == ()
    assert {game.provider for game in book.games} == {"cyberpunk", "quirky-rpg", "rpg"}
    assert all(game.cues for game in book.games)


def test_recipe_rejects_unsafe_output_and_incomplete_sample_provenance(tmp_path: Path) -> None:
    payload = json.loads(packaged_recipe_path().read_text(encoding="utf-8"))
    payload["games"]["fantasy-adventure"]["cues"][0]["output"] = "../escape"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecipeError, match="beneath the output"):
        load_recipe(bad)

    payload = json.loads(packaged_recipe_path().read_text(encoding="utf-8"))
    payload["sources"] = [{
        "id": "door", "title": "Door", "creator": "Recorder",
        "source_url": "https://example.invalid/door.wav", "license": "CC-BY-4.0",
        "license_evidence_url": "https://example.invalid/license", "accessed": "2026-07-21",
        "sha256": "0" * 64, "attribution": "",
    }]
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecipeError, match="CC0 or public domain"):
        load_recipe(bad)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attribution", "", "attribution must be a non-empty string"),
        ("accessed", "2026-02-30", "ISO 8601 calendar date"),
        ("source_url", "https://user:secret@example.invalid/source.wav", "without credentials"),
        ("license_evidence_url", "https://exa mple.invalid/license", "valid HTTP"),
    ],
)
def test_sample_provenance_fields_are_strict(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    payload = json.loads(packaged_recipe_path().read_text(encoding="utf-8"))
    source = {
        "id": "door",
        "title": "Door",
        "creator": "Recorder",
        "source_url": "https://example.invalid/door.wav",
        "license": "CC0-1.0",
        "license_evidence_url": "https://example.invalid/license",
        "accessed": "2026-07-21",
        "sha256": "0" * 64,
        "attribution": "Door by Recorder; released under CC0-1.0",
    }
    source[field] = value
    payload["sources"] = [source]
    recipe = tmp_path / "bad-provenance.json"
    recipe.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecipeError, match=message):
        load_recipe(recipe)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("style", "missing", "unknown cyberpunk style"),
        ("cue", "missing", "unknown cyberpunk cue"),
    ],
)
def test_recipe_validation_checks_provider_contract(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    payload = json.loads(packaged_recipe_path().read_text(encoding="utf-8"))
    if field == "style":
        payload["games"]["cyber-runner"][field] = value
    else:
        payload["games"]["cyber-runner"]["cues"][0][field] = value
    recipe = tmp_path / "bad-provider-contract.json"
    recipe.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecipeError, match=message):
        load_recipe(recipe)


def test_render_rejects_output_symlink_escape(tmp_path: Path) -> None:
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "cyber-runner").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RecipeError, match="symlink"):
        render_game(load_recipe(packaged_recipe_path()), "cyber-runner", output)


def test_render_rejects_preexisting_artifact_symlink(tmp_path: Path) -> None:
    destination = tmp_path / "output" / "cyber-runner" / "ui-cursor"
    destination.mkdir(parents=True)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"do not overwrite")
    (destination / "terminal_key_noir.wav").symlink_to(outside)
    with pytest.raises(RecipeError, match="pre-existing symlink"):
        render_game(
            load_recipe(packaged_recipe_path()),
            "cyber-runner",
            tmp_path / "output",
            ["ui.cursor"],
        )
    assert outside.read_bytes() == b"do not overwrite"


@pytest.mark.parametrize(
    ("game_id", "cue_id"),
    [
        ("cyber-runner", "ui.cursor"),
        ("fantasy-quest", "dialogue.open"),
        ("suburban-odyssey", "ui.cursor"),
        ("fantasy-adventure", "item.pickup"),
    ],
)
def test_each_recipe_provider_is_reproducible(
    tmp_path: Path, game_id: str, cue_id: str
) -> None:
    book = load_recipe(packaged_recipe_path())
    first = render_game(book, game_id, tmp_path / "first", [cue_id])
    second = render_game(book, game_id, tmp_path / "second", [cue_id])
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    assert first.read_bytes() == second.read_bytes()
    assert first_payload["recipe_schema"] == SCHEMA_VERSION
    assert first_payload["artifacts"] == second_payload["artifacts"]
    assert first_payload["sources"] == []
    row = first_payload["artifacts"][0]
    provider_manifest = first.parent / row["manifest"]
    assert hashlib.sha256(provider_manifest.read_bytes()).hexdigest() == row[
        "manifest_sha256"
    ]
    provider_payload = json.loads(provider_manifest.read_text(encoding="utf-8"))
    assert row["output_sha256"] == [
        artifact["sha256"] for artifact in provider_payload["artifacts"]
    ]
    for artifact in provider_payload["artifacts"]:
        output = provider_manifest.parent / artifact["file"]
        assert not Path(artifact["file"]).is_absolute()
        assert hashlib.sha256(output.read_bytes()).hexdigest() == artifact["sha256"]


def test_recipe_cli_validates_packaged_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert recipe_main(["validate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["games"] == [
        "cyber-runner",
        "fantasy-adventure",
        "fantasy-quest",
        "suburban-odyssey",
    ]
