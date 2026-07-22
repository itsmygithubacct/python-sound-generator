from __future__ import annotations

import json
import wave

import pytest

from python_sound_generator._audio.assets import resolve_asset
from python_sound_generator.cli import main
from python_sound_generator.rpg import (
    CUES,
    GROUPS,
    PACKS,
    SR,
    THEMES,
    generate,
    generate_with_sources,
    missing_sources,
    source_inventory,
)


def test_rpg_catalog_is_grouped_and_core_pack_covers_every_cue():
    assert GROUPS == ("inventory", "items", "progression", "world", "system")
    assert set(THEMES) == {"fantasy", "dark", "retro"}
    assert len(CUES) == 19
    assert {cue.group for cue in CUES.values()} == set(GROUPS)
    assert [entry.cue for entry in PACKS["core"].entries] == list(CUES)
    assert {name for name, cue in CUES.items() if cue.foley_kinds} == {
        "item_drop",
        "item_equip",
        "coin_pickup",
        "purchase",
        "potion_drink",
        "chest_open",
        "door_unlock",
    }


def test_rpg_optional_foley_inventory_is_complete_without_duplicate_masters():
    inventory = source_inventory()
    assert len(inventory) == 25
    assert missing_sources() == ()
    assert all(resolve_asset(source.path).is_file() for source in inventory)
    assert all("/sources/" in source.logical_name for source in inventory)


def test_rpg_foley_is_deterministic_optional_and_honestly_reported():
    sourceful = {name for name, cue in CUES.items() if cue.foley_kinds}
    for index, name in enumerate(CUES, 1):
        first = generate_with_sources(name, foley_mix=1.0, seed=100 + index)
        second = generate_with_sources(name, foley_mix=1.0, seed=100 + index)
        procedural = generate_with_sources(name, foley_mix=0.0, seed=100 + index)
        assert first == second
        assert procedural.sources == ()
        if name in sourceful:
            assert first.sources
            assert first.samples != procedural.samples
            assert all("/sources/" in source for source in first.sources)
        else:
            assert first.sources == ()


def test_programmatic_generation_is_deterministic_and_theme_aware():
    first = generate("coin_pickup", theme="fantasy", seed=73)
    second = generate("coin_pickup", theme="fantasy", seed=73)
    different_seed = generate("coin_pickup", theme="fantasy", seed=74)
    retro = generate("coin_pickup", theme="retro", seed=73)

    assert first == second
    assert first != different_seed
    assert first != retro
    assert len(first) == round(CUES["coin_pickup"].seconds * SR)
    assert max(abs(sample) for sample in first) == pytest.approx(10 ** (-4 / 20))


def test_rpg_render_writes_variations_and_a_reproducible_manifest(tmp_path):
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    arguments = [
        "rpg",
        "render",
        "coin_pickup",
        "--theme",
        "dark",
        "--count",
        "2",
        "--seed",
        "17",
    ]

    assert main([*arguments, "--out", str(first_out)]) == 0
    assert main([*arguments, "--out", str(second_out)]) == 0

    first_manifest = json.loads((first_out / "manifest.json").read_text())
    second_manifest = json.loads((second_out / "manifest.json").read_text())
    artifacts = first_manifest["artifacts"]
    assert first_manifest["generator"] == "rpg_sound_generator"
    assert [item["sha256"] for item in artifacts] == [
        item["sha256"] for item in second_manifest["artifacts"]
    ]
    assert len({item["sha256"] for item in artifacts}) == 2

    for artifact in artifacts:
        path = first_out / artifact["file"]
        assert artifact["options"] == {
            "theme": "dark",
            "intensity": 0.85,
            "foley_mix": 0.0,
        }
        assert artifact["sources"] == []
        with wave.open(str(path), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
                1,
                2,
                SR,
            )
            assert wav.getnframes() == round(CUES["coin_pickup"].seconds * SR)


def test_rpg_hybrid_render_records_sources_and_provenance(tmp_path):
    assert main([
        "rpg",
        "render",
        "potion_drink",
        "--theme",
        "fantasy",
        "--foley-mix",
        "0.7",
        "--seed",
        "23",
        "--out",
        str(tmp_path),
    ]) == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    artifact = manifest["artifacts"][0]
    assert manifest["source_licensing"] == "CC0 1.0"
    assert manifest["asset_catalog"] == "assets/catalog.json"
    assert artifact["options"]["foley_mix"] == 0.7
    assert len(artifact["sources"]) == 2
    assert any("cork_pop" in source for source in artifact["sources"])
    assert any("gulp" in source for source in artifact["sources"])


def test_rpg_sources_command_reports_the_complete_inventory(capsys):
    assert main(["rpg", "sources", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["missing"] == []
    assert payload["invalid"] == []
    assert len(payload["inventory"]) == 25
    assert {collection["license"] for collection in payload["collections"]} == {
        "CC0 1.0"
    }


def test_rpg_core_pack_renders_an_organised_drop_in_library(tmp_path):
    assert main([
        "rpg",
        "pack",
        "core",
        "--theme",
        "retro",
        "--seed",
        "31",
        "--out",
        str(tmp_path),
    ]) == 0

    root = tmp_path / "core" / "retro"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["pack"] == "core"
    assert {artifact["cue"] for artifact in manifest["artifacts"]} == set(CUES)
    assert all((root / artifact["file"]).is_file()
               for artifact in manifest["artifacts"])
    assert all("/" in artifact["file"] for artifact in manifest["artifacts"])


def test_rpg_core_pack_plays_every_cue_in_manifest_order(tmp_path, monkeypatch):
    from python_sound_generator.generators.rpg_sound_generator import cli

    played = []
    monkeypatch.setattr(cli, "play_audio", lambda path: played.append(path) or True)

    assert main([
        "rpg",
        "pack",
        "core",
        "--theme",
        "fantasy",
        "--seed",
        "41",
        "--out",
        str(tmp_path),
        "--play",
    ]) == 0

    root = tmp_path / "core" / "fantasy"
    manifest = json.loads((root / "manifest.json").read_text())
    assert [path.resolve().relative_to(root.resolve()).as_posix() for path in played] == [
        artifact["file"] for artifact in manifest["artifacts"]
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["rpg", "render", "not_a_cue"],
        ["rpg", "render", "coin_pickup", "--theme", "not_a_theme"],
        ["rpg", "render", "coin_pickup", "--intensity", "0"],
        ["rpg", "render", "coin_pickup", "--foley-mix", "-0.1"],
        ["rpg", "render", "coin_pickup", "--count", "0"],
        ["rpg", "pack", "not_a_pack"],
    ],
)
def test_rpg_cli_rejects_invalid_requests(arguments, capsys):
    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "soundgen rpg:" in captured.err
