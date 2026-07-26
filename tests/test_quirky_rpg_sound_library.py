from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
import wave

import pytest

from python_sound_generator._audio.assets import resolve_asset
from python_sound_generator.cli import main
from python_sound_generator.quirky_rpg import (
    CUES,
    GROUPS,
    PACKS,
    SR,
    STYLES,
    generate,
    generate_with_sources,
    missing_sources,
    source_inventory,
)


def _pcm_digest(samples: list[float]) -> str:
    frames = bytearray()
    for sample in samples:
        value = max(-1.0, min(1.0, float(sample)))
        frames.extend(struct.pack("<h", round(value * 32_767.0)))
    return hashlib.sha256(frames).hexdigest()


def test_quirky_rpg_catalog_is_grouped_and_packs_have_stable_coverage():
    assert GROUPS == (
        "interface",
        "dialogue",
        "items",
        "psychic",
        "battle",
        "world",
        "field",
        "roles",
        "services",
        "conditions",
    )
    assert set(STYLES) == {"suburban", "psychedelic", "lofi"}
    assert len(CUES) == 65
    assert {cue.group for cue in CUES.values()} == set(GROUPS)
    assert [entry.cue for entry in PACKS["core"].entries] == list(CUES)
    assert sum(entry.count for entry in PACKS["hometown"].entries) == 48
    assert sum(entry.count for entry in PACKS["psychic"].entries) == 15
    assert sum(entry.count for entry in PACKS["battle"].entries) == 15
    assert {name for name, cue in CUES.items() if cue.foley_kinds} == {
        "equip",
        "player_hit",
        "enemy_hit",
        "critical_hit",
        "door_transition",
        "attack_swing",
        "attack_miss",
        "enemy_attack",
        "enemy_defeat",
        "party_down",
    }


def test_quirky_rpg_reuses_a_complete_foley_inventory_without_new_wavs():
    inventory = source_inventory()
    assert len(inventory) == 24
    assert missing_sources() == ()
    assert all(resolve_asset(source.path).is_file() for source in inventory)

    from python_sound_generator.generators.quirky_rpg_sound_generator import foley

    assert not list((foley.PROVIDER_ROOT / "quirky_rpg_sound_generator").rglob("*.wav"))


def test_quirky_rpg_foley_is_optional_source_aware_and_cartridge_treated():
    sourceful = {name for name, cue in CUES.items() if cue.foley_kinds}
    for index, name in enumerate(CUES, 1):
        hybrid = generate_with_sources(name, foley_mix=1.0, seed=200 + index)
        procedural = generate_with_sources(name, foley_mix=0.0, seed=200 + index)
        assert procedural.sources == ()
        if name in sourceful:
            assert hybrid.sources
            assert hybrid.samples != procedural.samples
            assert all(
                "/sources/" in source or source.startswith("generator:")
                for source in hybrid.sources
            )
        else:
            assert hybrid.sources == ()


def test_programmatic_generation_is_deterministic_and_style_aware():
    first = generate("strange_event", style="suburban", seed=73)
    second = generate("strange_event", style="suburban", seed=73)
    different_seed = generate("strange_event", style="suburban", seed=74)
    psychedelic = generate("strange_event", style="psychedelic", seed=73)
    lofi = generate("strange_event", style="lofi", seed=73)

    assert first == second
    assert first != different_seed
    assert len({_pcm_digest(first), _pcm_digest(psychedelic), _pcm_digest(lofi)}) == 3
    assert len(first) == round(CUES["strange_event"].seconds * SR)
    assert max(abs(sample) for sample in first) == pytest.approx(10 ** (-4 / 20))


def test_every_quirky_rpg_cue_has_exact_duration_finite_unique_pcm():
    digests = set()
    for index, (name, cue) in enumerate(CUES.items(), 1):
        samples = generate(name, style="suburban", seed=100 + index)
        assert len(samples) == round(cue.seconds * SR), name
        assert all(math.isfinite(sample) for sample in samples), name
        assert max(abs(sample) for sample in samples) == pytest.approx(
            10 ** (cue.peak_db / 20)
        )
        digests.add(_pcm_digest(samples))
    assert len(digests) == len(CUES)


def test_quirky_rpg_public_import_stays_standard_library_only():
    script = """
import sys
from python_sound_generator.quirky_rpg import CUES, generate
assert len(CUES) == 65
assert generate('cursor_move', seed=3)
assert 'numpy' not in sys.modules
assert 'scipy' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_quirky_rpg_render_writes_variations_and_structured_manifest(tmp_path):
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    arguments = [
        "quirky-rpg",
        "render",
        "psi_cast",
        "--style",
        "psychedelic",
        "--intensity",
        "0.7",
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
    assert first_manifest["generator"] == "quirky_rpg_sound_generator"
    assert first_manifest["schema_version"] == 1
    assert first_manifest["seed_scheme"] == "stable-v2"
    assert "physical blunt-weapon movement" in first_manifest["synthesis"]
    assert "asset_catalog" not in first_manifest
    assert [item["sha256"] for item in artifacts] == [
        item["sha256"] for item in second_manifest["artifacts"]
    ]
    assert len({item["sha256"] for item in artifacts}) == 2

    for artifact in artifacts:
        path = first_out / artifact["file"]
        assert artifact["options"] == {
            "style": "psychedelic",
            "intensity": 0.7,
            "foley_mix": 0.0,
        }
        assert artifact["sources"] == []
        assert artifact["loop"] is False
        with wave.open(str(path), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
                1,
                2,
                SR,
            )
            assert wav.getnframes() == round(CUES["psi_cast"].seconds * SR)


def test_quirky_rpg_hybrid_manifest_reports_reused_panel_and_latch(tmp_path):
    assert main([
        "quirky-rpg",
        "render",
        "door_transition",
        "--style",
        "lofi",
        "--foley-mix",
        "0.75",
        "--seed",
        "23",
        "--out",
        str(tmp_path),
    ]) == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    artifact = manifest["artifacts"][0]
    assert manifest["source_licensing"] == "CC0 1.0"
    assert manifest["asset_catalog"] == "assets/catalog.json"
    assert artifact["options"]["foley_mix"] == 0.75
    assert len(artifact["sources"]) == 2
    assert any("scrape" in source or "scraping" in source
               for source in artifact["sources"])
    assert any("lock" in source for source in artifact["sources"])


def test_quirky_rpg_sources_command_reports_reuse_without_missing_files(capsys):
    assert main(["quirky-rpg", "sources", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["missing"] == []
    assert payload["invalid"] == []
    assert len(payload["inventory"]) == 24
    assert {collection["license"] for collection in payload["collections"]} == {
        "CC0 1.0"
    }


def test_quirky_rpg_core_pack_is_organised_and_plays_in_manifest_order(
    tmp_path,
    monkeypatch,
):
    from python_sound_generator.generators.quirky_rpg_sound_generator import cli

    played = []
    monkeypatch.setattr(cli, "play_audio", lambda path: played.append(path) or True)

    assert main([
        "quirky-rpg",
        "pack",
        "core",
        "--style",
        "lofi",
        "--seed",
        "31",
        "--out",
        str(tmp_path),
        "--play",
    ]) == 0

    root = tmp_path / "core" / "lofi"
    manifest = json.loads((root / "manifest.json").read_text())
    artifacts = manifest["artifacts"]
    assert manifest["pack"] == "core"
    assert {artifact["cue"] for artifact in artifacts} == set(CUES)
    assert all("/" in artifact["file"] for artifact in artifacts)
    assert all((root / artifact["file"]).is_file() for artifact in artifacts)
    assert [path.resolve().relative_to(root.resolve()).as_posix() for path in played] == [
        artifact["file"] for artifact in artifacts
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["quirky-rpg", "render", "not_a_cue"],
        ["quirky-rpg", "render", "psi_cast", "--style", "not_a_style"],
        ["quirky-rpg", "render", "psi_cast", "--intensity", "0"],
        ["quirky-rpg", "render", "psi_cast", "--foley-mix", "-0.1"],
        ["quirky-rpg", "render", "psi_cast", "--count", "0"],
        ["quirky-rpg", "pack", "not_a_pack"],
    ],
)
def test_quirky_rpg_cli_rejects_invalid_requests(arguments, capsys):
    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "soundgen quirky-rpg:" in captured.err
