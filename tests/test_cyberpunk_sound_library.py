from __future__ import annotations

import hashlib
import json
import math
import struct
import wave

import pytest

from python_sound_generator._audio.assets import resolve_asset
from python_sound_generator.cli import main
from python_sound_generator.cyberpunk import (
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


def test_cyberpunk_catalog_is_grouped_and_packs_have_stable_coverage():
    assert GROUPS == (
        "interface",
        "network",
        "security",
        "cyberware",
        "world",
        "combat",
        "stealth",
        "matrix",
        "magic",
    )
    assert set(STYLES) == {"electric", "noir", "corporate"}
    assert len(CUES) == 64
    assert {cue.group for cue in CUES.values()} == set(GROUPS)
    assert [entry.cue for entry in PACKS["core"].entries] == list(CUES)
    assert sum(entry.count for entry in PACKS["netrunner"].entries) == 30
    assert sum(entry.count for entry in PACKS["street-tech"].entries) == 28
    assert {name for name, cue in CUES.items() if cue.foley_kinds} == {
        "armor_hit",
        "dry_fire",
        "terminal_key",
        "implant_activate",
        "melee_hit",
        "melee_swing",
        "pistol_shot",
        "reload",
        "ricochet",
        "rifle_shot",
        "runner_downed",
        "shotgun_blast",
        "smg_burst",
    }


def test_cyberpunk_optional_hardware_inventory_resolves():
    inventory = source_inventory()
    assert len(inventory) == 42
    assert missing_sources() == ()
    assert all(resolve_asset(source.path).is_file() for source in inventory)


def test_cyberpunk_foley_is_deterministic_optional_and_physically_scoped():
    for cue in (
        "terminal_key",
        "implant_activate",
        "pistol_shot",
        "reload",
        "melee_swing",
        "melee_hit",
        "armor_hit",
        "ricochet",
        "runner_downed",
    ):
        first = generate_with_sources(cue, foley_mix=1.0, seed=19)
        second = generate_with_sources(cue, foley_mix=1.0, seed=19)
        procedural = generate_with_sources(cue, foley_mix=0.0, seed=19)
        assert first == second
        assert first.sources
        assert procedural.sources == ()
        assert first.samples != procedural.samples

    assert generate_with_sources(
        "hack_success", foley_mix=1.0, seed=19
    ).sources == ()


def test_programmatic_generation_is_deterministic_and_style_aware():
    first = generate("scan_ping", style="electric", seed=73)
    second = generate("scan_ping", style="electric", seed=73)
    different_seed = generate("scan_ping", style="electric", seed=74)
    noir = generate("scan_ping", style="noir", seed=73)
    corporate = generate("scan_ping", style="corporate", seed=73)

    assert first == second
    assert first != different_seed
    assert len({_pcm_digest(first), _pcm_digest(noir), _pcm_digest(corporate)}) == 3
    assert len(first) == round(CUES["scan_ping"].seconds * SR)
    assert max(abs(sample) for sample in first) == pytest.approx(10 ** (-4 / 20))


def test_every_cyberpunk_cue_has_exact_duration_finite_unique_pcm():
    digests = set()
    for index, (name, cue) in enumerate(CUES.items(), 1):
        samples = generate(name, style="electric", seed=100 + index)
        assert len(samples) == round(cue.seconds * SR), name
        assert all(math.isfinite(sample) for sample in samples), name
        assert max(abs(sample) for sample in samples) == pytest.approx(
            10 ** (cue.peak_db / 20)
        )
        digests.add(_pcm_digest(samples))
    assert len(digests) == len(CUES)


def test_cyberpunk_render_writes_variations_and_structured_manifest(tmp_path):
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    arguments = [
        "cyberpunk",
        "render",
        "implant_activate",
        "--style",
        "noir",
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
    assert first_manifest["generator"] == "cyberpunk_sound_generator"
    assert first_manifest["schema_version"] == 1
    assert first_manifest["seed_scheme"] == "stable-v2"
    assert "electronic identity layers" in first_manifest["synthesis"]
    assert "asset_catalog" not in first_manifest
    assert [item["sha256"] for item in artifacts] == [
        item["sha256"] for item in second_manifest["artifacts"]
    ]
    assert len({item["sha256"] for item in artifacts}) == 2

    for artifact in artifacts:
        path = first_out / artifact["file"]
        assert artifact["options"] == {
            "style": "noir",
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
            assert wav.getnframes() == round(
                CUES["implant_activate"].seconds * SR
            )


def test_cyberpunk_hybrid_manifest_names_the_servo_master(tmp_path):
    assert main([
        "cyberpunk",
        "render",
        "implant_activate",
        "--style",
        "corporate",
        "--foley-mix",
        "0.8",
        "--seed",
        "23",
        "--out",
        str(tmp_path),
    ]) == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    artifact = manifest["artifacts"][0]
    assert manifest["source_licensing"] == "CC0 1.0"
    assert manifest["asset_catalog"] == "assets/catalog.json"
    assert artifact["options"]["foley_mix"] == 0.8
    assert len(artifact["sources"]) == 1
    assert "servo_impulse" in artifact["sources"][0]


def test_cyberpunk_sources_command_reports_the_complete_inventory(capsys):
    assert main(["cyberpunk", "sources", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["missing"] == []
    assert payload["invalid"] == []
    assert len(payload["inventory"]) == 42
    assert {collection["license"] for collection in payload["collections"]} == {
        "CC0 1.0"
    }


def test_cyberpunk_core_pack_is_organised_and_plays_in_manifest_order(
    tmp_path,
    monkeypatch,
):
    from python_sound_generator.generators.cyberpunk_sound_generator import cli

    played = []
    monkeypatch.setattr(cli, "play_audio", lambda path: played.append(path) or True)

    assert main([
        "cyberpunk",
        "pack",
        "core",
        "--style",
        "corporate",
        "--seed",
        "31",
        "--out",
        str(tmp_path),
        "--play",
    ]) == 0

    root = tmp_path / "core" / "corporate"
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
        ["cyberpunk", "render", "not_a_cue"],
        ["cyberpunk", "render", "scan_ping", "--style", "not_a_style"],
        ["cyberpunk", "render", "scan_ping", "--intensity", "0"],
        ["cyberpunk", "render", "scan_ping", "--foley-mix", "1.1"],
        ["cyberpunk", "render", "scan_ping", "--count", "0"],
        ["cyberpunk", "pack", "not_a_pack"],
    ],
)
def test_cyberpunk_cli_rejects_invalid_requests(arguments, capsys):
    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "soundgen cyberpunk:" in captured.err
