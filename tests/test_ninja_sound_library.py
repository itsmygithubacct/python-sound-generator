from __future__ import annotations

import hashlib
import json
import math
import struct
import wave

import numpy as np
import pytest

from python_sound_generator._audio.assets import resolve_asset
from python_sound_generator.cli import main
from python_sound_generator.ninja import (
    CUES,
    GROUPS,
    Options,
    PACKS,
    SR,
    STYLES,
    generate,
    generate_with_sources,
    missing_sources,
    render,
    render_with_sources,
    source_inventory,
)


def _pcm_digest(samples: list[float]) -> str:
    frames = bytearray()
    for sample in samples:
        value = max(-1.0, min(1.0, float(sample)))
        frames.extend(struct.pack("<h", round(value * 32_767.0)))
    return hashlib.sha256(frames).hexdigest()


def test_ninja_catalog_is_grouped_and_packs_have_stable_coverage():
    assert GROUPS == (
        "movement",
        "blades",
        "projectiles",
        "tools",
        "stealth-world",
    )
    assert set(STYLES) == {"shadow", "arcade", "cinematic"}
    assert len(CUES) == 30
    assert {cue.group for cue in CUES.values()} == set(GROUPS)
    assert [entry.cue for entry in PACKS["core"].entries] == list(CUES)
    assert sum(entry.count for entry in PACKS["infiltration"].entries) == 33
    assert sum(entry.count for entry in PACKS["arsenal"].entries) == 36
    assert sum(entry.count for entry in PACKS["tools"].entries) == 18


def test_ninja_reuses_approved_sources_without_a_duplicate_wav_bank():
    inventory = source_inventory()
    assert len(inventory) == 35
    assert missing_sources() == ()
    assert all(resolve_asset(source.path).is_file() for source in inventory)
    assert all("/sources/" in source.logical_name for source in inventory)

    from python_sound_generator.generators.ninja_sound_generator import foley

    assert not list((foley.PROVIDER_ROOT / "ninja_sound_generator").rglob("*.wav"))


def test_programmatic_generation_is_deterministic_style_aware_and_source_aware():
    first = generate_with_sources("katana_clash", style="shadow", seed=73)
    second = generate_with_sources("katana_clash", style="shadow", seed=73)
    different_seed = generate_with_sources("katana_clash", style="shadow", seed=74)
    arcade = generate("katana_clash", style="arcade", seed=73)
    cinematic = generate("katana_clash", style="cinematic", seed=73)

    assert first == second
    assert first.samples != different_seed.samples
    assert first.sources
    assert all("/sources/" in source for source in first.sources)
    assert len({_pcm_digest(first.samples), _pcm_digest(arcade), _pcm_digest(cinematic)}) == 3
    assert len(first.samples) == round(CUES["katana_clash"].seconds * SR)
    assert max(abs(sample) for sample in first.samples) == pytest.approx(
        10 ** (CUES["katana_clash"].peak_db / 20)
    )


def test_ninja_render_contract_matches_other_source_aware_libraries():
    plain = render(
        "katana_clash", Options(), np.random.default_rng(23)
    )
    source_aware = render_with_sources(
        "katana_clash", Options(), np.random.default_rng(23)
    )
    assert isinstance(plain, list)
    assert plain == source_aware.samples
    assert source_aware.sources


def test_every_ninja_cue_has_exact_duration_finite_unique_pcm_and_honest_sources():
    digests = set()
    sourceful = set()
    for index, (name, cue) in enumerate(CUES.items(), 1):
        result = generate_with_sources(name, style="shadow", seed=100 + index)
        assert len(result.samples) == round(cue.seconds * SR), name
        assert all(math.isfinite(sample) for sample in result.samples), name
        assert max(abs(sample) for sample in result.samples) == pytest.approx(
            10 ** (cue.peak_db / 20)
        )
        if result.sources:
            sourceful.add(name)
        digests.add(_pcm_digest(result.samples))

    assert len(digests) == len(CUES)
    assert set(CUES) - sourceful == {
        "flash_powder",
        "mission_complete",
        "shuriken_throw",
    }


def test_foley_mix_can_be_removed_without_silencing_procedural_identity():
    hybrid = generate_with_sources("takedown", foley_mix=1.0, seed=19)
    procedural = generate_with_sources("takedown", foley_mix=0.0, seed=19)

    assert hybrid.sources
    assert procedural.sources == ()
    assert hybrid.samples != procedural.samples
    assert max(abs(sample) for sample in procedural.samples) > 0.1
    assert all(
        max(abs(sample) for sample in generate(name, foley_mix=0.0, seed=19))
        > 0.01
        for name in CUES
    )


def test_ninja_render_writes_variations_and_structured_source_manifest(tmp_path):
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    arguments = [
        "ninja",
        "render",
        "bamboo_break",
        "--style",
        "cinematic",
        "--intensity",
        "0.7",
        "--foley-mix",
        "0.8",
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
    assert first_manifest["generator"] == "ninja_sound_generator"
    assert first_manifest["schema_version"] == 1
    assert first_manifest["seed_scheme"] == "stable-v2"
    assert first_manifest["source_licensing"] == "CC0 1.0"
    assert first_manifest["asset_catalog"] == "assets/catalog.json"
    assert "no duplicated WAV bytes" in first_manifest["source_audio"]
    assert [item["sha256"] for item in artifacts] == [
        item["sha256"] for item in second_manifest["artifacts"]
    ]
    assert len({item["sha256"] for item in artifacts}) == 2

    for artifact in artifacts:
        path = first_out / artifact["file"]
        assert artifact["options"] == {
            "style": "cinematic",
            "intensity": 0.7,
            "foley_mix": 0.8,
        }
        assert artifact["sources"]
        assert artifact["loop"] is False
        with wave.open(str(path), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
                1,
                2,
                SR,
            )
            assert wav.getnframes() == round(CUES["bamboo_break"].seconds * SR)


def test_ninja_core_pack_is_organised_and_plays_in_manifest_order(
    tmp_path,
    monkeypatch,
):
    from python_sound_generator.generators.ninja_sound_generator import cli

    played = []
    monkeypatch.setattr(cli, "play_audio", lambda path: played.append(path) or True)

    assert main([
        "ninja",
        "pack",
        "core",
        "--style",
        "arcade",
        "--seed",
        "31",
        "--out",
        str(tmp_path),
        "--play",
    ]) == 0

    root = tmp_path / "core" / "arcade"
    manifest = json.loads((root / "manifest.json").read_text())
    artifacts = manifest["artifacts"]
    assert manifest["pack"] == "core"
    assert {artifact["cue"] for artifact in artifacts} == set(CUES)
    assert all("/" in artifact["file"] for artifact in artifacts)
    assert all((root / artifact["file"]).is_file() for artifact in artifacts)
    assert [path.resolve().relative_to(root.resolve()).as_posix() for path in played] == [
        artifact["file"] for artifact in artifacts
    ]


def test_ninja_sources_command_reports_complete_reuse_inventory(capsys):
    assert main(["ninja", "sources", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["missing"] == []
    assert payload["invalid"] == []
    assert len(payload["inventory"]) == 35
    assert {collection["license"] for collection in payload["collections"]} == {
        "CC0 1.0"
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["ninja", "render", "not_a_cue"],
        ["ninja", "render", "katana_clash", "--style", "not_a_style"],
        ["ninja", "render", "katana_clash", "--intensity", "0"],
        ["ninja", "render", "katana_clash", "--foley-mix", "-0.1"],
        ["ninja", "render", "katana_clash", "--count", "0"],
        ["ninja", "pack", "not_a_pack"],
    ],
)
def test_ninja_cli_rejects_invalid_requests(arguments, capsys):
    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "soundgen ninja:" in captured.err
