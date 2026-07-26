from __future__ import annotations

import hashlib
import json
import math
import random
import wave
from pathlib import Path

import pytest

from python_sound_generator.cli import main
from python_sound_generator.generators.computer_sounds_generator import (
    CUES,
    ONESHOTS,
    Options,
    SR,
    render,
)
from python_sound_generator.generators.computer_sounds_generator import engine


SOURCE_HASHES = {
    "mouse_press_cc0.wav":
        "880000eab71e35a1c35f8d97547f8e1e5a5d4c0ddc1cedb89c356c27d2c0366e",
    "mouse_release_cc0.wav":
        "38e2e206329fea5a7fcde4bc9c89452e3ae205174e3ed26ae446f4cd0dc7ef88",
}


def test_mouse_click_is_a_two_contact_recorded_mechanism():
    result = render("mouse_click", Options(), random.Random(17))
    press_end = round(0.08 * SR)
    gap_start = round(0.085 * SR)
    gap_end = round(0.115 * SR)
    release_start = round(0.12 * SR)
    release_end = round(0.20 * SR)

    assert CUES["mouse_click"].group == "mechanism"
    assert "mouse_click" in ONESHOTS
    assert result.sources == (
        "mouse_press_cc0.wav",
        "mouse_release_cc0.wav",
    )
    assert len(result.samples) == round(
        (0.12 + 3_196 / SR + 0.06) * SR
    )
    assert max(abs(sample) for sample in result.samples) == pytest.approx(
        10 ** (-8.0 / 20.0)
    )
    assert max(abs(sample) for sample in result.samples[:press_end]) > 0.05
    assert max(abs(sample) for sample in result.samples[gap_start:gap_end]) == 0
    assert max(
        abs(sample)
        for sample in result.samples[release_start:release_end]
    ) > 0.05


def test_mouse_click_sources_are_canonical_and_match_provenance():
    provenance = json.loads(
        (engine.SOURCES / "provenance.json").read_text(encoding="utf-8")
    )
    collection = next(
        row for row in provenance["collections"]
        if row["name"] == "Middle Mouse Click"
    )

    assert collection["license"] == "CC0 1.0"
    assert collection["source_page"] == (
        "https://opengameart.org/content/middle-mouse-click"
    )
    assert {
        row["file"]: row["sha256"] for row in collection["files"]
    } == SOURCE_HASHES
    for name, expected_digest in SOURCE_HASHES.items():
        path = engine.SOURCES / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_digest
        with wave.open(str(path), "rb") as wav:
            assert (
                wav.getnchannels(),
                wav.getsampwidth(),
                wav.getframerate(),
            ) == (1, 2, SR)


def test_computer_cli_renders_mouse_click_with_source_receipt(tmp_path: Path):
    assert main([
        "computer",
        "render",
        "mouse_click",
        "--seed",
        "23",
        "--out",
        str(tmp_path),
    ]) == 0

    output = tmp_path / "mouse_click.wav"
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    artifact = manifest["artifacts"][0]
    assert output.is_file()
    assert artifact["cue"] == "mouse_click"
    assert artifact["sources"] == [
        "mouse_press_cc0.wav",
        "mouse_release_cc0.wav",
    ]
    assert math.isclose(artifact["duration_seconds"], 0.252472, abs_tol=1e-6)
