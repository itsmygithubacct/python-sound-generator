from __future__ import annotations

import ast
import hashlib
import json
import random
import struct
import subprocess
import sys
import wave
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


def _pcm_digest(samples) -> str:
    pcm = bytearray()
    for sample in samples:
        clipped = max(-1.0, min(1.0, float(sample)))
        pcm.extend(struct.pack("<h", round(clipped * 32767.0)))
    return hashlib.sha256(pcm).hexdigest()


def test_shared_dsp_moves_preserve_reference_pcm():
    from python_sound_generator._audio import dsp, loops
    from python_sound_generator._audio.foley_numpy import master

    procedural = dsp.master(
        dsp.mix(
            dsp.sine(220.0, 0.08),
            dsp.scale(dsp.noise(0.08, random.Random(17)), 0.03),
        ),
        -6.0,
    )
    loop = loops.master(
        loops.bandpass(
            dsp.noise(0.1, random.Random(23)), 90.0, 2_000.0, 2
        ),
        -9.0,
    )
    foley = master(np.linspace(-0.8, 0.9, 4_410), -4.0)

    assert _pcm_digest(procedural) == (
        "7126a0810d178183a7f5e38fcc96f415c377c301563302311a14cb8b80341e19"
    )
    assert _pcm_digest(loop) == (
        "c1737c252a1628a8800766780ecb1c15d54f0543e86f299a64dca1b6c4eb31d4"
    )
    assert _pcm_digest(foley) == (
        "58a504c29ee70df49e85ffdd99e4c4833405ebf81175e81ce1ca351ffd1929ea"
    )


def test_shared_chiptune_primitives_preserve_length_and_validate_controls():
    from python_sound_generator._audio.chiptune import (
        bit_reduce,
        echo,
        oscillator_track,
        sample_hold,
    )

    frequencies = [440.0] * 32
    waveforms = {
        tuple(oscillator_track(frequencies, waveform))
        for waveform in ("sine", "square", "pulse", "triangle", "saw")
    }
    source = [-1.2, -0.33, 0.12, 0.91, 1.2]

    assert len(waveforms) == 4  # square and pulse share the requested duty
    assert len(bit_reduce(source, 5)) == len(source)
    assert len(sample_hold(source, 3)) == len(source)
    assert len(echo(source, 0.001, 0.25, sample_rate=1_000)) == len(source)
    assert max(bit_reduce(source, 5)) <= 1.0

    with pytest.raises(ValueError):
        bit_reduce(source, 1)
    with pytest.raises(ValueError):
        sample_hold(source, 0)
    with pytest.raises(ValueError):
        echo(source, 0.0, 0.25)


def test_legacy_dsp_imports_are_compatibility_aliases():
    from python_sound_generator._audio import dsp, loops
    from python_sound_generator._audio import foley_numpy
    from python_sound_generator.generators import foley_core
    from python_sound_generator.generators.combat_generators.lib import (
        dsp as legacy_dsp,
    )
    from python_sound_generator.generators.environment_ambience_generator import (
        loopdsp,
    )

    assert legacy_dsp.samples is dsp.samples
    assert loopdsp.quantize is loops.quantize
    assert foley_core.load_wav is foley_numpy.load_wav


def test_computer_engine_has_a_small_stdlib_import_closure():
    script = """
import sys
from python_sound_generator.generators.computer_sounds_generator import engine
assert engine.SR == 44100
assert 'numpy' not in sys.modules
assert 'scipy' not in sys.modules
assert not any(
    name.startswith(
        'python_sound_generator.generators.aircraft_spacecraft_generator'
    )
    for name in sys.modules
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_extracted_rotor_preserves_aircraft_reference_pcm():
    from python_sound_generator._audio.rotor import RotorProfile, blade_pass

    profile = RotorProfile(
        name="turbofan",
        shaft_rate=66.0,
        blades=22,
        open_rotor=False,
        tone_gain=0.8,
        noise_gain=1.0,
        roughness=0.12,
        idle=0.22,
    )
    rendered = blade_pass(
        profile,
        random.Random(31),
        power=0.6,
        duration=0.5,
        intensity=0.85,
    )
    assert _pcm_digest(rendered) == (
        "787f53f5f4f0570cc96c368f62d793d438a85668a46430ca0c5c0fdfa205c681"
    )


def test_stable_seed_is_named_and_order_independent():
    from python_sound_generator._audio.runtime import derive_seed

    expected = derive_seed(
        17,
        provider="fire",
        cue="explosion",
        filename="large/explosion.wav",
        variation=2,
    )
    assert expected == derive_seed(
        17,
        filename="large/explosion.wav",
        variation=2,
        cue="explosion",
        provider="fire",
    )
    assert expected != derive_seed(
        17,
        provider="fire",
        cue="fireball",
        filename="large/explosion.wav",
        variation=2,
    )
    assert expected != derive_seed(
        17,
        provider="fire",
        cue="explosion",
        filename="large/explosion.wav",
        variation=3,
    )


def test_shared_player_falls_back_when_an_installed_backend_fails(
    tmp_path,
    monkeypatch,
):
    from python_sound_generator._audio import runtime

    target = tmp_path / "effect.wav"
    target.touch()
    calls = []
    available = {"paplay": "/bin/paplay", "play": "/bin/play"}
    monkeypatch.setattr(runtime.shutil, "which", available.get)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0 if command[0] == "/bin/play" else 1)

    monkeypatch.setattr(runtime.subprocess, "run", run)

    assert runtime.play(target) is True
    assert [command[0] for command, _ in calls] == ["/bin/paplay", "/bin/play"]
    assert all(kwargs["stdout"] is subprocess.DEVNULL for _, kwargs in calls)
    assert all(kwargs["stderr"] is subprocess.DEVNULL for _, kwargs in calls)


def test_manifest_writer_emits_a_versioned_structured_audio_contract(tmp_path):
    from python_sound_generator._audio.runtime import manifest_payload, write_manifest

    target = write_manifest(
        tmp_path / "manifest.json",
        generator="test_generator",
        command="render",
        artifacts=[{"file": "effect.wav", "cue": "effect"}],
        extra={"theme": "test"},
    )
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["audio"] == {
        "sample_rate": 44_100,
        "channels": 1,
        "sample_width": 2,
        "encoding": "signed PCM WAV",
    }
    assert payload["format"] == "mono 44100 Hz signed 16-bit PCM WAV"
    assert payload["seed_scheme"] == "stable-v2"
    assert payload["theme"] == "test"

    normalized = manifest_payload(
        generator="test_generator",
        command="render",
        artifacts=[{
            "file": "loop.wav",
            "cue": "loop",
            "loop": {"seamless": True, "step_ratio": 0.1},
        }],
    )["artifacts"][0]
    assert normalized["loop"] is True
    assert normalized["seam"] == {"seamless": True, "step_ratio": 0.1}
    assert normalized["sources"] == []


def test_shared_audio_buffer_writes_mono_and_stereo_pcm(tmp_path):
    from python_sound_generator._audio.pcm import AudioBuffer, write_pcm16

    mono = AudioBuffer(22_050, ((0.0, 0.5, -0.5),))
    stereo = AudioBuffer(
        44_100,
        ((0.0, 0.5, -0.5), (0.25, -0.25, 0.0)),
        ("source.wav",),
    )

    mono_path = write_pcm16(tmp_path / "mono.wav", mono)
    stereo_path = write_pcm16(tmp_path / "stereo.wav", stereo)
    assert mono_path.is_file()
    assert stereo_path.is_file()
    assert mono.channel_count == 1
    assert stereo.channel_count == 2
    assert stereo.sources == ("source.wav",)


def test_shared_pcm_conversion_can_preserve_or_mix_down_channels():
    from python_sound_generator._audio.foley_numpy import pcm_to_float

    pcm = np.array([[32767, -32768], [16384, 0]], dtype=np.int16)
    channels = pcm_to_float(pcm, mixdown=False)
    mono = pcm_to_float(pcm)

    assert channels.shape == (2, 2)
    assert mono.shape == (2,)
    assert np.allclose(mono, channels.mean(axis=1))


def test_cached_foley_wav_is_immutable_and_rejects_silent_material(tmp_path):
    from python_sound_generator._audio.foley_numpy import load_wav

    audible = tmp_path / "audible.wav"
    silent = tmp_path / "silent.wav"
    for path, frames in (
        (audible, np.array([0, 1200, -900, 600], dtype="<i2")),
        (silent, np.zeros(4, dtype="<i2")),
    ):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(44_100)
            handle.writeframes(frames.tobytes())

    load_wav.cache_clear()
    try:
        first = load_wav(audible, trim_threshold=0.0)
        second = load_wav(audible, trim_threshold=0.0)
        assert first is second
        assert first.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            first[0] = 0.5
        with pytest.raises(ValueError, match="no audible signal"):
            load_wav(silent)
    finally:
        load_wav.cache_clear()


def test_shared_foley_windows_cover_the_last_offset_and_validate_edges():
    from python_sound_generator._audio.foley_numpy import (
        equal_power_crossfade_loop,
        random_window,
        tile_to_frames,
        varispeed_configured,
    )

    class LastWindow:
        bounds = None

        def integers(self, low, high):
            self.bounds = (low, high)
            return high - 1

    rng = LastWindow()
    selected = random_window(
        np.arange(4, dtype=float), rng, 2.0, sample_rate=1
    )
    assert rng.bounds == (0, 3)
    assert selected.tolist() == [2.0, 3.0]
    assert len(equal_power_crossfade_loop(
        np.array([1.0, -1.0]), 2, 10.0, sample_rate=1
    )) == 2

    with pytest.raises(ValueError, match="empty signal"):
        tile_to_frames(np.array([]), 4)
    with pytest.raises(ValueError, match="finite"):
        varispeed_configured(np.ones(4), float("nan"))


def test_foley_source_paths_are_confined_to_provider_source_banks():
    from python_sound_generator._audio.foley_bank import FoleySource

    valid = FoleySource("rpg_sound_generator", "subfolder/contact.wav")
    assert valid.logical_name == (
        "rpg_sound_generator/sources/subfolder/contact.wav"
    )
    for provider, filename in (
        ("../rpg_sound_generator", "contact.wav"),
        ("rpg_sound_generator", "../contact.wav"),
        ("rpg_sound_generator", "/tmp/contact.wav"),
        ("rpg_sound_generator", "contact.mp3"),
    ):
        with pytest.raises(ValueError):
            FoleySource(provider, filename)


def test_foley_provenance_verifier_detects_tampered_excerpts(tmp_path):
    from python_sound_generator._audio.foley_cli import provenance_errors

    source = tmp_path / "contact.wav"
    source.write_bytes(b"approved excerpt")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = tmp_path / "provenance.json"
    payload = {
        "collections": [{
            "name": "test",
            "files": [{"file": source.name, "sha256": digest}],
        }],
    }
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    assert provenance_errors(provenance) == ()

    source.write_bytes(b"tampered excerpt")
    assert provenance_errors(provenance) == (
        "checksum mismatch: contact.wav",
    )


def test_new_game_library_controls_reject_non_finite_values():
    from python_sound_generator import cyberpunk, ninja, quirky_rpg, rpg

    cases = (
        (rpg.generate, "coin_pickup"),
        (cyberpunk.generate, "terminal_key"),
        (quirky_rpg.generate, "equip"),
        (ninja.generate, "katana_clash"),
    )
    for generate, cue in cases:
        with pytest.raises(ValueError, match="intensity"):
            generate(cue, intensity=float("nan"))
        with pytest.raises(ValueError, match="foley"):
            generate(cue, foley_mix=float("inf"))


def test_shared_asset_catalog_resolves_every_alias_without_duplicate_banks():
    from python_sound_generator._audio.assets import AssetCatalog
    from python_sound_generator.generators import PROVIDER_ROOT

    catalog = AssetCatalog.default()
    assert len(catalog.aliases) >= 100
    assert catalog.validation_errors() == ()
    logical_name, alias = next(iter(catalog.aliases.items()))
    assert catalog.resolve(PROVIDER_ROOT / logical_name) == alias.path.resolve()

    digests: dict[str, Path] = {}
    for path in PROVIDER_ROOT.rglob("*.wav"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest not in digests, (digests.get(digest), path)
        digests[digest] = path


def test_new_hybrid_foley_ledgers_match_all_provider_owned_excerpt_bytes():
    root = Path(__file__).parents[1] / "python_sound_generator" / "generators"
    checked = []
    for provider in ("rpg_sound_generator", "cyberpunk_sound_generator"):
        source_root = root / provider / "sources"
        ledger = json.loads((source_root / "provenance.json").read_text())
        for collection in ledger["collections"]:
            for record in collection.get("files", []):
                path = source_root / record["file"]
                assert path.is_file()
                assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
                checked.append(path)
    assert len(checked) == 15


def test_cue_honours_and_pack_filenames_use_shared_models():
    from python_sound_generator._audio.models import Honours, VariationFilenames

    class Entry:
        filename = "player/hurt"

    honours = Honours("style, seed, intensity")
    assert honours == ("style", "seed", "intensity")
    assert str(honours) == "style, seed, intensity"
    assert VariationFilenames.filename(Entry(), 0) == "player/hurt.wav"
    assert VariationFilenames.filename(Entry(), 1) == "player/hurt_v02.wav"


def test_every_manifest_cli_uses_shared_runtime_and_no_positional_seeds():
    root = Path(__file__).parents[1] / "python_sound_generator" / "generators"
    manifest_clis = []
    for path in root.glob("*/cli.py"):
        source = path.read_text(encoding="utf-8")
        if "manifest" in source and "artifacts" in source:
            manifest_clis.append(path)
            assert "write_shared_manifest" in source, path
        assert "1009" not in source, path
    assert len(manifest_clis) == 29


def test_generators_do_not_derive_seeds_with_direct_arithmetic():
    root = Path(__file__).parents[1] / "python_sound_generator" / "generators"
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp):
                continue
            names = {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            }
            attributes = {
                child.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
            }
            if any(name.lower().endswith("seed") for name in names | attributes):
                violations.append((path, node.lineno))
    assert violations == []


def test_borrowed_pack_entries_match_their_source_provider_seed_identity():
    from python_sound_generator.generators.projectile_sounds.cues import (
        PACKS as PROJECTILE_PACKS,
        weapons_render,
        weapons_seed,
    )
    from python_sound_generator.generators.unarmed_foley_generator.cues import (
        PACKS as UNARMED_PACKS,
        combat_render,
        combat_seed,
    )
    from python_sound_generator.generators.weapons_destruction_generator import (
        cues as weapons,
    )
    from python_sound_generator.generators.combat_generators.lib import (
        cues as combat,
    )

    base_seed = 29
    weapon_entry = PROJECTILE_PACKS["arena-fps"].borrowed_entries()[0]
    weapon_seed = weapons_seed(weapon_entry, 0, base_seed)
    assert weapons_render(weapon_entry, 0, base_seed) == weapons.render(
        weapon_entry.cue,
        weapon_entry.options,
        random.Random(weapon_seed),
    )

    combat_entry = UNARMED_PACKS["arcade-fighter"].borrowed_entries()[0]
    source_seed = combat_seed(combat_entry, 0, base_seed)
    assert combat_render(combat_entry, 0, base_seed) == combat.render(
        combat_entry.cue,
        combat_entry.opts,
        random.Random(source_seed),
    )


def test_power_tool_pack_uses_named_seed_fields(tmp_path):
    from python_sound_generator._audio.runtime import derive_seed
    from python_sound_generator.cli import main
    from python_sound_generator.generators.power_tool_generator import cli

    assert main([
        "power-tools",
        "pack",
        "showcase",
        "--seed",
        "37",
        "--out",
        str(tmp_path),
    ]) == 0
    manifest = json.loads(
        (tmp_path / "showcase" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifacts"][0]["seed"] == derive_seed(
        37,
        provider=cli.__package__,
        cue="drill_loaded",
        filename="drill_loaded",
    )
