from __future__ import annotations

import json
import subprocess
import sys
import wave

import pytest

from python_sound_generator import get_provider, providers, run
from python_sound_generator.cli import _doctor_row, main
from python_sound_generator.generators import PROVIDER_ROOT
from python_sound_generator.providers import PROVIDERS, Provider, provider_named


def test_registry_has_32_unique_providers():
    assert len(PROVIDERS) == 32
    names = [provider.slug for provider in PROVIDERS]
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_every_provider_is_packaged_and_has_an_entrypoint():
    for provider in PROVIDERS:
        assert provider.directory.is_dir(), provider.slug
        assert (provider.directory / "__init__.py").is_file(), provider.slug
        assert provider.validation_errors() == (), provider.slug
        assert callable(provider.load()), provider.slug

    assert not any(provider.package_name in sys.modules for provider in PROVIDERS)
    assert "foley_core" not in sys.modules
    assert "lib" not in sys.modules


def test_registry_loading_is_lazy():
    script = """
import contextlib
import io
import sys
from python_sound_generator.cli import main
from python_sound_generator.providers import PROVIDERS
assert PROVIDERS
with contextlib.redirect_stdout(io.StringIO()):
    assert main(["list"]) == 0
prefix = "python_sound_generator.generators."
assert not any(
    prefix + provider.package_name in sys.modules for provider in PROVIDERS
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_programmatic_api_exposes_and_runs_providers(capsys):
    assert providers() is PROVIDERS
    assert get_provider("household-appliance").slug == "appliances"
    assert run("fire", ["list"]) == 0
    assert capsys.readouterr().out.strip()


def test_aliases_accept_module_and_hyphen_names():
    assert provider_named("fire_sound_generator").slug == "fire"
    assert provider_named("household-appliance").slug == "appliances"
    assert provider_named("projectile_sounds").slug == "projectiles"
    assert provider_named("role-playing-game").slug == "rpg"
    assert provider_named("rpg_sound_generator").slug == "rpg"
    assert provider_named("cyberpunk-game").slug == "cyberpunk"
    assert provider_named("cyberpunk_sound_generator").slug == "cyberpunk"
    assert provider_named("16-bit-rpg").slug == "quirky-rpg"
    assert provider_named("quirky_rpg_sound_generator").slug == "quirky-rpg"
    assert provider_named("ninja-game").slug == "ninja"
    assert provider_named("ninja_sound_generator").slug == "ninja"


def test_list_json_is_machine_readable(capsys):
    assert main(["list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in payload] == [p.slug for p in PROVIDERS]
    assert all("description" in row for row in payload)


def test_info_reports_packaged_source_provenance(capsys):
    assert main(["info", "appliances", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["module"] == "household_appliance_generator"
    assert payload["source_files"] > 0
    assert payload["audio_files"] > 0
    assert payload["asset_files"] >= payload["audio_files"]
    assert payload["asset_bytes"] > 0
    assert payload["provenance"] is True


def test_info_counts_audio_outside_conventional_source_directories(capsys):
    assert main(["info", "car-engine", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_files"] == 0
    assert payload["audio_files"] == 21
    assert payload["provenance"] is True


def test_car_engine_exposes_bundled_sample_bank(capsys):
    assert main(["car-engine", "inspect"]) == 0
    output = capsys.readouterr().out
    assert "CC0 1.0" in output
    assert "freesound.org" in output
    assert "idle_0900.wav" in output


def test_car_engine_can_render_from_bundled_sample_bank(tmp_path):
    output = tmp_path / "sample-engine.wav"
    assert main([
        "car-engine", "sample", str(output), "--duration", "0.25",
        "--seed", "23", "--gear-shifts", "0",
    ]) == 0
    with wave.open(str(output), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
            1, 2, 44_100
        )
        assert wav.getnframes() == 11_025


def test_provider_text_exit_becomes_clean_cli_error(capsys):
    assert main([
        "car-engine", "sample", "--duration", "-1", "--gear-shifts", "0",
    ]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "duration must be between 0.2 and 120 seconds" in captured.err
    assert "Traceback" not in captured.err


def test_unknown_provider_suggests_a_close_name(capsys):
    assert main(["fir", "list"]) == 2
    assert "did you mean fire" in capsys.readouterr().err


def test_doctor_imports_every_provider(capsys):
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == len(PROVIDERS)
    assert {row["status"] for row in payload} == {"ok"}


def test_doctor_row_reports_validation_exceptions(monkeypatch):
    provider = provider_named("aircraft")
    original = Provider.validation_errors

    def validation_errors(self):
        if self is provider:
            raise TypeError("broken manifest shape")
        return original(self)

    monkeypatch.setattr(Provider, "validation_errors", validation_errors)
    row = _doctor_row(provider)
    assert row["status"] == "error"
    assert "validation TypeError: broken manifest shape" in row["error"]


def test_sample_manifest_layers_must_be_an_array(monkeypatch):
    registry_module = sys.modules["python_sound_generator.providers"]
    monkeypatch.setattr(
        registry_module.json,
        "loads",
        lambda _: {"source": "test", "license": "CC0", "layers": None},
    )
    errors = provider_named("car-engine").validation_errors()
    assert any("sample manifest layers is not an array" in error for error in errors)


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda provider: provider.slug)
def test_every_provider_exposes_native_help(provider, capsys):
    assert provider.invoke(["--help"]) == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()
    assert captured.err == ""


@pytest.mark.parametrize("name", ["fire", "ui", "projectiles", "wood"])
def test_provider_list_commands_dispatch(name, capsys):
    assert main([name, "list"]) == 0
    assert capsys.readouterr().out.strip()


def test_car_engine_can_render_through_the_unified_cli(tmp_path):
    output = tmp_path / "engine.wav"
    assert main([
        "car-engine", str(output), "--duration", "0.2", "--seed", "7",
        "--sample-rate", "44100",
    ]) == 0
    with wave.open(str(output), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
            1, 2, 44_100
        )
        assert wav.getnframes() == pytest.approx(8_820, abs=1)


@pytest.mark.parametrize(
    ("provider", "arguments"),
    [
        ("aircraft", ["render", "touchdown", "--duration", "0.5"]),
        ("vehicles", ["render", "collision", "--duration", "0.5"]),
        ("voice-foley", ["render", "pain", "--duration", "0.2"]),
        ("voice-foley", ["render", "death", "--duration", "0.2"]),
    ],
)
def test_cross_provider_compositions_render(provider, arguments, tmp_path, capsys):
    output = tmp_path / provider
    assert main([provider, *arguments, "--out", str(output)]) == 0
    assert list(output.rglob("*.wav"))
    assert "missing Python module" not in capsys.readouterr().err


@pytest.mark.parametrize("provider", ["human-voice", "magic"])
def test_nested_pack_manifest_paths_exist(provider, tmp_path, capsys):
    assert main([provider, "pack", "fantasy-adventure", "--out", str(tmp_path)]) == 0
    output = tmp_path / "fantasy-adventure"
    artifacts = json.loads((output / "manifest.json").read_text())["artifacts"]
    assert any("/" in artifact["file"] for artifact in artifacts)
    assert all((output / artifact["file"]).is_file() for artifact in artifacts)
    capsys.readouterr()


def test_wood_debris_uses_a_supported_default_form(tmp_path, capsys):
    output = tmp_path / "wood"
    assert main(["wood", "render", "debris", "--out", str(output)]) == 0
    artifact = json.loads((output / "manifest.json").read_text())["artifacts"][0]
    assert artifact["options"]["form"] == "crate"
    capsys.readouterr()


def test_default_output_stays_outside_the_installed_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main([
        "appliances", "render", "microwave", "--seconds", "1", "--seed", "17",
    ]) == 0

    output = tmp_path / "output" / "appliances"
    rendered = output / "microwave.wav"
    manifest = output / "manifest.json"
    assert rendered.is_file()
    assert manifest.is_file()
    with wave.open(str(rendered), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
            1, 2, 44_100
        )
        assert wav.getnframes() == 44_100
    artifact = json.loads(manifest.read_text())["artifacts"][0]
    assert artifact["sources"]
    assert artifact["sha256"]


def test_only_runtime_provider_material_is_packaged():
    forbidden = {"output", "out", "research", "tests"}
    for path in PROVIDER_ROOT.rglob("*"):
        assert not forbidden.intersection(path.parts), path


def test_no_provider_markdown_or_requirements_are_packaged():
    assert not list(PROVIDER_ROOT.rglob("*.md"))
    assert not list(PROVIDER_ROOT.rglob("requirements*.txt"))
