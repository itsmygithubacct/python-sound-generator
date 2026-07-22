"""Registry-driven command line for projectile impact SFX."""

from __future__ import annotations

from ..._audio.runtime import artifact_metrics

from ..._audio.runtime import derive_seed

from ..._audio.runtime import play as play_audio

from ..._audio.runtime import write_manifest as write_shared_manifest

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


from . import engine
from .cues import (
    CUES,
    GROUPS,
    PACKS,
    Options,
    render,
    stem_for,
    weapons_render,
    weapons_seed,
)


DEFAULT_OUT = Path("output") / "projectiles"
# Deliberately weapons_destruction_generator's default and not a fresh one: the
# two CLIs must agree, or a borrowed report stops being byte-identical.
DEFAULT_SEED = 20260715


def _header(text: str) -> None:
    print(text)


def _item(name: str, detail: str) -> None:
    print(f"  {name.ljust(20)} {detail}")


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _options_from_args(args: argparse.Namespace) -> Options:
    return Options(
        projectile=engine.projectile_named(args.projectile),
        material=engine.material_named(args.material),
        intensity=args.intensity,
    )


def _options_dict(options: Options, cue_name: str) -> dict[str, Any]:
    values = {
        "projectile": options.projectile.name,
        "material": options.material.name,
        "intensity": options.intensity,
    }
    honoured = {
        name.strip() for name in CUES[cue_name].honours
        if name.strip()
    }
    return {name: values[name] for name in values if name in honoured}


def _artifact(path: Path, samples, cue_name: str, seed: int, variation: int,
              options: Options, sources: tuple[str, ...]) -> dict[str, Any]:
    data = np.asarray(samples)
    metrics = artifact_metrics(path, data, engine.SR)
    return {
        "file": path.name,
        "cue": cue_name,
        "renderer": "foley",
        "seed": seed,
        "variation": variation,
        "options": _options_dict(options, cue_name),
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        # Which recordings went into this exact file. A licence you cannot
        # trace to an artifact is not much of a licence.
        "sources": sorted(set(sources)),
    }


def _write_manifest(path: Path, artifacts: list[dict[str, Any]], *,
                    command: str, pack: str | None = None) -> Path:
    payload: dict[str, Any] = {
        "generator": "projectile_sounds",
        "command": command,
        "source_licensing": (
            "projectile_sounds/sources/provenance.json and "
            "wood_noise_generator/sources/provenance.json (CC0 only)"
        ),
        "artifacts": artifacts,
    }
    if pack is not None:
        payload["pack"] = pack
    generator = payload.pop("generator")
    command_name = payload.pop("command")
    artifact_rows = payload.pop("artifacts")
    return write_shared_manifest(
        path,
        generator=generator,
        command=command_name,
        artifacts=artifact_rows,
        extra=payload,
    )


def cmd_list(args: argparse.Namespace) -> int:
    selected = any((args.cues, args.projectiles, args.materials, args.packs))
    if args.cues or not selected:
        for group in GROUPS:
            _header(f"{group} cues")
            for cue in CUES.values():
                if cue.group != group:
                    continue
                _item(cue.name, cue.description)
                print(f"  {'':20} honours: {cue.honours or 'no options'}")
        print()
    if args.projectiles or not selected:
        _header("projectiles (weapons_destruction_generator's registry)")
        for profile in engine.PROJECTILES.values():
            _item(profile.name,
                  f"{profile.kind}; weight {profile.weight:.2f}")
        print()
    if args.materials or not selected:
        _header("materials")
        for material in engine.MATERIALS.values():
            _item(material.name, material.description)
        print()
    if args.packs or not selected:
        _header("packs")
        for pack in PACKS.values():
            count = sum(entry.count for entry in pack.entries)
            detail = f"{pack.description} ({count} files"
            if pack.borrows:
                detail += f" + {len(pack.borrows)} borrowed"
            _item(pack.name, detail + ")")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    projectile = engine.projectile_named(args.projectile)
    material = engine.material_named(args.material)

    _header(f"projectile: {projectile.name}")
    print(f"  kind {projectile.kind}; weight {projectile.weight:.2f}; "
          f"brightness {projectile.brightness:.2f}")
    print(f"  plays its contact at {engine._weight_speed(projectile):.2f}x "
          f"speed: heavier is slower, lower and longer")

    _header(f"material: {material.label}")
    print(f"  {material.description}")
    print(f"  contact {list(material.contact)}")
    print(f"  debris  {list(material.debris) or 'none — it throws nothing'}")
    if material.wet:
        print(f"  wet     {list(material.wet)}")
    print(f"  darkened above {material.darken:.0f} Hz; "
          f"sub at {material.sub_frequency:.0f} Hz")

    if args.cue:
        if args.cue not in CUES:
            print(f"unknown cue {args.cue!r}; see `list --cues`",
                  file=sys.stderr)
            return 2
        cue = CUES[args.cue]
        print()
        _header(f"cue: {cue.name}")
        print(f"  {cue.description}")
        print(f"  group {cue.group}; peak {cue.peak_db:.1f} dBFS; "
              f"honours {cue.honours or 'no options'}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    if args.count < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    if not 0.05 <= args.intensity <= 1.0:
        print("--intensity must be between 0.05 and 1", file=sys.stderr)
        return 2

    options = _options_from_args(args)
    out_dir = Path(args.out)
    stem = args.stem or stem_for(args.cue, options)
    artifacts: list[dict[str, Any]] = []
    last: Path | None = None
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        result = render(args.cue, options, random.Random(seed))
        name = f"{stem}.wav" if index == 0 else f"{stem}_v{index + 1:02d}.wav"
        last = engine.write_wav(out_dir / name, result.samples)
        artifacts.append(_artifact(last, result.samples, args.cue, seed,
                                   index + 1, options, result.sources))
        print(f"  wrote {last}  {len(result.samples) / engine.SR:.2f}s  "
              f"from {len(set(result.sources))} recordings")
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="render")
    print(f"  manifest {manifest}")
    if args.play and last is not None:
        _play(last)
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`",
              file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name
    artifacts: list[dict[str, Any]] = []
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")

    for entry in pack.entries:
        for index in range(entry.count):
            seed = derive_seed(args.seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            result = render(entry.cue, entry.options, random.Random(seed))
            path = engine.write_wav(out_dir / pack.filename(entry, index),
                                    result.samples)
            artifacts.append(_artifact(path, result.samples, entry.cue, seed,
                                       index + 1, entry.options,
                                       result.sources))
            print(f"  wrote {path.name}  "
                  f"{len(result.samples) / engine.SR:.2f}s")
            if args.play:
                _play(path)

    for entry in pack.borrowed_entries():
        for index in range(entry.count):
            seed = weapons_seed(entry, index, args.seed)
            samples = weapons_render(entry, index, args.seed)
            path = engine.write_wav(out_dir / pack.filename(entry, index),
                                    samples)
            artifact = {
                "file": path.name,
                "cue": entry.cue,
                "renderer": "procedural",
                "borrowed_from": "weapons_destruction_generator",
                "seed": seed,
                "variation": index + 1,
                "duration_seconds": round(len(samples) / engine.SR, 6),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            artifacts.append(artifact)
            print(f"  wrote {path.name}  {len(samples) / engine.SR:.2f}s  "
                  f"(borrowed)")
            if args.play:
                _play(path)

    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="pack", pack=pack.name)
    print(f"  {len(artifacts)} files; manifest {manifest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen projectiles",
        description=(
            "Generate projectile impacts and debris from bundled CC0 "
            "recordings. Reports, flybys and blasts stay in "
            "weapons_destruction_generator and are borrowed by the game packs."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<verb>")

    list_parser = sub.add_parser("list", help="show registries and packs")
    list_parser.add_argument("--cues", action="store_true")
    list_parser.add_argument("--projectiles", action="store_true")
    list_parser.add_argument("--materials", action="store_true")
    list_parser.add_argument("--packs", action="store_true")
    list_parser.set_defaults(func=cmd_list)

    info_parser = sub.add_parser("info", help="describe the selected models")
    info_parser.add_argument("--cue", help="also describe a cue")
    info_parser.set_defaults(func=cmd_info)

    render_parser = sub.add_parser("render", help="render one cue")
    render_parser.add_argument("cue", help="cue name; see `list --cues`")
    render_parser.add_argument("--count", type=int, default=1)
    render_parser.add_argument("--stem",
                               help="override the generated filename stem")
    render_parser.set_defaults(func=cmd_render)

    pack_parser = sub.add_parser("pack", help="render a named artifact bank")
    pack_parser.add_argument("pack", help="pack name; see `list --packs`")
    pack_parser.set_defaults(func=cmd_pack)

    for target in (info_parser, render_parser):
        target.add_argument("--projectile", default="bullet",
                            choices=engine.PROJECTILES)
        target.add_argument("--material", default="metal",
                            choices=engine.MATERIALS)
    render_parser.add_argument("--intensity", type=float, default=0.85)
    for target in (render_parser, pack_parser):
        target.add_argument("--seed", type=int, default=DEFAULT_SEED)
        target.add_argument("--out", type=Path, default=DEFAULT_OUT)
        target.add_argument("--play", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (OSError, ValueError, KeyError) as error:
        print(f"soundgen projectiles: {error}", file=sys.stderr)
        return 1
