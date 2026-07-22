"""Registry-driven command line for weapons and destruction SFX."""

from __future__ import annotations

from ..._audio.runtime import artifact_metrics

from ..._audio.runtime import derive_seed

from ..._audio.runtime import play as play_audio

from ..._audio.runtime import write_manifest as write_shared_manifest

import argparse
import random
import sys
from pathlib import Path
from typing import Any

from . import engine
from .cues import CUES, GROUPS, PACKS, Options, render, stem_for


DEFAULT_OUT = Path("output") / "weapons"
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
        weapon=engine.firearm_named(args.weapon),
        projectile=engine.projectile_named(args.projectile),
        explosive=engine.explosive_named(args.explosive),
        material=engine.material_named(args.material),
        shots=args.shots,
        intensity=args.intensity,
    )


def _options_dict(options: Options, cue_name: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "weapon": options.weapon.name,
        "projectile": options.projectile.name,
        "explosive": options.explosive.name,
        "material": options.material.name,
        "shots": options.shots,
        "intensity": options.intensity,
    }
    honoured = {
        name.strip() for name in CUES[cue_name].honours if name.strip()
    }
    return {name: values[name] for name in values if name in honoured}


def _artifact(
    path: Path,
    signal: list[float],
    cue_name: str,
    seed: int,
    variation: int,
    options: Options,
) -> dict[str, Any]:
    metrics = artifact_metrics(path, signal, engine.SR)
    return {
        "file": path.name,
        "cue": cue_name,
        "seed": seed,
        "variation": variation,
        "options": _options_dict(options, cue_name),
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
    }


def _write_manifest(
    path: Path,
    artifacts: list[dict[str, Any]],
    *,
    command: str,
    pack: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "generator": "weapons_destruction_generator",
        "command": command,
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
    selected = any(
        (args.cues, args.weapons, args.projectiles, args.explosives,
         args.materials, args.packs)
    )
    if args.cues or not selected:
        for group in GROUPS:
            _header(f"{group} cues")
            for cue in CUES.values():
                if cue.group == group:
                    _item(cue.name, cue.description)
                    if cue.honours:
                        print(f"  {'':20} honours: {cue.honours}")
        print()
    if args.weapons or not selected:
        _header("weapons")
        for profile in engine.FIREARMS.values():
            _item(profile.name, profile.description)
        print()
    if args.projectiles or not selected:
        _header("projectiles")
        for profile in engine.PROJECTILES.values():
            _item(profile.name, profile.description)
        print()
    if args.explosives or not selected:
        _header("explosives")
        for profile in engine.EXPLOSIVES.values():
            _item(profile.name, profile.description)
        print()
    if args.materials or not selected:
        _header("impact materials")
        for material in engine.MATERIALS.values():
            _item(material.name, material.description)
        print()
    if args.packs or not selected:
        _header("packs")
        for pack in PACKS.values():
            count = sum(entry.count for entry in pack.entries)
            _item(pack.name, f"{pack.description} ({count} files)")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    weapon = engine.firearm_named(args.weapon)
    projectile = engine.projectile_named(args.projectile)
    explosive = engine.explosive_named(args.explosive)
    material = engine.material_named(args.material)

    _header(f"weapon: {weapon.label}")
    print(f"  {weapon.description}")
    print(f"  category {weapon.category}; scale {weapon.scale:.2f}; "
          f"body {weapon.body_frequency:.0f} Hz; "
          f"cadence {weapon.rounds_per_second:.2f}/s")
    _header(f"projectile: {projectile.label}")
    print(f"  {projectile.description}")
    print(f"  kind {projectile.kind}; impact weight {projectile.weight:.2f}; "
          f"flyby {projectile.flyby_duration:.2f}s")
    _header(f"explosive: {explosive.label}")
    print(f"  {explosive.description}")
    print(f"  scale {explosive.scale:.2f}; body {explosive.body_frequency:.0f} Hz; "
          f"tail {explosive.duration:.2f}s")
    _header(f"material: {material.label}")
    print(f"  {material.description}")
    print(f"  kind {material.kind}; contact {material.contact_low:.0f}-"
          f"{material.contact_high:.0f} Hz; {len(material.modes)} ring modes")
    if args.cue:
        if args.cue not in CUES:
            print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
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
    if not 2 <= args.shots <= 12:
        print("--shots must be between 2 and 12", file=sys.stderr)
        return 2
    if not 0.05 <= args.intensity <= 1.0:
        print("--intensity must be between 0.05 and 1", file=sys.stderr)
        return 2

    options = _options_from_args(args)
    out_dir = Path(args.out)
    stem = args.stem or stem_for(args.cue, options)
    artifacts: list[dict[str, Any]] = []
    last_path: Path | None = None
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        signal = render(args.cue, options, random.Random(seed))
        filename = f"{stem}.wav" if index == 0 else f"{stem}_v{index + 1:02d}.wav"
        last_path = engine.write_wav(out_dir / filename, signal)
        artifacts.append(_artifact(
            last_path, signal, args.cue, seed, index + 1, options
        ))
        print(f"  wrote {last_path}  {len(signal) / engine.SR:.2f}s")
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="render")
    print(f"  manifest {manifest}")
    if args.play and last_path is not None:
        _play(last_path)
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`", file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name
    artifacts: list[dict[str, Any]] = []
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")
    for entry in pack.entries:
        for index in range(entry.count):
            seed = derive_seed(args.seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            signal = render(entry.cue, entry.options, random.Random(seed))
            path = engine.write_wav(out_dir / pack.filename(entry, index), signal)
            artifacts.append(_artifact(
                path, signal, entry.cue, seed, index + 1, entry.options
            ))
            print(f"  wrote {path.name}  {len(signal) / engine.SR:.2f}s")
            if args.play:
                _play(path)
    manifest = _write_manifest(
        out_dir / "manifest.json", artifacts, command="pack", pack=pack.name
    )
    print(f"  {len(artifacts)} files; manifest {manifest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen weapons",
        description=(
            "Generate firearms, energy weapons, ordnance, explosions, and "
            "ricochets with dependency-free procedural DSP."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<verb>")

    list_parser = sub.add_parser("list", help="show registries and packs")
    list_parser.add_argument("--cues", action="store_true")
    list_parser.add_argument("--weapons", action="store_true")
    list_parser.add_argument("--projectiles", action="store_true")
    list_parser.add_argument("--explosives", action="store_true")
    list_parser.add_argument("--materials", action="store_true")
    list_parser.add_argument("--packs", action="store_true")
    list_parser.set_defaults(func=cmd_list)

    info_parser = sub.add_parser("info", help="describe the selected models")
    info_parser.add_argument("--cue", help="also describe a cue")
    info_parser.set_defaults(func=cmd_info)

    render_parser = sub.add_parser("render", help="render one cue")
    render_parser.add_argument("cue", help="cue name; see `list --cues`")
    render_parser.add_argument("--count", type=int, default=1)
    render_parser.add_argument("--stem", help="override the generated filename stem")
    render_parser.set_defaults(func=cmd_render)

    pack_parser = sub.add_parser("pack", help="render a named artifact bank")
    pack_parser.add_argument("pack", help="pack name; see `list --packs`")
    pack_parser.set_defaults(func=cmd_pack)

    for target in (info_parser, render_parser):
        target.add_argument("--weapon", default="rifle", choices=engine.FIREARMS)
        target.add_argument(
            "--projectile", default="bullet", choices=engine.PROJECTILES
        )
        target.add_argument(
            "--explosive", default="grenade", choices=engine.EXPLOSIVES
        )
        target.add_argument("--material", default="metal", choices=engine.MATERIALS)
    for target in (render_parser,):
        target.add_argument("--shots", type=int, default=4)
        target.add_argument("--intensity", type=float, default=0.85)
    for target in (render_parser, pack_parser):
        target.add_argument("--seed", type=int, default=DEFAULT_SEED)
        target.add_argument("--out", type=Path, default=DEFAULT_OUT)
        target.add_argument("--play", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (OSError, ValueError) as error:
        print(f"soundgen weapons: {error}", file=sys.stderr)
        return 1
