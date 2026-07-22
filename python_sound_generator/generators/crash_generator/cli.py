"""Registry-driven command line for crash composition."""

from __future__ import annotations

from ..._audio.runtime import artifact_metrics

from ..._audio.runtime import derive_seed

from ..._audio.runtime import play as play_audio

from ..._audio.runtime import write_manifest as write_shared_manifest

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import engine
from .cues import CUES, GROUPS, PACKS, Options, Pack, render, stem_for


DEFAULT_OUT = Path("output") / "crash"
DEFAULT_SEED = 20260716
PROVENANCE = {
    "projectile_sounds": "projectile_sounds/sources/provenance.json",
    "stone_noise_generator": "stone_noise_generator/sources/provenance.json",
    "wood_noise_generator": "wood_noise_generator/sources/provenance.json",
    "aircraft_spacecraft_generator": (
        "aircraft_spacecraft_generator/sources/provenance.json"
    ),
}


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _options(args: argparse.Namespace) -> Options:
    return Options(
        structure=engine.structure_named(args.structure),
        surface=engine.surface_named(args.surface),
        speed=args.speed,
        intensity=args.intensity,
    )


def _artifact(path: Path, samples: list[float], sources: tuple[str, ...],
              cue_name: str, seed: int, variation: int,
              options: Options) -> dict[str, Any]:
    metrics = artifact_metrics(path, samples, engine.SR)
    honoured = {
        name.strip() for name in CUES[cue_name].honours
        if name.strip()
    }
    option_values: dict[str, Any] = {
        "structure": options.structure.name,
        "surface": options.surface.name,
        "speed": round(options.speed, 4),
        "intensity": round(options.intensity, 4),
    }
    return {
        "file": path.name,
        "cue": cue_name,
        "seed": seed,
        "variation": variation,
        "options": {
            key: value for key, value in option_values.items()
            if key in honoured
        },
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        "sources": list(sources),
    }


def _write_manifest(path: Path, artifacts: list[dict[str, Any]], *,
                    command: str, pack: str | None = None) -> Path:
    payload: dict[str, Any] = {
        "generator": "crash_generator",
        "command": command,
        "source_audio": "CC0 recordings composed from existing material banks",
        "provenance": PROVENANCE,
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
    show_all = not (args.cues or args.structures or args.surfaces or args.packs)
    if args.cues or show_all:
        for group in GROUPS:
            print(f"\nCues: {group}")
            for cue in CUES.values():
                if cue.group == group:
                    print(f"  {cue.name:<12} {cue.description}")
                    print(f"  {'':<12} honours: {cue.honours}")
    if args.structures or show_all:
        print("\nStructures")
        for profile in engine.STRUCTURES.values():
            materials = "+".join(c.material for c in profile.components)
            print(f"  {profile.name:<12} {profile.description} [{materials}]")
    if args.surfaces or show_all:
        print("\nSurfaces")
        for surface in engine.SURFACES.values():
            print(f"  {surface.name:<12} {surface.description}")
    if args.packs or show_all:
        print("\nPacks")
        for pack in PACKS.values():
            count = sum(entry.count for entry in pack.entries)
            print(f"  {pack.name:<16} {pack.description} ({count} files)")
    print()
    return 0


def _render_pack(pack: Pack, out_dir: Path, seed: int,
                 play: bool) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for entry in pack.entries:
        for index in range(entry.count):
            file_seed = derive_seed(seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            result = render(
                entry.cue, entry.options, np.random.default_rng(file_seed)
            )
            path = engine.write_wav(
                out_dir / pack.filename(entry, index), result.samples
            )
            artifacts.append(_artifact(
                path, result.samples, result.sources, entry.cue, file_seed,
                index + 1, entry.options,
            ))
            print(f"  wrote {path.name}  {len(result.samples) / engine.SR:.2f}s")
            if play:
                _play(path)
    return artifacts


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    options = _options(args)
    out_dir = Path(args.out)
    artifacts: list[dict[str, Any]] = []
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        result = render(args.cue, options, np.random.default_rng(seed))
        stem = args.stem or stem_for(args.cue, options)
        filename = f"{stem}.wav" if index == 0 else f"{stem}_v{index + 1:02d}.wav"
        path = engine.write_wav(out_dir / filename, result.samples)
        artifacts.append(_artifact(
            path, result.samples, result.sources, args.cue, seed, index + 1,
            options,
        ))
        print(f"  wrote {path}  {len(result.samples) / engine.SR:.2f}s")
        if args.play:
            _play(path)
    manifest = _write_manifest(
        out_dir / "manifest.json", artifacts, command="render"
    )
    print(f"  manifest {manifest}")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`",
              file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")
    artifacts = _render_pack(pack, out_dir, args.seed, args.play)
    manifest = _write_manifest(
        out_dir / "manifest.json", artifacts, command="pack", pack=pack.name
    )
    print(f"  {len(artifacts)} files; manifest {manifest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen crash",
        description="Compose crashes from existing CC0 material Foley banks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show cues, axes, and packs")
    p_list.add_argument("--cues", action="store_true")
    p_list.add_argument("--structures", action="store_true")
    p_list.add_argument("--surfaces", action="store_true")
    p_list.add_argument("--packs", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_render = sub.add_parser("render", help="render one cue")
    p_render.add_argument("cue", help="cue name; see `list --cues`")
    p_render.add_argument("--count", type=int, default=1)
    p_render.add_argument("--stem", default=None)
    p_render.set_defaults(func=cmd_render)

    p_pack = sub.add_parser("pack", help="render a named artifact bank")
    p_pack.add_argument("pack", help="pack name; see `list --packs`")
    p_pack.set_defaults(func=cmd_pack)

    for command in (p_render, p_pack):
        command.add_argument("--seed", type=int, default=DEFAULT_SEED)
        command.add_argument("--out", type=Path, default=DEFAULT_OUT)
        command.add_argument("--play", action="store_true")
    p_render.add_argument("--structure", default="sedan",
                          help=f"one of: {', '.join(engine.STRUCTURES)}")
    p_render.add_argument("--surface", default="asphalt",
                          help=f"one of: {', '.join(engine.SURFACES)}")
    p_render.add_argument("--speed", type=float, default=0.5)
    p_render.add_argument("--intensity", type=float, default=0.85)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "count", 1) < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
