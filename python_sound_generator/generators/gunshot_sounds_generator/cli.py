"""Command line for gunshot context and firearm Foley."""

from __future__ import annotations

from ..._audio.runtime import artifact_metrics

from ..._audio.runtime import derive_seed

from ..._audio.runtime import play as play_audio

from ..._audio.runtime import write_manifest as write_shared_manifest

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import random

from . import engine
from .cues import CUES, GROUPS, PACKS, Options, Pack, render, stem_for


DEFAULT_OUT = Path("output") / "gunshots"
DEFAULT_SEED = 20260716
ROOT = Path(__file__).resolve().parents[1]


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _options(args: argparse.Namespace) -> Options:
    return Options(
        weapon=engine.firearm_named(args.weapon),
        space=engine.space_named(args.space),
        suppression=engine.suppression_named(args.suppression),
        surface=engine.surface_named(args.surface),
        shots=args.shots,
        intensity=args.intensity,
    )


def _option_values(options: Options) -> dict[str, Any]:
    return {
        "weapon": options.weapon.name,
        "space": options.space.name,
        "suppression": options.suppression.name,
        "surface": options.surface.name,
        "shots": options.shots,
        "intensity": round(options.intensity, 4),
    }


def _honoured_options(cue_name: str, options: Options) -> dict[str, Any]:
    honoured = {
        value.strip() for value in CUES[cue_name].honours
        if value.strip()
    }
    return {
        name: value for name, value in _option_values(options).items()
        if name in honoured
    }


def _source_hashes(sources: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for source in sources:
        if source.startswith(("generator:", "generated-space:",
                              "suppression-balance:")):
            continue
        path = ROOT / source
        if not path.is_file():
            raise ValueError(f"render named a missing source: {source}")
        hashes[source] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _artifact(path: Path, signal: list[float], sources: tuple[str, ...],
              cue_name: str, seed: int, variation: int,
              options: Options) -> dict[str, Any]:
    metrics = artifact_metrics(path, signal, engine.SR)
    return {
        "file": path.name,
        "cue": cue_name,
        "seed": seed,
        "variation": variation,
        "options": _honoured_options(cue_name, options),
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        "sources": list(sources),
        "source_sha256": _source_hashes(sources),
    }


def _write_manifest(path: Path, artifacts: list[dict[str, Any]], *,
                    command: str, pack: str | None = None) -> Path:
    payload: dict[str, Any] = {
        "generator": "gunshot_sounds_generator",
        "command": command,
        "source_audio": (
            "CC0 firearm Foley plus recordings consumed in place from "
            "projectile_sounds, stone_noise_generator, and wood_noise_generator"
        ),
        "source_provenance": {
            "gunshot_sounds_generator": (
                "gunshot_sounds_generator/sources/provenance.json"
            ),
            "projectile_sounds": "projectile_sounds/sources/provenance.json",
            "stone_noise_generator": (
                "stone_noise_generator/sources/provenance.json"
            ),
            "wood_noise_generator": "wood_noise_generator/sources/provenance.json",
            "report": "weapons_destruction_generator (procedural)",
        },
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
    selected = any((args.cues, args.weapons, args.spaces, args.suppressions,
                    args.surfaces, args.packs))
    if args.cues or not selected:
        for group in GROUPS:
            print(f"\nCues: {group}")
            for cue in CUES.values():
                if cue.group == group:
                    print(f"  {cue.name:<18} {cue.description}")
                    print(f"  {'':<18} honours: {cue.honours}")
    if args.weapons or not selected:
        print("\nWeapons (borrowed from weapons_destruction_generator)")
        for profile in engine.FIREARMS.values():
            print(f"  {profile.name:<18} {profile.description}")
    if args.spaces or not selected:
        print("\nSpaces")
        for profile in engine.SPACES.values():
            print(f"  {profile.name:<18} {profile.description}")
    if args.suppressions or not selected:
        print("\nSuppression balances")
        for profile in engine.SUPPRESSIONS.values():
            print(f"  {profile.name:<18} {profile.description}")
    if args.surfaces or not selected:
        print("\nCasing surfaces")
        for profile in engine.SURFACES.values():
            print(f"  {profile.name:<18} {profile.description}")
    if args.packs or not selected:
        print("\nPacks")
        for pack in PACKS.values():
            count = sum(entry.count for entry in pack.entries)
            print(f"  {pack.name:<18} {pack.description} ({count} files)")
    print()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    options = _options(args)
    print(f"weapon: {options.weapon.label}\n  {options.weapon.description}")
    print(f"  category {options.weapon.category}; cadence "
          f"{options.weapon.rounds_per_second:.2f}/s")
    print(f"space: {options.space.label}\n  {options.space.description}")
    print(f"  tail {options.space.tail_seconds:.2f}s; cutoff "
          f"{options.space.tail_cutoff:.0f} Hz")
    print(f"suppression: {options.suppression.label}\n  "
          f"{options.suppression.description}")
    print(f"surface: {options.surface.label}\n  {options.surface.description}")
    if args.cue:
        if args.cue not in CUES:
            print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
            return 2
        cue = CUES[args.cue]
        print(f"cue: {cue.name}\n  {cue.description}\n  group {cue.group}; "
              f"peak {cue.peak_db:.1f} dBFS; honours {cue.honours}")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    print("Auditioned firearm Foley:")
    path = ROOT / "gunshot_sounds_generator" / "sources" / "provenance.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"  gunshot_sounds_generator   {data['source_audio_policy']}")
    print(f"  {'':<27} {path}")
    print("Consumed material source banks:")
    for package in ("projectile_sounds", "stone_noise_generator",
                    "wood_noise_generator"):
        path = ROOT / package / "sources" / "provenance.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        policy = data.get("source_audio_policy", "CC0 recordings")
        print(f"  {package:<28} {policy}")
        print(f"  {'':<28} {path}")
    print("Procedural report:")
    print("  weapons_destruction_generator/fire")
    return 0


def _validate(args: argparse.Namespace) -> int:
    if getattr(args, "count", 1) < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    if not 2 <= args.shots <= 12:
        print("--shots must be between 2 and 12", file=sys.stderr)
        return 2
    if not 0.05 <= args.intensity <= 1.0:
        print("--intensity must be between 0.05 and 1", file=sys.stderr)
        return 2
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    invalid = _validate(args)
    if invalid:
        return invalid
    options = _options(args)
    out_dir = Path(args.out)
    stem = args.stem or stem_for(args.cue, options)
    artifacts: list[dict[str, Any]] = []
    last_path: Path | None = None
    print(f"{args.cue}: {CUES[args.cue].description}")
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        result = render(args.cue, options, random.Random(seed))
        filename = f"{stem}.wav" if index == 0 else f"{stem}_v{index + 1:02d}.wav"
        last_path = engine.write_wav(out_dir / filename, result.samples)
        artifacts.append(_artifact(
            last_path, result.samples, result.sources, args.cue, seed,
            index + 1, options,
        ))
        print(f"  wrote {last_path}  {len(result.samples) / engine.SR:.2f}s")
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="render")
    print(f"  manifest {manifest}")
    if args.play and last_path is not None:
        _play(last_path)
    return 0


def _render_pack(pack: Pack, out_dir: Path, seed: int,
                 play: bool) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for entry in pack.entries:
        for index in range(entry.count):
            file_seed = derive_seed(seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            result = render(entry.cue, entry.options, random.Random(file_seed))
            path = engine.write_wav(out_dir / pack.filename(entry, index),
                                    result.samples)
            artifacts.append(_artifact(
                path, result.samples, result.sources, entry.cue, file_seed,
                index + 1, entry.options,
            ))
            print(f"  wrote {path.name}  {len(result.samples) / engine.SR:.2f}s")
            if play:
                _play(path)
    return artifacts


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`", file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")
    artifacts = _render_pack(pack, out_dir, args.seed, args.play)
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="pack", pack=pack.name)
    print(f"  {len(artifacts)} files; manifest {manifest}")
    return 0


def cmd_audition(args: argparse.Namespace) -> int:
    from .audition import print_queue, print_status, run_audition

    if args.list:
        print_queue()
        return 0
    if args.status:
        print_status(Path(args.out))
        return 0
    return run_audition(
        Path(args.out), args.seed, restart=args.restart,
        preferred_player=args.player,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen gunshots",
        description=(
            "Compose firearm reports with recorded mechanisms, reloads, casing "
            "contacts, suppression balance, and acoustic spaces."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show registries and packs")
    p_list.add_argument("--cues", action="store_true")
    p_list.add_argument("--weapons", action="store_true")
    p_list.add_argument("--spaces", action="store_true")
    p_list.add_argument("--suppressions", action="store_true")
    p_list.add_argument("--surfaces", action="store_true")
    p_list.add_argument("--packs", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="describe selected profiles")
    p_info.add_argument("--cue")
    p_info.set_defaults(func=cmd_info)

    p_sources = sub.add_parser("sources", help="show borrowed source provenance")
    p_sources.set_defaults(func=cmd_sources)

    p_render = sub.add_parser("render", help="render one cue")
    p_render.add_argument("cue", help="cue name; see `list --cues`")
    p_render.add_argument("--count", type=int, default=1)
    p_render.add_argument("--stem")
    p_render.set_defaults(func=cmd_render)

    p_pack = sub.add_parser("pack", help="render a named bank")
    p_pack.add_argument("pack", help="pack name; see `list --packs`")
    p_pack.set_defaults(func=cmd_pack)

    p_audition = sub.add_parser(
        "audition",
        help="play a resumable one-at-a-time yes/no audition",
    )
    p_audition.add_argument(
        "--list", action="store_true", help="show the queue without starting"
    )
    p_audition.add_argument(
        "--status", action="store_true", help="show saved progress and exit"
    )
    p_audition.add_argument(
        "--restart", action="store_true", help="discard saved verdicts and restart"
    )
    p_audition.add_argument(
        "--player", choices=("paplay", "aplay", "play", "ffplay"),
        help="force one player instead of trying installed players in order",
    )
    p_audition.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_audition.add_argument("--out", type=Path,
                            default=DEFAULT_OUT / "audition")
    p_audition.set_defaults(func=cmd_audition)

    for target in (p_info, p_render):
        target.add_argument("--weapon", default="rifle", choices=engine.FIREARMS)
        target.add_argument("--space", default="outdoor", choices=engine.SPACES)
        target.add_argument("--suppression", default="unsuppressed",
                            choices=engine.SUPPRESSIONS)
        target.add_argument("--surface", default="concrete",
                            choices=engine.SURFACES)
        target.add_argument("--shots", type=int, default=5)
        target.add_argument("--intensity", type=float, default=0.85)
    for target in (p_render, p_pack):
        target.add_argument("--seed", type=int, default=DEFAULT_SEED)
        target.add_argument("--out", type=Path, default=DEFAULT_OUT)
        target.add_argument("--play", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"soundgen gunshots: {error}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
