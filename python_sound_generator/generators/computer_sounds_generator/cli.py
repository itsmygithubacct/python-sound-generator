"""Command line for deterministic computer-machine sound rendering."""

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

from ..._audio import loops as loopdsp

from . import engine
from .cues import CUES, GROUPS, LOOPS, PACKS, Options, Render, render


DEFAULT_OUT = Path("output") / "computer"
PROVENANCE = engine.SOURCES / "provenance.json"


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _options(args: argparse.Namespace) -> Options:
    return Options(
        size=engine.size_named(args.size),
        region=engine.region_named(args.region),
        seconds=args.seconds,
        intensity=args.intensity,
    )


def _artifact(path: Path, result: Render, cue_name: str, seed: int,
              variation: int, options: Options) -> dict[str, Any]:
    signal = result.samples
    metrics = artifact_metrics(path, signal, engine.SR)
    entry: dict[str, Any] = {
        "file": path.name,
        "cue": cue_name,
        "seed": seed,
        "variation": variation,
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        "sources": list(result.sources),
        "options": {
            "size": options.size.name,
            "region": options.region.name,
            "intensity": options.intensity,
        },
    }
    if CUES[cue_name].loop:
        seam = loopdsp.seam(signal)
        entry["options"]["seconds"] = options.seconds
        entry["loop"] = {
            "seamless": seam.is_seamless(),
            "step_ratio": round(seam.step_ratio, 4),
            "level_ratio": round(seam.level_ratio, 4),
        }
    return entry


def _manifest(path: Path, artifacts: list[dict[str, Any]], command: str,
              pack: str | None = None) -> Path:
    payload: dict[str, Any] = {
        "generator": "computer_sounds_generator",
        "command": command,
        "fan_model": "aircraft_spacecraft_generator.engine.propeller",
        "source_audio": "CC0 only; see sources/provenance.json",
        "artifacts": artifacts,
    }
    if pack:
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
    show_all = not (args.cues or args.sizes or args.regions or args.packs)
    if args.cues or show_all:
        for group in GROUPS:
            print(f"\nCues: {group}")
            for cue in CUES.values():
                if cue.group == group:
                    print(f"  {cue.name:<10} {cue.description}")
                    print(f"  {'':<10} honours: {cue.honours}")
    if args.sizes or show_all:
        print("\nSizes")
        for size in engine.SIZES.values():
            print(f"  {size.name:<10} {size.description}")
    if args.regions or show_all:
        print("\nRegions")
        for region in engine.REGIONS.values():
            print(f"  {region.name:<10} {region.label}")
    if args.packs or show_all:
        print("\nPacks")
        for pack in PACKS.values():
            count = sum(entry.count for entry in pack.entries)
            print(f"  {pack.name:<10} {pack.description} ({count} files)")
    print()
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    del args
    print(PROVENANCE.read_text(encoding="utf-8"), end="")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    try:
        options = _options(args)
        out = Path(args.out)
        artifacts = []
        for index in range(args.count):
            seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
            result = render(args.cue, options, random.Random(seed))
            stem = args.stem or args.cue
            filename = f"{stem}.wav" if args.count == 1 else f"{stem}_v{index + 1:02d}.wav"
            path = engine.write_wav(out / filename, result.samples)
            artifacts.append(
                _artifact(path, result, args.cue, seed, index + 1, options)
            )
            print(f"wrote {path}")
            if args.play:
                _play(path)
        _manifest(out / "manifest.json", artifacts, "render")
        return 0
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`", file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out = Path(args.out) / pack.name
    artifacts = []
    for entry in pack.entries:
        for index in range(entry.count):
            seed = derive_seed(args.seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            result = render(entry.cue, entry.options, random.Random(seed))
            path = engine.write_wav(out / pack.filename(entry, index), result.samples)
            artifacts.append(
                _artifact(path, result, entry.cue, seed, index + 1, entry.options)
            )
            print(f"wrote {path}")
            if args.play:
                _play(path)
    _manifest(out / "manifest.json", artifacts, "pack", pack.name)
    print(f"{len(artifacts)} files")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    failures = 0
    for cue_name in LOOPS:
        for size in engine.SIZES.values():
            for region in engine.REGIONS.values():
                options = Options(
                    size=size, region=region, seconds=args.seconds,
                    intensity=args.intensity,
                )
                result = render(cue_name, options, random.Random(args.seed))
                seam = loopdsp.seam(result.samples)
                ok = seam.is_seamless()
                failures += 0 if ok else 1
                print(
                    f"{cue_name:<6} {size.name:<8} {region.name:<4} "
                    f"step={seam.step_ratio:.3f} level={seam.level_ratio:.3f} "
                    f"{'ok' if ok else 'FAIL'}"
                )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soundgen computer")
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list")
    listing.add_argument("--cues", action="store_true")
    listing.add_argument("--sizes", action="store_true")
    listing.add_argument("--regions", action="store_true")
    listing.add_argument("--packs", action="store_true")
    listing.set_defaults(func=cmd_list)

    sources = sub.add_parser("sources")
    sources.set_defaults(func=cmd_sources)

    rendering = sub.add_parser("render")
    rendering.add_argument("cue")
    rendering.add_argument("--size", default="desktop")
    rendering.add_argument("--region", default="ntsc")
    rendering.add_argument("--seconds", type=float, default=4.0)
    rendering.add_argument("--intensity", type=float, default=0.7)
    rendering.add_argument("--count", type=int, default=1)
    rendering.add_argument("--seed", type=int, default=20260716)
    rendering.add_argument("--stem")
    rendering.add_argument("--out", default=str(DEFAULT_OUT))
    rendering.add_argument("--play", action="store_true")
    rendering.set_defaults(func=cmd_render)

    packing = sub.add_parser("pack")
    packing.add_argument("pack")
    packing.add_argument("--seed", type=int, default=20260716)
    packing.add_argument("--out", default=str(DEFAULT_OUT))
    packing.add_argument("--play", action="store_true")
    packing.set_defaults(func=cmd_pack)

    verify = sub.add_parser("verify")
    verify.add_argument("--seconds", type=float, default=2.0)
    verify.add_argument("--intensity", type=float, default=0.7)
    verify.add_argument("--seed", type=int, default=20260716)
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "count", 1) < 1:
        print("count must be at least 1", file=sys.stderr)
        return 2
    return args.func(args)
