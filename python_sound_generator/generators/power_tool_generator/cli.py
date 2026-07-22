"""Command line for deterministic power-tool Foley rendering."""

from __future__ import annotations

from ..._audio.runtime import artifact_metrics

from ..._audio.runtime import derive_seed

from ..._audio.runtime import write_manifest as write_shared_manifest

import argparse
import random
from pathlib import Path

from . import engine
from .cues import CUES, PACKS, Options, Render, render


DEFAULT_OUT = Path("output") / "power-tools"
PROVENANCE = engine.SOURCES / "provenance.json"


def _options(args: argparse.Namespace) -> Options:
    return Options(args.seconds, args.speed, args.brightness, args.intensity)


def _artifact(path: Path, result: Render, cue: str, seed: int,
              options: Options) -> dict:
    signal = result.samples
    metrics = artifact_metrics(path, signal, engine.SR)
    return {
        "file": path.name,
        "cue": cue,
        "seed": seed,
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        "sources": list(result.sources),
        "options": {
            "seconds": options.seconds,
            "speed": options.speed,
            "brightness": options.brightness,
            "intensity": options.intensity,
        },
    }


def _manifest(path: Path, command: str, artifacts: list[dict],
              pack: str | None = None) -> None:
    payload = {
        "generator": "power_tool_generator",
        "command": command,
        "source_audio": "CC0 only; see sources/provenance.json",
        "artifacts": artifacts,
    }
    if pack is not None:
        payload["pack"] = pack
    write_shared_manifest(
        path,
        generator=payload.pop("generator"),
        command=payload.pop("command"),
        artifacts=payload.pop("artifacts"),
        extra=payload,
    )


def cmd_list(args: argparse.Namespace) -> int:
    del args
    print("Cues")
    for cue in CUES.values():
        print(f"  {cue.name:<16} {cue.description}")
    print("\nPacks")
    for pack in PACKS.values():
        print(f"  {pack.name:<16} {pack.description}")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    del args
    print(PROVENANCE.read_text(encoding="utf-8"), end="")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    try:
        options = _options(args)
        out = Path(args.out)
        artifacts = []
        for index in range(args.count):
            seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
            result = render(args.cue, options, random.Random(seed))
            stem = args.stem or args.cue
            name = f"{stem}.wav" if args.count == 1 else f"{stem}_v{index + 1:02d}.wav"
            path = engine.write_wav(out / name, result.samples)
            artifacts.append(_artifact(path, result, args.cue, seed, options))
            print(f"wrote {path}")
        _manifest(out / "manifest.json", "render", artifacts)
        return 0
    except (OSError, ValueError) as error:
        print(error)
        return 2


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; expected one of {sorted(PACKS)}")
        return 2
    pack = PACKS[args.pack]
    out = Path(args.out) / pack.name
    artifacts = []
    try:
        for entry in pack.entries:
            seed = derive_seed(
                args.seed,
                provider=__package__,
                cue=entry.cue,
                filename=entry.filename,
            )
            result = render(entry.cue, entry.options, random.Random(seed))
            path = engine.write_wav(out / f"{entry.filename}.wav", result.samples)
            artifacts.append(_artifact(path, result, entry.cue, seed, entry.options))
            print(f"wrote {path}")
        _manifest(out / "manifest.json", "pack", artifacts, pack.name)
        print(f"{len(artifacts)} files")
        return 0
    except (OSError, ValueError) as error:
        print(error)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soundgen power-tools")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.set_defaults(func=cmd_list)
    sources = sub.add_parser("sources")
    sources.set_defaults(func=cmd_sources)
    rendering = sub.add_parser("render")
    rendering.add_argument("cue")
    rendering.add_argument("--seconds", type=float, default=1.2)
    rendering.add_argument("--speed", type=float, default=1.0)
    rendering.add_argument("--brightness", type=float, default=0.35)
    rendering.add_argument("--intensity", type=float, default=0.75)
    rendering.add_argument("--count", type=int, default=1)
    rendering.add_argument("--seed", type=int, default=20260717)
    rendering.add_argument("--stem")
    rendering.add_argument("--out", default=str(DEFAULT_OUT))
    rendering.set_defaults(func=cmd_render)
    packing = sub.add_parser("pack")
    packing.add_argument("pack")
    packing.add_argument("--seed", type=int, default=20260717)
    packing.add_argument("--out", default=str(DEFAULT_OUT))
    packing.set_defaults(func=cmd_pack)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "count", 1) < 1:
        print("count must be at least 1")
        return 2
    return args.func(args)
