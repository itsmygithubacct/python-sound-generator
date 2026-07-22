"""Registry-driven command line for fantasy magic sound effects."""

from __future__ import annotations

from ..._audio.runtime import artifact_metrics

from ..._audio.runtime import derive_seed

from ..._audio.runtime import write_manifest as write_shared_manifest

import argparse
import random
import sys
from pathlib import Path
from typing import Any

from . import engine
from .cues import CUES, PACKS, Options, render

DEFAULT_OUT = Path("output") / "magic-energy"
DEFAULT_SEED = 20260716


def _artifact(path: Path, root: Path, signal: list[float], cue: str,
              seed: int, variation: int, options: Options) -> dict[str, Any]:
    metrics = artifact_metrics(path, signal, engine.SR)
    return {
        "file": path.relative_to(root).as_posix(),
        "cue": cue,
        "seed": seed,
        "variation": variation,
        "options": {"intensity": options.intensity, "duration": options.duration},
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        "sources": [],
    }


def _manifest(path: Path, artifacts: list[dict[str, Any]], *, command: str,
              pack: str | None = None) -> Path:
    payload: dict[str, Any] = {
        "generator": "magic_energy_generator",
        "command": command,
        "source_audio": "none; deterministic procedural DSP",
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
    if args.packs:
        for pack in PACKS.values():
            print(f"{pack.name:<14} {pack.description}")
    else:
        for cue in CUES.values():
            print(f"{cue.name:<14} {cue.description}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list`", file=sys.stderr)
        return 2
    out = Path(args.out)
    options = Options(args.intensity, args.duration)
    artifacts: list[dict[str, Any]] = []
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        result = render(args.cue, options, random.Random(seed))
        suffix = "" if index == 0 else f"_v{index + 1:02d}"
        path = engine.write_wav(out / f"{args.cue}{suffix}.wav", result.samples)
        artifacts.append(_artifact(
            path, out.resolve(), result.samples, args.cue, seed, index + 1, options
        ))
        print(f"  wrote {path}  {len(result.samples) / engine.SR:.3f}s")
    manifest = _manifest(out / "manifest.json", artifacts, command="render")
    print(f"  manifest {manifest}")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`", file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out = Path(args.out) / pack.name
    artifacts: list[dict[str, Any]] = []
    print(f"pack {pack.name}: {pack.description} -> {out}")
    for entry in pack.entries:
        for index in range(entry.count):
            seed = derive_seed(args.seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            result = render(entry.cue, entry.options, random.Random(seed))
            path = engine.write_wav(out / pack.filename(entry, index), result.samples)
            artifacts.append(_artifact(
                path, out.resolve(), result.samples, entry.cue, seed,
                index + 1, entry.options,
            ))
            print(f"  wrote {path.relative_to(out.resolve())}")
    manifest = _manifest(
        out / "manifest.json", artifacts, command="pack", pack=pack.name
    )
    print(f"  {len(artifacts)} files; manifest {manifest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen magic-energy",
        description="Fantasy magic gestures made with deterministic DSP.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list cues or packs")
    listing.add_argument("--packs", action="store_true")
    listing.set_defaults(func=cmd_list)
    one = sub.add_parser("render", help="render one cue")
    one.add_argument("cue")
    one.add_argument("--count", type=int, default=1)
    one.add_argument("--intensity", type=float, default=0.9)
    one.add_argument("--duration", type=float)
    one.set_defaults(func=cmd_render)
    pack = sub.add_parser("pack", help="render a named game pack")
    pack.add_argument("pack")
    pack.set_defaults(func=cmd_pack)
    for target in (one, pack):
        target.add_argument("--seed", type=int, default=DEFAULT_SEED)
        target.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "count", 1) < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    try:
        return args.func(args) or 0
    except (OSError, ValueError) as error:
        print(f"soundgen magic-energy: {error}", file=sys.stderr)
        return 1
