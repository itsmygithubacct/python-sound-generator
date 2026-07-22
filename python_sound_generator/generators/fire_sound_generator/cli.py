"""Command line for the fire generator."""

from __future__ import annotations

from ..._audio.runtime import artifact_metrics

from ..._audio.runtime import derive_seed

from ..._audio.runtime import play as play_audio

from ..._audio.runtime import write_manifest as write_shared_manifest

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# The seam metric is imported rather than reimplemented: environment_ambience
# _generator already worked out how to tell a loop from a click and a swell, and
# two generators disagreeing about what "seamless" means would be worse than
# either answer. That package is standard-library-only, so this costs nothing.
from ..._audio import loops as loopdsp

from . import engine
from .cues import CUES, GROUPS, PACKS, Options, render

DEFAULT_OUT = Path("output") / "fire"
PROVENANCE = engine.SOURCE_DIRECTORY / "provenance.json"


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _options(args: argparse.Namespace) -> Options:
    return Options(scale=engine.scale_named(args.scale),
                   intensity=args.intensity,
                   seconds=getattr(args, "seconds", None))


def _seam_of(samples: list[float]) -> loopdsp.Seam:
    return loopdsp.seam(samples)


def _artifact(path: Path, signal: list[float], sources: tuple[str, ...],
              cue_name: str, seed: int, variation: int,
              options: Options) -> dict[str, Any]:
    metrics = artifact_metrics(path, signal, engine.SR)
    entry: dict[str, Any] = {
        "file": path.name,
        "cue": cue_name,
        "seed": seed,
        "variation": variation,
        "options": {"scale": options.scale.name,
                    "intensity": round(options.intensity, 4)},
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        "sources": list(sources),
    }
    if CUES[cue_name].loop:
        seam = _seam_of(signal)
        entry["loop"] = {
            "seamless": seam.is_seamless(),
            "step_ratio": round(seam.step_ratio, 4),
            "level_ratio": round(seam.level_ratio, 4),
        }
    return entry


def _write_manifest(path: Path, artifacts: list[dict[str, Any]], *,
                    command: str, pack: str | None = None) -> Path:
    payload: dict[str, Any] = {
        "generator": "fire_sound_generator",
        "command": command,
        "source_audio": "CC0 recordings; see sources/provenance.json",
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
    show_all = not (args.cues or args.scales or args.packs)
    if args.cues or show_all:
        for group in GROUPS:
            print(f"\nCues: {group}")
            for cue in CUES.values():
                if cue.group == group:
                    print(f"  {cue.name:<15} {cue.description}")
                    if cue.honours:
                        print(f"  {'':<15} honours: {cue.honours}")
    if args.scales or show_all:
        print("\nScales")
        for s in engine.SCALES.values():
            print(f"  {s.name:<15} {s.description}")
    if args.packs or show_all:
        print("\nPacks")
        for pack in PACKS.values():
            total = sum(e.count for e in pack.entries)
            print(f"  {pack.name:<15} {pack.description}  ({total} files)")
    print()
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    data = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    print(f"\nPolicy: {data['source_audio_policy']}")
    print(f"Retrieved: {data['retrieved_utc']}")
    for collection in data["collections"]:
        print(f"\n  {collection['name']} - {collection['license']}")
        print(f"    author:  {collection['author']}")
        print(f"    page:    {collection['source_page']}")
        for f in collection["files"]:
            print(f"      {f['file']}  <- {f['original']}")
    if args.excluded:
        print("\nConsidered and not used:")
        for item in data["excluded"]:
            print(f"\n  {item['name']}")
            print(f"    {item['source_page']}")
            print(f"    {item['reason']}")
    print()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Measure every loop's seam rather than trusting the crossfade."""

    print(f"\n{'cue':16s} {'scale':10s} {'step':>8s} {'level':>8s}  result")
    worst = 0
    for name, cue in CUES.items():
        if not cue.loop:
            continue
        for scale in engine.SCALES:
            options = Options(scale=engine.SCALES[scale], seconds=args.seconds)
            result = render(name, options, np.random.default_rng(args.seed))
            seam = _seam_of(result.samples)
            ok = seam.is_seamless()
            worst = max(worst, 0 if ok else 1)
            print(f"{name:16s} {scale:10s} {seam.step_ratio:8.3f} "
                  f"{seam.level_ratio:8.3f}  {'ok' if ok else 'FAILS'}")
    print(f"\n(step <= {loopdsp.MAX_STEP_RATIO:g}, level within "
          f"{loopdsp.MAX_LEVEL_DRIFT:g}x)\n")
    return worst


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    options = _options(args)
    out_dir = Path(args.out)
    artifacts: list[dict[str, Any]] = []
    print(f"{args.cue}: {CUES[args.cue].description}")
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        result = render(args.cue, options, np.random.default_rng(seed))
        stem = f"{args.cue}_{options.scale.name}"
        name = f"{stem}.wav" if args.count == 1 else f"{stem}_{index + 1:02d}.wav"
        path = engine.write_wav(out_dir / name, result.samples)
        artifact = _artifact(path, result.samples, result.sources, args.cue,
                             seed, index + 1, options)
        artifacts.append(artifact)
        detail = f"  wrote {path}  {len(result.samples) / engine.SR:.2f}s"
        if "loop" in artifact:
            loop = artifact["loop"]
            detail += (f"  [{'seamless' if loop['seamless'] else 'SEAM FAILS'}"
                       f" step {loop['step_ratio']:.3f}"
                       f" level {loop['level_ratio']:.2f}]")
        print(detail)
        if args.play:
            _play(path)
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="render")
    print(f"  manifest {manifest}\n")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`",
              file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")
    artifacts: list[dict[str, Any]] = []
    for entry in pack.entries:
        for index in range(entry.count):
            seed = derive_seed(args.seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            result = render(entry.cue, entry.options,
                            np.random.default_rng(seed))
            path = engine.write_wav(out_dir / pack.filename(entry, index),
                                    result.samples)
            artifacts.append(_artifact(path, result.samples, result.sources,
                                       entry.cue, seed, index + 1,
                                       entry.options))
            print(f"  wrote {path.name}  "
                  f"{len(result.samples) / engine.SR:.2f}s")
            if args.play:
                _play(path)
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="pack", pack=pack.name)
    print(f"  {len(artifacts)} files; manifest {manifest}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen fire",
        description="Fire sound effects built from bundled CC0 recordings.",
        epilog="examples:\n"
               "  soundgen fire list\n"
               "  soundgen fire render burning "
               "--scale bonfire --seconds 20 --play\n"
               "  soundgen fire render fireball --count 4\n"
               "  soundgen fire verify\n"
               "  soundgen fire sources --excluded\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show cues, scales, and packs")
    p_list.add_argument("--cues", action="store_true")
    p_list.add_argument("--scales", action="store_true")
    p_list.add_argument("--packs", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_sources = sub.add_parser("sources",
                               help="show the bundled recordings and licences")
    p_sources.add_argument("--excluded", action="store_true",
                           help="also show what was considered and rejected")
    p_sources.set_defaults(func=cmd_sources)

    p_verify = sub.add_parser("verify", help="measure every loop's seam")
    p_verify.add_argument("--seconds", type=float, default=8.0)
    p_verify.add_argument("--seed", type=int, default=20260715)
    p_verify.set_defaults(func=cmd_verify)

    p_render = sub.add_parser("render", help="render one cue")
    p_render.add_argument("cue", help="cue name; see `list --cues`")
    p_render.add_argument("--count", type=int, default=1,
                          help="variations, each on its own seed")
    p_render.add_argument("--seconds", type=float, default=None,
                          help="loop length; loops only")
    p_render.set_defaults(func=cmd_render)

    p_pack = sub.add_parser("pack", help="render a named artifact bank")
    p_pack.add_argument("pack", help="pack name; see `list --packs`")
    p_pack.set_defaults(func=cmd_pack)

    for p in (p_render, p_pack):
        p.add_argument("--seed", type=int, default=20260715,
                       help="base seed; renders are reproducible")
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--play", action="store_true",
                       help="play each file after writing it")
    for p in (p_render,):
        p.add_argument("--scale", default="campfire",
                       help=f"one of: {', '.join(engine.SCALES)}")
        p.add_argument("--intensity", type=float, default=0.9,
                       help="0..1, how hard the event lands")
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
