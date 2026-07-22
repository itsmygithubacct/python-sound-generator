"""Command line for the water generator."""

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
# either being wrong. Items 9 and 13 import it for the same reason.
from ..._audio import loops as loopdsp

from . import engine
from .cues import (
    CUES,
    GROUPS,
    MAX_SECONDS,
    MIN_SECONDS,
    NOT_MADE,
    PACKS,
    Options,
    Pack,
    duration_for,
    render,
)

DEFAULT_OUT = Path("output") / "water"
PROVENANCE = engine.LOCAL_SOURCES / "provenance.json"


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _options(args: argparse.Namespace) -> Options:
    return Options(mass=engine.mass_named(args.mass),
                   intensity=args.intensity,
                   seconds=getattr(args, "seconds", None))


def _artifact(path: Path, signal: list[float], sources: tuple[str, ...],
              cue_name: str, seed: int, variation: int,
              options: Options) -> dict[str, Any]:
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
        "sources": list(sources),
        "options": {
            "mass": options.mass.name,
            "intensity": round(options.intensity, 4),
        },
    }
    if CUES[cue_name].loop:
        seam = loopdsp.seam(signal)
        entry["loop"] = {
            "seamless": seam.is_seamless(),
            "step_ratio": round(seam.step_ratio, 4),
            "level_ratio": round(seam.level_ratio, 4),
        }
        entry["options"]["seconds"] = round(duration_for(cue_name, options), 4)
    return entry


def _write_manifest(path: Path, artifacts: list[dict[str, Any]], *,
                    command: str, pack: str | None = None) -> Path:
    payload: dict[str, Any] = {
        "generator": "water_sounds_generator",
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
    show_all = not (args.cues or args.masses or args.packs)
    if args.cues or show_all:
        for group in GROUPS:
            print(f"\nCues: {group}")
            for cue in CUES.values():
                if cue.group == group:
                    print(f"  {cue.name:<10} {cue.description}")
                    if cue.honours:
                        print(f"  {'':<10} honours: {cue.honours}")
                    if cue.replaces:
                        print(f"  {'':<10} replaces: {cue.replaces}")
        print("\nNot made here")
        for name, why in NOT_MADE.items():
            print(f"  {name:<12} {why}")
    if args.masses or show_all:
        print("\nMasses (which recording, then how fast)")
        for mass in engine.MASSES.values():
            print(f"  {mass.name:<10} {mass.description}")
    if args.packs or show_all:
        print("\nPacks")
        for pack in PACKS.values():
            made = sum(e.count for e in pack.entries)
            print(f"  {pack.name:<16} {pack.description}  ({made} files)")
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
            where = f.get("bundled_in", "here")
            print(f"      {f['file']:<34} <- {f['original']:<20} [{where}]")
    for key in ("provenance_note", "coverage_note"):
        if data.get(key):
            print(f"\n{key.replace('_', ' ').capitalize()}:")
            print(f"  {data[key]}")
    if args.excluded:
        print("\nConsidered and not used:")
        for item in data.get("excluded", []):
            print(f"\n  {item['name']}")
            print(f"    {item['source_page']}")
            print(f"    {item['reason']}")
    print()
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        hint = f"  ({NOT_MADE[args.cue]})" if args.cue in NOT_MADE else ""
        print(f"unknown cue {args.cue!r}; see `list --cues`{hint}",
              file=sys.stderr)
        return 2
    options = _options(args)
    out_dir = Path(args.out)
    artifacts: list[dict[str, Any]] = []
    print(f"{args.cue}: {CUES[args.cue].description}")
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        result = render(args.cue, options, np.random.default_rng(seed))
        stem = f"{args.cue}_{options.mass.name}"
        name = f"{stem}.wav" if args.count == 1 else f"{stem}_{index + 1:02d}.wav"
        path = engine.write_wav(out_dir / name, result.samples)
        artifact = _artifact(path, result.samples, result.sources, args.cue,
                             seed, index + 1, options)
        artifacts.append(artifact)
        detail = ""
        if "loop" in artifact:
            loop = artifact["loop"]
            detail = (f"   [{'seamless' if loop['seamless'] else 'SEAM FAILS'}"
                      f" step {loop['step_ratio']:.3f}]")
        print(f"  wrote {path}  {len(result.samples) / engine.SR:.2f}s{detail}")
        if args.play:
            _play(path)
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="render")
    print(f"  manifest {manifest}\n")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Measure every loop's seam rather than trusting the crossfade.

    These loops are crossfaded rather than wrapped, because their beds are
    continuous recordings that cannot be cut anywhere -- the fire_sound_generator
    case, not the stone_noise_generator one. The beds are also *shorter* than most
    of the loops built from them, so the material is tiled first, which makes
    measuring the join the whole point rather than a formality.
    """
    print(f"\n{'cue':12s} {'mass':10s} {'step':>8s} {'level':>8s}")
    failures = 0
    for name, cue in CUES.items():
        if not cue.loop:
            continue
        for mass in engine.MASSES:
            options = Options(mass=engine.MASSES[mass], seconds=args.seconds)
            result = render(name, options, np.random.default_rng(args.seed))
            seam = loopdsp.seam(result.samples)
            ok = seam.is_seamless()
            failures += 0 if ok else 1
            print(f"{name:12s} {mass:10s} {seam.step_ratio:8.3f} "
                  f"{seam.level_ratio:8.3f}  {'ok' if ok else 'FAILS'}")
    print(f"\n(step <= {loopdsp.MAX_STEP_RATIO:g}, level within "
          f"{loopdsp.MAX_LEVEL_DRIFT:g}x)")
    print(f"{failures} failing\n")
    return 1 if failures else 0


def _render_pack(pack: Pack, out_dir: Path, seed: int,
                 play: bool) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for entry in pack.entries:
        for index in range(entry.count):
            file_seed = derive_seed(seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            result = render(entry.cue, entry.options,
                            np.random.default_rng(file_seed))
            path = engine.write_wav(out_dir / pack.filename(entry, index),
                                    result.samples)
            artifact = _artifact(path, result.samples, result.sources,
                                 entry.cue, file_seed, index + 1, entry.options)
            artifacts.append(artifact)
            detail = ""
            if "loop" in artifact:
                detail = ("   [seamless]" if artifact["loop"]["seamless"]
                          else "   [SEAM FAILS]")
            print(f"  wrote {path.name}  "
                  f"{len(result.samples) / engine.SR:.2f}s{detail}")
            if play:
                _play(path)
    return artifacts


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`",
              file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")
    artifacts = _render_pack(pack, out_dir, args.seed, args.play)
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="pack", pack=pack.name)
    print(f"  {len(artifacts)} files; manifest {manifest}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen water",
        description="Water -- entries, drips, bubbles, rain, tanks and wakes -- "
                    "built from CC0 recordings.",
        epilog="examples:\n"
               "  soundgen water list\n"
               "  soundgen water render splash --mass boat "
               "--count 5 --play\n"
               "  soundgen water render aquarium --seconds 60\n"
               "  soundgen water pack replacements\n"
               "  soundgen water verify\n"
               "  soundgen water sources --excluded\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show cues, masses, and packs")
    p_list.add_argument("--cues", action="store_true")
    p_list.add_argument("--masses", action="store_true")
    p_list.add_argument("--packs", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_sources = sub.add_parser("sources",
                               help="show the recordings and licences")
    p_sources.add_argument("--excluded", action="store_true",
                           help="also show what was considered and rejected")
    p_sources.set_defaults(func=cmd_sources)

    p_render = sub.add_parser("render", help="render one cue")
    p_render.add_argument("cue", help="cue name; see `list --cues`")
    p_render.add_argument("--count", type=int, default=1,
                          help="variations, each on its own seed")
    p_render.add_argument("--seconds", type=float, default=None,
                          help=f"loop length, {MIN_SECONDS:g}-{MAX_SECONDS:g}s "
                               f"(loops only)")
    p_render.set_defaults(func=cmd_render)

    p_pack = sub.add_parser("pack", help="render a named artifact bank")
    p_pack.add_argument("pack", help="pack name; see `list --packs`")
    p_pack.set_defaults(func=cmd_pack)

    p_verify = sub.add_parser("verify", help="measure every loop's seam")
    p_verify.add_argument("--seconds", type=float, default=8.0)
    p_verify.set_defaults(func=cmd_verify)

    for p in (p_render, p_pack, p_verify):
        p.add_argument("--seed", type=int, default=20260715,
                       help="base seed; renders are reproducible")
    for p in (p_render, p_pack):
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--play", action="store_true",
                       help="play each file after writing it")
    for p in (p_render,):
        p.add_argument("--mass", default="body",
                       help=f"one of: {', '.join(engine.MASSES)}")
        p.add_argument("--intensity", type=float, default=0.8,
                       help="0..1, how much of it")
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
