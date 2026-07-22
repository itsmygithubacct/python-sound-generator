"""Command line interface for the procedural RPG sound library."""

from __future__ import annotations

from ..._audio.runtime import artifact_metrics

from ..._audio.runtime import derive_seed
from ..._audio.runtime import play as play_audio

from ..._audio.runtime import write_manifest as write_shared_manifest

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from ..._audio.foley_cli import report_sources
from ..._audio.validation import bounded_float
from . import engine
from .cues import CUES, GROUPS, PACKS, Options, render_with_sources, stem_for


DEFAULT_OUT = Path("output") / "rpg"
DEFAULT_SEED = 20260721
PROVENANCE = Path(__file__).resolve().parent / "sources" / "provenance.json"


def _options(args: argparse.Namespace) -> Options:
    return Options(
        theme=engine.theme_named(args.theme),
        intensity=args.intensity,
        foley=args.foley_mix,
    )


def _validate_options(args: argparse.Namespace) -> Options:
    options = _options(args)
    bounded_float(options.intensity, "--intensity", 0.05, 1.0)
    bounded_float(options.foley, "--foley-mix", 0.0, 1.0)
    return options


def _artifact(path: Path, root: Path, signal: list[float],
              sources: tuple[str, ...], cue_name: str,
              seed: int, variation: int,
              options: Options) -> dict[str, Any]:
    metrics = artifact_metrics(path, signal, engine.SR)
    return {
        "file": path.resolve().relative_to(root.resolve()).as_posix(),
        "cue": cue_name,
        "group": CUES[cue_name].group,
        "seed": seed,
        "variation": variation,
        "options": {
            "theme": options.theme.name,
            "intensity": round(options.intensity, 4),
            "foley_mix": round(options.foley, 4),
        },
        "sources": list(sources),
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
    }


def _write_manifest(path: Path, artifacts: list[dict[str, Any]], *,
                    command: str, pack: str | None = None) -> Path:
    extra: dict[str, Any] = {
        "source_audio": (
            "optional CC0 physical layers from provider-local excerpts and "
            "approved recordings reused by logical path"
        ),
        "source_licensing": "CC0 1.0",
        "source_provenance": (
            "generators/rpg_sound_generator/sources/provenance.json"
        ),
        "synthesis": (
            "deterministic procedural RPG gestures with an optional Foley mix"
        ),
    }
    if pack is not None:
        extra["pack"] = pack
    return write_shared_manifest(
        path,
        generator="rpg_sound_generator",
        command=command,
        artifacts=artifacts,
        extra=extra,
    )


def _describe(path: Path, signal: list[float], root: Path, sources: int) -> None:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    source_label = f"{sources} Foley" if sources else "procedural"
    print(
        f"  wrote {relative:<38} {len(signal) / engine.SR:.2f}s  {source_label}"
    )


def _play(path: Path) -> None:
    if not play_audio(path):
        raise OSError(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)"
        )


def cmd_list(args: argparse.Namespace) -> int:
    if args.json:
        payload = {
            "cues": [
                {
                    "name": cue.name,
                    "group": cue.group,
                    "description": cue.description,
                    "duration_seconds": cue.seconds,
                    "layers": list(cue.layers),
                    "foley": list(cue.foley_kinds),
                }
                for cue in CUES.values()
            ],
            "themes": [
                {"name": theme.name, "description": theme.description}
                for theme in engine.THEMES.values()
            ],
            "packs": [
                {
                    "name": pack.name,
                    "description": pack.description,
                    "files": sum(entry.count for entry in pack.entries),
                }
                for pack in PACKS.values()
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    selected = any((args.cues, args.themes, args.packs))
    if args.cues or not selected:
        for group in GROUPS:
            print(f"\n{group} cues")
            for cue in CUES.values():
                if cue.group == group:
                    suffix = (
                        f" [Foley: {', '.join(cue.foley_kinds)}]"
                        if cue.foley_kinds else " [procedural]"
                    )
                    print(f"  {cue.name:<18} {cue.description}{suffix}")
    if args.themes or not selected:
        print("\nthemes")
        for theme in engine.THEMES.values():
            print(f"  {theme.name:<18} {theme.description}")
    if args.packs or not selected:
        print("\npacks")
        for pack in PACKS.values():
            total = sum(entry.count for entry in pack.entries)
            print(f"  {pack.name:<18} {pack.description} ({total} files)")
    print()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    if args.cue is not None and args.cue not in CUES:
        raise ValueError(f"unknown cue {args.cue!r}; see `list --cues`")
    options = _validate_options(args)
    names = [args.cue] if args.cue else list(CUES)
    for name in names:
        cue = CUES[name]
        print(f"\n{cue.name}: {cue.description}")
        print(f"  group {cue.group}; one-shot; {cue.seconds:.2f}s; "
              f"peak {cue.peak_db:.1f} dBFS")
        print(f"  theme {options.theme.name}: {options.theme.description}")
        print(f"  layers: {'; '.join(cue.layers)}")
        print(
            "  Foley: "
            + (", ".join(cue.foley_kinds) if cue.foley_kinds else "none")
        )
    print()
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    from . import foley

    return report_sources(
        PROVENANCE,
        source_inventory=foley.source_inventory,
        missing_sources=foley.missing_sources,
        use_key="rpg_use",
        json_output=args.json,
    )


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        raise ValueError(f"unknown cue {args.cue!r}; see `list --cues`")
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    options = _validate_options(args)
    out_dir = Path(args.out)
    stem = args.stem or stem_for(args.cue, options)
    artifacts: list[dict[str, Any]] = []

    print(f"{args.cue}: {CUES[args.cue].description}")
    for index in range(args.count):
        seed = derive_seed(
            args.seed,
            provider=__package__,
            cue=args.cue,
            filename=stem,
            variation=index + 1,
        )
        result = render_with_sources(args.cue, options, random.Random(seed))
        signal = result.samples
        suffix = "" if index == 0 else f"_v{index + 1:02d}"
        path = engine.write_wav(out_dir / f"{stem}{suffix}.wav", signal)
        artifacts.append(
            _artifact(path, out_dir, signal, result.sources, args.cue,
                      seed, index + 1,
                      options)
        )
        _describe(path, signal, out_dir, len(result.sources))
        if args.play:
            _play(path)

    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="render")
    print(f"  manifest {manifest}\n")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        raise ValueError(f"unknown pack {args.pack!r}; see `list --packs`")
    options = _validate_options(args)
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name / options.theme.name
    artifacts: list[dict[str, Any]] = []
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")
    for entry in pack.entries:
        for index in range(entry.count):
            seed = derive_seed(
                args.seed,
                provider=__package__,
                cue=entry.cue,
                filename=entry.filename,
                variation=index + 1,
            )
            result = render_with_sources(
                entry.cue, options, random.Random(seed)
            )
            signal = result.samples
            path = engine.write_wav(
                out_dir / pack.filename(entry, index), signal
            )
            artifacts.append(
                _artifact(path, out_dir, signal, result.sources, entry.cue,
                          seed, index + 1,
                          options)
            )
            _describe(path, signal, out_dir, len(result.sources))
            if args.play:
                _play(path)

    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="pack", pack=pack.name)
    print(f"  {len(artifacts)} files; manifest {manifest}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen rpg",
        description=(
            "Deterministic RPG interface and progression sounds: inventory, "
            "loot, quests, dialogue, saving, and rest."
        ),
        epilog=(
            "examples:\n"
            "  soundgen rpg list\n"
            "  soundgen rpg render coin_pickup --theme fantasy --count 3\n"
            "  soundgen rpg render encounter --theme dark\n"
            "  soundgen rpg pack core --theme retro\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="<verb>"
    )

    list_parser = subparsers.add_parser(
        "list", help="show cues, themes, and packs"
    )
    list_parser.add_argument("--cues", action="store_true")
    list_parser.add_argument("--themes", action="store_true")
    list_parser.add_argument("--packs", action="store_true")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=cmd_list)

    info_parser = subparsers.add_parser(
        "info", help="describe cue layers and duration"
    )
    info_parser.add_argument("cue", nargs="?", help="one cue; default is all")
    info_parser.set_defaults(func=cmd_info)

    sources_parser = subparsers.add_parser(
        "sources", help="show optional CC0 recordings and provenance"
    )
    sources_parser.add_argument("--json", action="store_true")
    sources_parser.set_defaults(func=cmd_sources)

    render_parser = subparsers.add_parser("render", help="render one cue")
    render_parser.add_argument("cue", help="cue name; see `list --cues`")
    render_parser.add_argument("--count", type=int, default=1,
                               help="number of deterministic variations")
    render_parser.add_argument("--stem", help="override the output file stem")
    render_parser.add_argument(
        "--play",
        action="store_true",
        help="play each rendered variation in order",
    )
    render_parser.set_defaults(func=cmd_render)

    pack_parser = subparsers.add_parser(
        "pack", help="render an organised sound bank"
    )
    pack_parser.add_argument("pack", help="pack name; see `list --packs`")
    pack_parser.add_argument(
        "--play",
        action="store_true",
        help="play every generated cue sequentially",
    )
    pack_parser.set_defaults(func=cmd_pack)

    for command_parser in (info_parser, render_parser, pack_parser):
        command_parser.add_argument(
            "--theme",
            default="fantasy",
            help=f"sound theme: {', '.join(engine.THEMES)}",
        )
        command_parser.add_argument(
            "--intensity",
            type=float,
            default=0.85,
            help="0.05..1, density and weight of supporting layers",
        )
        command_parser.add_argument(
            "--foley-mix",
            type=float,
            default=0.0,
            help="0..1, optional contribution of recorded physical layers",
        )
    for command_parser in (render_parser, pack_parser):
        command_parser.add_argument(
            "--seed", type=int, default=DEFAULT_SEED,
            help="base seed; identical options produce identical PCM",
        )
        command_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (OSError, ValueError, KeyError) as error:
        print(f"soundgen rpg: {error}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
