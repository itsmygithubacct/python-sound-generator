"""Command line interface for the procedural cyberpunk sound library."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from ..._audio.foley_cli import report_sources
from ..._audio.runtime import artifact_metrics, derive_seed
from ..._audio.validation import bounded_float
from ..._audio.runtime import play as play_audio
from ..._audio.runtime import write_manifest as write_shared_manifest
from . import engine
from .cues import CUES, GROUPS, PACKS, Options, render_with_sources, stem_for


DEFAULT_OUT = Path("output") / "cyberpunk"
DEFAULT_SEED = 20260721
PROVENANCE = Path(__file__).resolve().parent / "sources" / "provenance.json"


def _options(args: argparse.Namespace) -> Options:
    return Options(
        style=engine.style_named(args.style),
        intensity=args.intensity,
        foley=args.foley_mix,
    )


def _validate_options(args: argparse.Namespace) -> Options:
    options = _options(args)
    bounded_float(options.intensity, "--intensity", 0.05, 1.0)
    bounded_float(options.foley, "--foley-mix", 0.0, 1.0)
    return options


def _artifact(
    path: Path,
    root: Path,
    signal: list[float],
    sources: tuple[str, ...],
    cue_name: str,
    seed: int,
    variation: int,
    options: Options,
) -> dict[str, Any]:
    metrics = artifact_metrics(path, signal, engine.SR)
    return {
        "file": path.resolve().relative_to(root.resolve()).as_posix(),
        "cue": cue_name,
        "group": CUES[cue_name].group,
        "seed": seed,
        "variation": variation,
        "options": {
            "style": options.style.name,
            "intensity": round(options.intensity, 4),
            "foley_mix": round(options.foley, 4),
        },
        "sources": list(sources),
        **metrics,
    }


def _write_manifest(
    path: Path,
    artifacts: list[dict[str, Any]],
    *,
    command: str,
    pack: str | None = None,
) -> Path:
    extra: dict[str, Any] = {
        "source_audio": (
            "optional CC0 key contact and servo movement selected by logical path"
        ),
        "source_licensing": "CC0 1.0",
        "source_provenance": (
            "generators/cyberpunk_sound_generator/sources/provenance.json"
        ),
        "synthesis": (
            "deterministic procedural electronic DSP with an optional tiny "
            "hardware-contact Foley layer"
        ),
    }
    if pack is not None:
        extra["pack"] = pack
    return write_shared_manifest(
        path,
        generator="cyberpunk_sound_generator",
        command=command,
        artifacts=artifacts,
        extra=extra,
    )


def _describe(path: Path, signal: list[float], root: Path, sources: int) -> None:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    source_label = f"{sources} Foley" if sources else "procedural"
    print(
        f"  wrote {relative:<42} {len(signal) / engine.SR:.2f}s  {source_label}"
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
            "styles": [
                {"name": style.name, "description": style.description}
                for style in engine.STYLES.values()
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

    selected = any((args.cues, args.styles, args.packs))
    if args.cues or not selected:
        for group in GROUPS:
            print(f"\n{group} cues")
            for cue in CUES.values():
                if cue.group == group:
                    suffix = (
                        f" [Foley: {', '.join(cue.foley_kinds)}]"
                        if cue.foley_kinds else " [procedural]"
                    )
                    print(f"  {cue.name:<20} {cue.description}{suffix}")
    if args.styles or not selected:
        print("\nstyles")
        for style in engine.STYLES.values():
            print(f"  {style.name:<20} {style.description}")
    if args.packs or not selected:
        print("\npacks")
        for pack in PACKS.values():
            total = sum(entry.count for entry in pack.entries)
            print(f"  {pack.name:<20} {pack.description} ({total} files)")
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
        print(
            f"  group {cue.group}; one-shot; {cue.seconds:.2f}s; "
            f"peak {cue.peak_db:.1f} dBFS"
        )
        print(f"  style {options.style.name}: {options.style.description}")
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
        use_key="cyberpunk_use",
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
            _artifact(
                path,
                out_dir,
                signal,
                result.sources,
                args.cue,
                seed,
                index + 1,
                options,
            )
        )
        _describe(path, signal, out_dir, len(result.sources))
        if args.play:
            _play(path)

    manifest = _write_manifest(
        out_dir / "manifest.json", artifacts, command="render"
    )
    print(f"  manifest {manifest}\n")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        raise ValueError(f"unknown pack {args.pack!r}; see `list --packs`")
    options = _validate_options(args)
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name / options.style.name
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
                _artifact(
                    path,
                    out_dir,
                    signal,
                    result.sources,
                    entry.cue,
                    seed,
                    index + 1,
                    options,
                )
            )
            _describe(path, signal, out_dir, len(result.sources))
            if args.play:
                _play(path)

    manifest = _write_manifest(
        out_dir / "manifest.json",
        artifacts,
        command="pack",
        pack=pack.name,
    )
    print(f"  {len(artifacts)} files; manifest {manifest}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen cyberpunk",
        description=(
            "Deterministic cyberpunk interface, network, hacking, cyberware, "
            "and world-feedback sounds."
        ),
        epilog=(
            "examples:\n"
            "  soundgen cyberpunk list\n"
            "  soundgen cyberpunk render scan_ping --style electric\n"
            "  soundgen cyberpunk render emp_blast --style noir --count 3\n"
            "  soundgen cyberpunk pack core --style corporate --play\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="<verb>"
    )

    list_parser = subparsers.add_parser(
        "list", help="show cues, styles, and packs"
    )
    list_parser.add_argument("--cues", action="store_true")
    list_parser.add_argument("--styles", action="store_true")
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
    render_parser.add_argument(
        "--count", type=int, default=1, help="number of deterministic variations"
    )
    render_parser.add_argument("--stem", help="override the output file stem")
    render_parser.add_argument(
        "--play", action="store_true", help="play each rendered variation in order"
    )
    render_parser.set_defaults(func=cmd_render)

    pack_parser = subparsers.add_parser(
        "pack", help="render an organised sound bank"
    )
    pack_parser.add_argument("pack", help="pack name; see `list --packs`")
    pack_parser.add_argument(
        "--play", action="store_true", help="play every generated cue sequentially"
    )
    pack_parser.set_defaults(func=cmd_pack)

    for command_parser in (info_parser, render_parser, pack_parser):
        command_parser.add_argument(
            "--style",
            default="electric",
            help=f"sound style: {', '.join(engine.STYLES)}",
        )
        command_parser.add_argument(
            "--intensity",
            type=float,
            default=0.85,
            help="0.05..1, weight and density of supporting layers",
        )
        command_parser.add_argument(
            "--foley-mix",
            type=float,
            default=0.0,
            help="0..1, optional contribution of recorded hardware contact",
        )
    for command_parser in (render_parser, pack_parser):
        command_parser.add_argument(
            "--seed",
            type=int,
            default=DEFAULT_SEED,
            help="base seed; identical options produce identical PCM",
        )
        command_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (OSError, ValueError, KeyError) as error:
        print(f"soundgen cyberpunk: {error}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
