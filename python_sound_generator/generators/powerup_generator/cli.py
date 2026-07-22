"""Command line for the powerup generator: list / info / render / pack."""

from __future__ import annotations

from ..._audio.runtime import artifact_metrics

from ..._audio.runtime import derive_seed

from ..._audio.runtime import play as play_audio

from ..._audio.runtime import write_manifest as write_shared_manifest

import argparse
import sys
from pathlib import Path

import numpy as np

from . import engine as e
from .cues import CUES, GROUPS, PACKS, render
from .styles import DEFAULT_STYLE, STYLES, get_style

DEFAULT_OUT = Path("output") / "powerups"
DEFAULT_SEED = 20260714


# --------------------------------------------------------------------------
# display
# --------------------------------------------------------------------------

def _tty() -> bool:
    return sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _tty() else text


def _header(text: str) -> None:
    print(_c(text, "1;36"))


def _item(name: str, detail: str) -> None:
    print(f"  {_c(name.ljust(18), '1')} {detail}")


# --------------------------------------------------------------------------
# verbs
# --------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    show_all = not (args.cues or args.styles or args.packs)
    if args.cues or show_all:
        for group in GROUPS:
            _header(f"{group} cues")
            for cue in CUES.values():
                if cue.group == group:
                    _item(cue.name, cue.description)
        print()
    if args.styles or show_all:
        _header("styles")
        for name, style in STYLES.items():
            marker = " (default)" if name == DEFAULT_STYLE else ""
            _item(name, f"{style.description}{marker}")
        print()
    if args.packs or show_all:
        _header("packs")
        for name, entries in PACKS.items():
            _item(name, f"{len(entries)} cues")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    style = get_style(args.style)
    _header(f"style: {style.label}")
    print(f"  {style.description}")
    print(f"  note scale {style.note_scale}  gap scale {style.gap_scale}  "
          f"tail scale {style.tail_scale}")
    print(f"  body {style.body_gain}  accent {style.accent_gain}  "
          f"sparkle {style.sparkle_gain} x{style.sparkle_count}  "
          f"drive {style.drive}")
    if args.cue:
        cue = CUES.get(args.cue)
        if cue is None:
            print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
            return 2
        print()
        _header(f"cue: {cue.name}")
        print(f"  group     {cue.group}")
        print(f"  {cue.description}")
        print(f"  peak      {cue.peak_db} dBFS")
        print(f"  honours   {cue.honours}")
        rng = np.random.default_rng(args.seed)
        signal = render(cue, style, rng)
        print(f"  renders   {len(signal) / e.SR:.2f}s in style {style.label}")
    return 0


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _render_one(cue_name: str, style_name: str, seed: int, out_dir: Path,
                index: int, total: int,
                stem: str | None = None) -> tuple[Path, np.ndarray]:
    cue = CUES[cue_name]
    style = get_style(style_name)
    # One rng per file, derived from the seed and the variation index, so a
    # render is reproducible on its own rather than depending on what was
    # rendered before it.
    file_seed = derive_seed(
        seed,
        provider=__package__,
        cue=cue_name,
        filename=stem or cue_name,
        variation=index + 1,
    )
    rng = np.random.default_rng(file_seed)
    signal = render(cue, style, rng)
    # A pack passes the exact stem the game loads; `render` has no game to
    # answer to and numbers its own variations from _v01.
    name = stem
    if name is None:
        name = cue_name
        if total > 1:
            name = f"{name}_v{index + 1:02d}"
    path = out_dir / f"{name}.wav"
    e.write_wav(path, signal)
    print(f"  wrote {path}  {len(signal) / e.SR:.2f}s")
    return path, signal


def _artifact(path: Path, signal: np.ndarray, cue_name: str, style_name: str,
              seed: int, variation: int) -> dict[str, object]:
    metrics = artifact_metrics(path, signal, e.SR)
    return {
        "file": path.name,
        "cue": cue_name,
        "style": style_name,
        "seed": seed,
        "variation": variation,
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        "sources": [],
    }


def _write_pack_manifest(path: Path, pack: str,
                         artifacts: list[dict[str, object]]) -> Path:
    payload = {
        "generator": "powerup_generator",
        "command": "pack",
        "pack": pack,
        "source_audio": "none; deterministic procedural synthesis",
        "artifacts": artifacts,
    }
    return write_shared_manifest(
        path,
        generator=payload.pop("generator"),
        command=payload.pop("command"),
        artifacts=payload.pop("artifacts"),
        extra=payload,
    )


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    if args.style not in STYLES:
        print(f"unknown style {args.style!r}; see `list --styles`",
              file=sys.stderr)
        return 2
    if args.count < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    last = None
    for i in range(args.count):
        last, _ = _render_one(args.cue, args.style, args.seed, out_dir, i,
                              args.count)
    if args.play and last:
        _play(last)
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`", file=sys.stderr)
        return 2
    if args.style not in STYLES:
        print(f"unknown style {args.style!r}; see `list --styles`",
              file=sys.stderr)
        return 2
    out_dir = Path(args.out) / args.pack / args.style
    _header(f"pack {args.pack} in style {args.style} -> {out_dir}")
    artifacts: list[dict[str, object]] = []
    for entry in PACKS[args.pack]:
        for i in range(entry.count):
            path, signal = _render_one(
                entry.cue, args.style, args.seed, out_dir, i,
                entry.count, entry.filename(i),
            )
            artifacts.append(_artifact(
                path,
                signal,
                entry.cue,
                args.style,
                derive_seed(
                    args.seed,
                    provider=__package__,
                    cue=entry.cue,
                    filename=entry.filename(i),
                    variation=i + 1,
                ),
                i + 1,
            ))
    manifest = _write_pack_manifest(
        out_dir / "manifest.json", args.pack, artifacts
    )
    print(f"  {len(artifacts)} files; manifest {manifest}")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen powerups",
        description="Generate powerup and item-acquisition SFX with "
                    "procedural DSP. No samples.",
    )
    sub = parser.add_subparsers(dest="_verb", required=True, metavar="<verb>")

    p_list = sub.add_parser("list", help="show cues, styles, packs")
    p_list.add_argument("--cues", action="store_true")
    p_list.add_argument("--styles", action="store_true")
    p_list.add_argument("--packs", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="show a style's tuning, or a cue")
    p_info.add_argument("--cue", help="also describe and time this cue")
    p_info.set_defaults(func=cmd_info)

    p_render = sub.add_parser("render", help="render one cue")
    p_render.add_argument("cue", help="cue name; see `list --cues`")
    p_render.add_argument("--count", type=int, default=1,
                          help="variations, each on its own seed")
    p_render.add_argument("--play", action="store_true",
                          help="play the last render")
    p_render.set_defaults(func=cmd_render)

    p_pack = sub.add_parser("pack", help="render a named set of cues")
    p_pack.add_argument("pack", help="pack name; see `list --packs`")
    p_pack.set_defaults(func=cmd_pack)

    for p in (p_info, p_render, p_pack):
        p.add_argument("--style", default=DEFAULT_STYLE, choices=list(STYLES),
                       help=f"timbre house (default: {DEFAULT_STYLE})")
        p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help="base rng seed; renders are reproducible")
    for p in (p_render, p_pack):
        p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                       help=f"output directory (default: {DEFAULT_OUT})")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except (OSError, ValueError) as error:
        print(f"soundgen powerups: {error}", file=sys.stderr)
        return 1
