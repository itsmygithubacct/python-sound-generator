"""Registry-driven command line for the magic generator."""

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

# The seam metric is imported rather than reimplemented, exactly as
# fire_sound_generator imports it: environment_ambience_generator already
# worked out how to tell a loop from a click and a swell, and two generators
# disagreeing about what "seamless" means would be worse than either answer.
from ..._audio import loops as loopdsp

from . import engine
from .cues import CUES, GROUPS, LOOPS, PACKS, Options, render, stem_for
from .engine import STYLES, style_named


DEFAULT_OUT = Path("output") / "magic"
DEFAULT_SEED = 20260716


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _options(args: argparse.Namespace) -> Options:
    return Options(style=style_named(args.style),
                   intensity=args.intensity,
                   seconds=getattr(args, "seconds", None))


def _options_dict(options: Options, cue_name: str) -> dict[str, Any]:
    cue = CUES[cue_name]
    values: dict[str, Any] = {
        "style": options.style.name,
        "intensity": round(options.intensity, 4),
        "seconds": round(cue.duration(options), 6),
    }
    return {name: values[name] for name in cue.honours}


def _artifact(
    path: Path,
    root: Path,
    signal: list[float],
    cue_name: str,
    seed: int,
    variation: int,
    options: Options,
) -> dict[str, Any]:
    metrics = artifact_metrics(path, signal, engine.SR)
    artifact: dict[str, Any] = {
        "file": path.resolve().relative_to(root.resolve()).as_posix(),
        "cue": cue_name,
        "seed": seed,
        "variation": variation,
        "options": _options_dict(options, cue_name),
        "loop": CUES[cue_name].loop,
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
    }
    # A loop that is not seamless is a defective artifact, so the evidence goes
    # in the manifest beside the checksum rather than only on the terminal.
    if CUES[cue_name].loop:
        measured = loopdsp.seam(signal)
        artifact["seam"] = {
            "step_ratio": round(measured.step_ratio, 6),
            "level_ratio": round(measured.level_ratio, 6),
            "seamless": measured.is_seamless(),
        }
    return artifact


def _write_manifest(path: Path, artifacts: list[dict[str, Any]], *,
                    command: str, pack: str | None = None) -> Path:
    payload: dict[str, Any] = {
        "generator": "magic_sound_generator",
        "command": command,
        "synthesis": "procedural DSP, standard library only; no recordings",
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


def _describe(path: Path, signal: list[float], cue_name: str) -> None:
    detail = f"  wrote {path}  {len(signal) / engine.SR:.2f}s"
    if CUES[cue_name].loop:
        measured = loopdsp.seam(signal)
        mark = "seamless" if measured.is_seamless() else "SEAM"
        detail += (f"  [{mark} step {measured.step_ratio:.3f} "
                   f"level {measured.level_ratio:.2f}]")
    print(detail)


def cmd_list(args: argparse.Namespace) -> int:
    selected = any((args.cues, args.styles, args.packs))
    if args.cues or not selected:
        for group in GROUPS:
            print(f"\n{group} cues")
            for cue in CUES.values():
                if cue.group == group:
                    print(f"  {cue.name:<14} {cue.description}")
                    print(f"  {'':<14} honours: {', '.join(cue.honours)}")
    if args.styles or not selected:
        print("\nelements (style axis)")
        for s in STYLES.values():
            print(f"  {s.name:<14} {s.description}")
    if args.packs or not selected:
        print("\npacks")
        for pack in PACKS.values():
            total = sum(entry.count for entry in pack.entries)
            print(f"  {pack.name:<14} {pack.description}  ({total} files)")
    print()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    if args.cue is not None and args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    names = [args.cue] if args.cue else list(CUES)
    options = _options(args)
    for name in names:
        cue = CUES[name]
        kind = "loop" if cue.loop else "one-shot"
        print(f"\n{cue.name}: {cue.description}")
        print(f"  group {cue.group}; {kind}; peak {cue.peak_db:.1f} dBFS")
        print(f"  duration {cue.duration(options):g}s"
              f"{'' if cue.loop else ' (fixed)'}; "
              f"honours {', '.join(cue.honours)}")
        print(f"  intensity is {cue.intensity_note}")
        if cue.layers:
            print(f"  layers: {'; '.join(cue.layers)}")
    print()
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    if args.count < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    if args.style not in STYLES:
        print(f"unknown style {args.style!r}; see `list --styles`",
              file=sys.stderr)
        return 2
    if args.seconds is not None and not (
        engine.MIN_SECONDS <= args.seconds <= engine.MAX_SECONDS
    ):
        print(f"--seconds must be between {engine.MIN_SECONDS} and "
              f"{engine.MAX_SECONDS}", file=sys.stderr)
        return 2
    if not 0.05 <= args.intensity <= 1.0:
        print("--intensity must be between 0.05 and 1", file=sys.stderr)
        return 2

    options = _options(args)
    out_dir = Path(args.out)
    stem = args.stem or stem_for(args.cue, options)
    artifacts: list[dict[str, Any]] = []
    last_path: Path | None = None
    print(f"{args.cue}: {CUES[args.cue].description}")
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        signal = render(args.cue, options, random.Random(seed))
        filename = (f"{stem}.wav" if index == 0
                    else f"{stem}_v{index + 1:02d}.wav")
        last_path = engine.write_wav(out_dir / filename, signal)
        artifacts.append(
            _artifact(
                last_path, out_dir, signal, args.cue, seed, index + 1, options
            )
        )
        _describe(last_path, signal, args.cue)
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="render")
    print(f"  manifest {manifest}\n")
    if args.play and last_path is not None:
        _play(last_path)
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`",
              file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name
    artifacts: list[dict[str, Any]] = []
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")
    for entry in pack.entries:
        for index in range(entry.count):
            seed = derive_seed(args.seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            signal = render(entry.cue, entry.options, random.Random(seed))
            path = engine.write_wav(out_dir / pack.filename(entry, index),
                                    signal)
            artifacts.append(
                _artifact(
                    path, out_dir, signal, entry.cue, seed, index + 1,
                    entry.options,
                )
            )
            _describe(path, signal, entry.cue)
            if args.play:
                _play(path)
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="pack", pack=pack.name)
    print(f"  {len(artifacts)} files; manifest {manifest}\n")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Render every loop cue and report what its loop point measures.

    "It loops" is a claim that can be checked; a generator making it should be
    able to show its work.
    """

    if args.cue is not None and args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    names = [args.cue] if args.cue else list(LOOPS)
    names = [name for name in names if CUES[name].loop]
    if not names:
        print("no loop cues selected; one-shots have no loop point",
              file=sys.stderr)
        return 2

    print(f"\n{'cue':14s} {'style':8s} {'step':>8s} {'level':>8s}  result")
    failed = 0
    for name in names:
        for style in STYLES:
            options = Options(style=STYLES[style], seconds=args.seconds)
            signal = render(name, options, random.Random(args.seed))
            measured = loopdsp.seam(signal)
            ok = measured.is_seamless()
            failed += 0 if ok else 1
            print(f"{name:14s} {style:8s} {measured.step_ratio:8.4f} "
                  f"{measured.level_ratio:8.3f}  "
                  f"{'seamless' if ok else 'NOT SEAMLESS'}")
    total = len(names) * len(STYLES)
    print(f"\n{total - failed}/{total} seamless "
          f"(step <= {loopdsp.MAX_STEP_RATIO:g}, level within "
          f"{1 / loopdsp.MAX_LEVEL_DRIFT:g}-{loopdsp.MAX_LEVEL_DRIFT:g})\n")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen magic",
        description="Procedural magic and energy sound effects, "
                    "standard-library DSP, no recordings.",
        epilog="examples:\n"
               "  soundgen magic list\n"
               "  soundgen magic render beam --count 5\n"
               "  soundgen magic render charge "
               "--style holy --play\n"
               "  soundgen magic render shield_loop "
               "--seconds 12\n"
               "  soundgen magic verify\n"
               "  soundgen magic pack fantasy-adventure\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="<verb>")

    p_list = sub.add_parser("list", help="show cues, elements, and packs")
    p_list.add_argument("--cues", action="store_true")
    p_list.add_argument("--styles", action="store_true")
    p_list.add_argument("--packs", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="describe cues and their layers")
    p_info.add_argument("--cue", help="describe one cue instead of all")
    p_info.set_defaults(func=cmd_info)

    p_render = sub.add_parser("render", help="render one cue")
    p_render.add_argument("cue", help="cue name; see `list --cues`")
    p_render.add_argument("--count", type=int, default=1,
                          help="variations, each on its own seed")
    p_render.add_argument("--stem", help="override the generated filename stem")
    p_render.set_defaults(func=cmd_render)

    p_pack = sub.add_parser("pack", help="render a named artifact bank")
    p_pack.add_argument("pack", help="pack name; see `list --packs`")
    p_pack.set_defaults(func=cmd_pack)

    p_verify = sub.add_parser("verify",
                              help="measure the loop point of every loop cue")
    p_verify.add_argument("--cue", help="verify one cue instead of all")
    p_verify.add_argument("--seconds", type=float, default=8.0)
    p_verify.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_verify.set_defaults(func=cmd_verify)

    for p in (p_render, p_pack):
        p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help="base seed; renders are reproducible")
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--play", action="store_true",
                       help="play each file after writing it")
    for p in (p_info, p_render):
        p.add_argument("--style", default="arcane",
                       help=f"element: {', '.join(STYLES)}")
        p.add_argument("--intensity", type=float, default=0.85,
                       help="0.05..1, how hard the spell lands")
        p.add_argument("--seconds", type=float, default=None,
                       help="loop length; loops only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (OSError, ValueError, KeyError) as error:
        print(f"soundgen magic: {error}", file=sys.stderr)
        return 2
