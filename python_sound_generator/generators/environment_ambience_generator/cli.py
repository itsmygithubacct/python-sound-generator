"""Registry-driven command line for environmental ambience."""

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
from . import engine, horn
from .cues import CUES, GROUPS, LOOPS, PACKS, Options, render, stem_for


DEFAULT_OUT = Path("output") / "ambience"
DEFAULT_SEED = 20260715


def _header(text: str) -> None:
    print(text)


def _item(name: str, detail: str) -> None:
    print(f"  {name.ljust(20)} {detail}")


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _options_from_args(args: argparse.Namespace) -> Options:
    return Options(
        seconds=args.seconds,
        intensity=args.intensity,
        foliage=args.foliage,
        pattern=args.pattern,
        quarter=args.quarter,
        hour=args.hour,
    )


def _options_dict(options: Options, cue_name: str) -> dict[str, Any]:
    cue = CUES[cue_name]
    values: dict[str, Any] = {
        "seconds": round(cue.duration(options), 6),
        "intensity": options.intensity,
        "foliage": options.foliage,
        "pattern": options.pattern,
        "quarter": options.quarter,
        "hour": options.hour,
    }
    return {name: values[name] for name in cue.honours}


def _seam_dict(signal: list[float]) -> dict[str, Any]:
    measured = loopdsp.seam(signal)
    return {
        "step_ratio": round(measured.step_ratio, 6),
        "level_ratio": round(measured.level_ratio, 6),
        "seamless": measured.is_seamless(),
    }


def _artifact(
    path: Path,
    signal: list[float],
    cue_name: str,
    seed: int,
    variation: int,
    options: Options,
) -> dict[str, Any]:
    metrics = artifact_metrics(path, signal, engine.SR)
    artifact: dict[str, Any] = {
        "file": path.name,
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
        artifact["seam"] = _seam_dict(signal)
    return artifact


def _write_manifest(
    path: Path,
    artifacts: list[dict[str, Any]],
    *,
    command: str,
    pack: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "generator": "environment_ambience_generator",
        "command": command,
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
    selected = any((args.cues, args.loops, args.oneshots, args.packs))
    if args.cues or not selected:
        for group in GROUPS:
            _header(f"{group} cues")
            for cue in CUES.values():
                if cue.group == group:
                    _item(cue.name, cue.description)
        print()
    if args.loops or not selected:
        _header("loops")
        for name in LOOPS:
            _item(name, f"default {CUES[name].seconds:g}s seamless loop")
        print()
    if args.oneshots or not selected:
        _header("one-shots")
        for cue in CUES.values():
            if not cue.loop:
                _item(cue.name, f"intensity is {cue.intensity_note}")
        print()
    if args.packs or not selected:
        _header("packs")
        for pack in PACKS.values():
            count = sum(entry.count for entry in pack.entries)
            _item(pack.name, f"{pack.description} ({count} files)")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    if args.cue is not None and args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    names = [args.cue] if args.cue else list(CUES)
    options = _options_from_args(args)
    for name in names:
        cue = CUES[name]
        kind = "loop" if cue.loop else "one-shot"
        _header(f"{cue.name}: {cue.description}")
        print(f"  group {cue.group}; {kind}; peak {cue.peak_db:.1f} dBFS")
        print(f"  duration {cue.duration(options):g}s"
              f"{'' if cue.loop else ' (fixed)'}; "
              f"honours {', '.join(cue.honours)}")
        print(f"  intensity is {cue.intensity_note}")
        if cue.layers:
            print(f"  layers: {'; '.join(cue.layers)}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    if args.count < 1:
        print("--count must be at least 1", file=sys.stderr)
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

    options = _options_from_args(args)
    out_dir = Path(args.out)
    stem = args.stem or stem_for(args.cue, options)
    artifacts: list[dict[str, Any]] = []
    last_path: Path | None = None
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        signal = render(args.cue, options, random.Random(seed))
        filename = (f"{stem}.wav" if index == 0
                    else f"{stem}_v{index + 1:02d}.wav")
        last_path = engine.write_wav(out_dir / filename, signal)
        artifacts.append(
            _artifact(last_path, signal, args.cue, seed, index + 1, options)
        )
        _describe(last_path, signal, args.cue)
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="render")
    print(f"  manifest {manifest}")
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
            path = engine.write_wav(
                out_dir / pack.filename(entry, index), signal
            )
            artifacts.append(
                _artifact(path, signal, entry.cue, seed, index + 1,
                          entry.options)
            )
            _describe(path, signal, entry.cue)
            if args.play:
                _play(path)
    manifest = _write_manifest(
        out_dir / "manifest.json", artifacts, command="pack", pack=pack.name
    )
    print(f"  {len(artifacts)} files; manifest {manifest}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Render loops and report what their loop points measure.

    This exists because "it loops" is a claim that can be checked, and a
    generator that makes the claim should be able to show its work.
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

    options = _options_from_args(args)
    print(f"{'cue':14s} {'step':>8s} {'level':>8s}  result")
    failed = 0
    for name in names:
        signal = render(name, options, random.Random(args.seed))
        measured = loopdsp.seam(signal)
        ok = measured.is_seamless()
        failed += 0 if ok else 1
        print(f"{name:14s} {measured.step_ratio:8.4f} "
              f"{measured.level_ratio:8.3f}  "
              f"{'seamless' if ok else 'NOT SEAMLESS'}")
    print(f"\n{len(names) - failed}/{len(names)} seamless "
          f"(step <= {loopdsp.MAX_STEP_RATIO:g}, level within "
          f"{1 / loopdsp.MAX_LEVEL_DRIFT:g}-{loopdsp.MAX_LEVEL_DRIFT:g})")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen ambience",
        description=(
            "Generate seamless environmental ambience loops and matching "
            "one-shots with dependency-free procedural DSP."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<verb>")

    list_parser = sub.add_parser("list", help="show registries and packs")
    list_parser.add_argument("--cues", action="store_true")
    list_parser.add_argument("--loops", action="store_true")
    list_parser.add_argument("--oneshots", action="store_true")
    list_parser.add_argument("--packs", action="store_true")
    list_parser.set_defaults(func=cmd_list)

    info_parser = sub.add_parser("info", help="describe cues and their layers")
    info_parser.add_argument("--cue", help="describe one cue instead of all")
    info_parser.set_defaults(func=cmd_info)

    render_parser = sub.add_parser("render", help="render one cue")
    render_parser.add_argument("cue", help="cue name; see `list --cues`")
    render_parser.add_argument("--count", type=int, default=1)
    render_parser.add_argument("--stem",
                               help="override the generated filename stem")
    render_parser.set_defaults(func=cmd_render)

    pack_parser = sub.add_parser("pack", help="render a named artifact bank")
    pack_parser.add_argument("pack", help="pack name; see `list --packs`")
    pack_parser.set_defaults(func=cmd_pack)

    verify_parser = sub.add_parser(
        "verify", help="measure the loop point of every loop cue"
    )
    verify_parser.add_argument("--cue", help="verify one cue instead of all")
    verify_parser.set_defaults(func=cmd_verify)

    for target in (info_parser, render_parser, verify_parser):
        target.add_argument(
            "--seconds", type=float, default=None,
            help="loop length; default is per-cue (ignored by one-shots)",
        )
        target.add_argument("--intensity", type=float, default=0.6)
        target.add_argument(
            "--foliage", type=float, default=0.5,
            help="0 open field, 1 dense forest (wind, forest)",
        )
        target.add_argument(
            "--pattern", default="underway", choices=sorted(horn.PATTERNS),
            help="fog signal pattern (foghorn)",
        )
        target.add_argument(
            "--quarter", type=int, default=4, choices=(1, 2, 3, 4),
            help="Westminster quarter (bell_chime)",
        )
        target.add_argument(
            "--hour", type=int, default=0,
            help="strike the hour after the chime, 0 for none (bell_chime)",
        )
    for target in (render_parser, pack_parser, verify_parser):
        target.add_argument("--seed", type=int, default=DEFAULT_SEED)
    for target in (render_parser, pack_parser):
        target.add_argument("--out", type=Path, default=DEFAULT_OUT)
        target.add_argument("--play", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (OSError, ValueError) as error:
        print(f"soundgen ambience: {error}", file=sys.stderr)
        return 1
