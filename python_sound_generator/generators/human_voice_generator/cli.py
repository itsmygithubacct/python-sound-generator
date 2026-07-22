"""Command line for the human voice generator."""

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

from . import engine
from .cues import CUES, GROUPS, PACKS, Options, render

DEFAULT_OUT = Path("output") / "human-voice"
PROVENANCE = engine.SOURCE_DIRECTORY / "provenance.json"


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _options(args: argparse.Namespace) -> Options:
    return Options(body=engine.body_named(args.body),
                   voice=engine.voice_named(args.voice),
                   intensity=args.intensity)


def _onset_seconds(samples: list[float]) -> float:
    """When the vocal first becomes audible: the 3%-of-peak crossing.

    This measures dead lead-in, not the attack shape. A yell is allowed to
    swell up to its peak over tens of milliseconds -- that is the sound -- but
    it must not be preceded by room tone. 3% is the level the render-time trim
    cuts to, so a cue that leads with its onset crosses it within ~15 ms, which
    is what `verify` checks and the manifest records.
    """
    data = np.abs(np.asarray(samples))
    if not len(data) or data.max() <= 0.0:
        return 0.0
    return float(np.argmax(data > data.max() * 0.03)) / engine.SR


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
        "seed": seed,
        "variation": variation,
        "options": {"body": options.body.name,
                    "voice": options.voice.name,
                    "intensity": round(options.intensity, 4)},
        "duration_seconds": metrics['duration_seconds'],
        "onset_seconds": round(_onset_seconds(signal), 4),
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        "sources": list(sources),
    }


def _write_manifest(path: Path, artifacts: list[dict[str, Any]], *,
                    command: str, pack: str | None = None) -> Path:
    payload: dict[str, Any] = {
        "generator": "human_voice_generator",
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
    show_all = not (args.cues or args.bodies or args.packs)
    if args.cues or show_all:
        for group in GROUPS:
            print(f"\nCues: {group}")
            for cue in CUES.values():
                if cue.group != group:
                    continue
                tag = "  (ABSENT)" if cue.absent else ""
                print(f"  {cue.name:<10} {cue.description}{tag}")
                if cue.honours and not cue.absent:
                    voices = "/".join(cue.voices)
                    print(f"  {'':<10} honours: {cue.honours}  voices: {voices}")
    if args.bodies or show_all:
        print("\nBodies (the size axis, applied as varispeed)")
        for b in engine.BODIES.values():
            print(f"  {b.name:<10} {b.description}  (speed x{b.speed:.2f})")
        print("\nVoices")
        for v in engine.VOICES.values():
            print(f"  {v.name}")
    if args.packs or show_all:
        print("\nPacks")
        for pack in PACKS.values():
            total = sum(e.count for e in pack.entries)
            print(f"  {pack.name:<10} {pack.description}  ({total} files)")
    print()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    cue = CUES[args.cue]
    print(f"\n{cue.name}  [{cue.group}]")
    print(f"  {cue.description}")
    if cue.absent:
        print(f"  ABSENT: {cue.absent_reason}")
        print()
        return 0
    print(f"  honours: {cue.honours}")
    print(f"  voices:  {', '.join(cue.voices)}")
    print(f"  peak:    {cue.peak_db:g} dBFS")
    print("  reachable recordings:")
    for name in _cue_sources(args.cue):
        print(f"    {name}")
    print()
    return 0


def _cue_sources(cue_name: str) -> tuple[str, ...]:
    """The recordings a cue can draw from, for `info` and coverage checks."""

    if cue_name in ("hurt", "death"):
        base: tuple[str, ...] = (engine.PAIN_BANK,)
        if cue_name == "hurt":
            base += engine.MALE_GRUNTS + engine.FEMALE_HURT
        return base
    if cue_name == "effort":
        return engine.MALE_GRUNTS + engine.FEMALE_EFFORT
    if cue_name == "shout":
        return engine.MALE_YELLS
    return ()


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
    """Verify every renderable cue leads with its onset, is click-free, and is
    deterministic.

    There are no loop cues here -- every cue is a one-shot vocalization -- so
    there is no loop seam to measure. What matters for a one-shot is that it
    starts at the sound and not after room tone (a late grunt flams over the
    hit it is stacked on), that it does not click at either edge, and that a
    seed reproduces a file exactly. That is what this checks.
    """
    print(f"\n{'cue':10s} {'body':8s} {'onset ms':>9s} {'edge':>8s}  result")
    worst = 0
    for name, cue in CUES.items():
        if cue.absent:
            continue
        for body in engine.BODIES:
            options = Options(body=engine.BODIES[body],
                              voice=engine.VOICES[cue.voices[0]],
                              intensity=0.85)
            a = render(name, options, np.random.default_rng(args.seed))
            b = render(name, options, np.random.default_rng(args.seed))
            onset = _onset_seconds(a.samples)
            edge = max(abs(a.samples[0]), abs(a.samples[-1]))
            ok = (onset <= 0.030 and edge < 0.02 and a.samples == b.samples)
            worst = max(worst, 0 if ok else 1)
            print(f"{name:10s} {body:8s} {onset * 1000:9.1f} {edge:8.4f}  "
                  f"{'ok' if ok else 'FAILS'}")
    print("\n(onset <= 30 ms, |edge| < 0.02, identical seed -> identical file)\n")
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
        stem = f"{args.cue}_{options.body.name}_{options.voice.name}"
        name = f"{stem}.wav" if args.count == 1 else f"{stem}_{index + 1:02d}.wav"
        path = engine.write_wav(out_dir / name, result.samples)
        artifact = _artifact(path, out_dir, result.samples, result.sources,
                             args.cue, seed, index + 1, options)
        artifacts.append(artifact)
        print(f"  wrote {path}  {len(result.samples) / engine.SR:.2f}s"
              f"  onset {artifact['onset_seconds'] * 1000:.0f}ms"
              f"  <- {', '.join(result.sources)}")
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
            artifacts.append(_artifact(
                path, out_dir, result.samples, result.sources, entry.cue, seed,
                index + 1, entry.options,
            ))
            print(f"  wrote {pack.filename(entry, index)}  "
                  f"{len(result.samples) / engine.SR:.2f}s")
            if args.play:
                _play(path)
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="pack", pack=pack.name)
    print(f"  {len(artifacts)} files; manifest {manifest}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen human-voice",
        description="Human non-verbal voice built from bundled CC0 recordings.",
        epilog="examples:\n"
               "  soundgen human-voice list\n"
               "  soundgen human-voice info hurt\n"
               "  soundgen human-voice render hurt --count 5 --play\n"
               "  soundgen human-voice render hurt --voice female\n"
               "  soundgen human-voice render death --body giant\n"
               "  soundgen human-voice pack fantasy-adventure\n"
               "  soundgen human-voice verify\n"
               "  soundgen human-voice sources --excluded\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show cues, bodies, and packs")
    p_list.add_argument("--cues", action="store_true")
    p_list.add_argument("--bodies", action="store_true")
    p_list.add_argument("--packs", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="show one cue and the recordings it uses")
    p_info.add_argument("cue", help="cue name; see `list --cues`")
    p_info.set_defaults(func=cmd_info)

    p_sources = sub.add_parser("sources",
                               help="show the bundled recordings and licences")
    p_sources.add_argument("--excluded", action="store_true",
                           help="also show what was considered and rejected")
    p_sources.set_defaults(func=cmd_sources)

    p_verify = sub.add_parser(
        "verify", help="check every cue's onset, edges, and determinism")
    p_verify.add_argument("--seed", type=int, default=20260716)
    p_verify.set_defaults(func=cmd_verify)

    p_render = sub.add_parser("render", help="render one cue")
    p_render.add_argument("cue", help="cue name; see `list --cues`")
    p_render.add_argument("--count", type=int, default=1,
                          help="variations, each on its own seed")
    p_render.set_defaults(func=cmd_render)

    p_pack = sub.add_parser("pack", help="render a named artifact bank")
    p_pack.add_argument("pack", help="pack name; see `list --packs`")
    p_pack.set_defaults(func=cmd_pack)

    for p in (p_render, p_pack):
        p.add_argument("--seed", type=int, default=20260716,
                       help="base seed; renders are reproducible")
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--play", action="store_true",
                       help="play each file after writing it")
    for p in (p_render,):
        p.add_argument("--body", default="adult",
                       help=f"one of: {', '.join(engine.BODIES)}")
        p.add_argument("--voice", default="male",
                       help=f"one of: {', '.join(engine.VOICES)}")
        p.add_argument("--intensity", type=float, default=0.85,
                       help="0..1, how hard the vocalization lands")
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
