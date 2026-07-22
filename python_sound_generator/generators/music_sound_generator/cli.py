"""Command line for the music generator.

Verbs: ``list``, ``info``, ``render``, ``pack``, ``verify``, ``sources``.

Unlike its NumPy siblings, ``render`` and ``pack`` need the external Strudel
toolchain (a dev server + a headless browser + ffmpeg). When any of that is
missing they fail with a message naming the missing piece rather than a
traceback. ``list``, ``info``, ``verify`` (cycle math), and ``sources`` work
with no toolchain at all.
"""

from __future__ import annotations

from ..._audio.runtime import play as play_audio

from ..._audio.runtime import write_manifest as write_shared_manifest

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The seam metric is imported rather than reimplemented: environment_ambience
# _generator already worked out how to tell a loop from a click and a swell.
# For a recording it measures the acoustic seam; the cycle-math guarantee is
# separate and is what `verify` actually gates on.
from ..._audio import loops as loopdsp

from . import engine
from .cues import CUES, GROUPS, PACKS
from .engine import (
    MusicToolchain,
    Options,
    RecorderUnavailable,
    RecordFailed,
    is_integer_cycles,
    pattern_sha256,
    record_seconds,
    recording_session,
)

DEFAULT_OUT = Path("output") / "music"
PROVENANCE = engine.SOURCE_DIRECTORY / "provenance.json"
SYNTHESIS = ('synthesised only: s("sine"/"sawtooth"/"triangle"/"square") + '
             'noise ("white"/"pink"/"brown"); no sample banks')


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _options(args: argparse.Namespace) -> Options:
    return Options(seconds=getattr(args, "seconds", None),
                   emit_ogg=getattr(args, "ogg", False),
                   seed=getattr(args, "seed", 0))


def _seam(samples) -> loopdsp.Seam:
    return loopdsp.seam(list(samples))


def _artifact(render: engine.Render, options: Options,
              take: int) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "file": render.wav_path.name,
        "cue": render.cue,
        "take": take,
        "seed": options.seed,
        "options": {
            "seconds": options.seconds,
            "emit_ogg": options.emit_ogg,
        },
        "record_seconds": round(render.seconds, 6),
        "duration_seconds": render.duration,
        "cps": render.cps,
        "cycles": render.cycles,
        "loop": render.loop,
        "peak_dbfs": render.peak_dbfs,
        "rms_dbfs": render.rms_dbfs,
        # As-rendered record, NOT a golden: a real-time capture is not
        # byte-reproducible. The pinned input is pattern.sha256 below.
        "sha256": render.wav_sha256,
        "webm_sha256": render.webm_sha256,
        "pattern": {
            "id": render.cue,
            "sha256": render.pattern_sha256,
            "source": render.source,
            "synthesis": SYNTHESIS,
        },
    }
    if render.ogg_path is not None:
        entry["ogg"] = render.ogg_path.name
    if render.loop:
        seam = _seam(render.samples)
        entry["integer_cycles"] = is_integer_cycles(CUES[render.cue],
                                                    render.seconds)
        entry["seam"] = {
            "step_ratio": round(seam.step_ratio, 4),
            "level_ratio": round(seam.level_ratio, 4),
            "note": ("acoustic seam of the capture; the head carries real-time "
                     "recording latency, so level_ratio reflects that, not the "
                     "pattern. The clean wrap comes from integer_cycles above."),
        }
    return entry


def _write_manifest(path: Path, artifacts: list[dict[str, Any]], *,
                    command: str, tc: MusicToolchain,
                    pack: str | None = None) -> Path:
    payload: dict[str, Any] = {
        "generator": "music_sound_generator",
        "command": command,
        "engine": ("Strudel (AGPL-3.0-or-later) driven headless as an external "
                   "tool; the audio is the output of running it, not linked "
                   "into Python"),
        "synthesis": SYNTHESIS,
        "determinism": ("real-time MediaRecorder capture is not "
                        "byte-deterministic; each artifact's sha256 is an "
                        "as-rendered record, not a golden. The pinned input is "
                        "the pattern text (pattern.sha256) and configured toolchain."),
        "workspace": {
            "configured_with": "STRUDEL_DIR",
            "directory_name": tc.strudel_dir.name,
        },
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


# --------------------------------------------------------------------------
# read-only verbs (no toolchain needed)
# --------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    show_all = not (args.cues or args.packs)
    if args.cues or show_all:
        for group in GROUPS:
            print(f"\nCues: {group}")
            for cue in CUES.values():
                if cue.group != group:
                    continue
                shape = (f"loop {cue.cycles} cycles @ {cue.cps} cps "
                         f"= {cue.seconds:g}s" if cue.loop
                         else f"one-shot {cue.seconds:g}s @ {cue.cps} cps")
                print(f"  {cue.name:<16} {cue.description}")
                print(f"  {'':<16} {shape}")
    if args.packs or show_all:
        print("\nPacks")
        for pack in PACKS.values():
            total = sum(e.count for e in pack.entries)
            print(f"  {pack.name:<16} {pack.description}  ({total} files)")
    if show_all:
        print("\nAxes (Options)")
        print(f"  {'--seconds':<16} override the capture length (loops: overrides "
              f"the integer-cycle default)")
        print(f"  {'--count':<16} number of takes; each is a distinct real-time "
              f"capture, not a seeded variant")
        print(f"  {'--ogg':<16} also write an .ogg beside each .wav")
        print(f"  {'--seed':<16} recorded in the manifest; does not change the "
              f"audio (the capture is not seeded)")
        _print_toolchain()
    print()
    return 0


def _print_toolchain() -> None:
    tc = MusicToolchain.from_env()
    print("\nToolchain (configurable via STRUDEL_DIR, CHROME_BIN, STRUDEL_PORT)")
    print(f"  {'strudel_dir':<16} {tc.strudel_dir}")
    print(f"  {'chrome_bin':<16} {tc.chrome_bin}")
    print(f"  {'port':<16} {tc.port}")
    issues = tc.problems()
    if issues:
        print("  status           MISSING PIECES:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  status           ready")


def cmd_info(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    cue = CUES[args.cue]
    print(f"\n{cue.name}  [{cue.group}]")
    print(f"  {cue.description}")
    print(f"  tempo:    {cue.cps} cps")
    if cue.loop:
        print(f"  shape:    loop, {cue.cycles} cycles = {cue.seconds:g}s "
              f"(integer cycles -> clean wrap)")
    else:
        print(f"  shape:    one-shot, {cue.seconds:g}s capture")
    print(f"  source:   {cue.source}")
    print(f"  pattern:  sha256 {pattern_sha256(cue)}")
    print(f"  synth:    {SYNTHESIS}")
    print("\n  --- pattern ---")
    for line in cue.pattern.strip("\n").splitlines():
        print(f"  {line}")
    print()
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    data = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    print(f"\nPolicy:  {data['source_audio_policy']}")
    print(f"Tool:    {data['tool']['name']} ({data['tool']['license']}) - "
          f"{data['tool']['role']}")
    print(f"Synth:   {data['synthesis_policy']}")
    print(f"Determinism: {data['determinism']}")
    print("\nPinned patterns:")
    for pat in data["patterns"]:
        shape = (f"loop {pat['cycles']}c" if pat["loop"]
                 else "one-shot")
        print(f"  {pat['cue']:<16} {shape:<10} cps {pat['cps']:<5} "
              f"{pat['sha256'][:16]}...  <- {pat['source']}")
    if args.excluded:
        print("\nConsidered and not used:")
        for item in data["excluded"]:
            print(f"  {item['name']}")
            print(f"    {item['reason']}")
    print()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Report each loop's wrap: the cycle math we control, and -- if a rendered
    file is present -- the acoustic seam that carries capture latency.

    The exit status gates on the cycle math (integer cycles = a clean tempo-grid
    wrap), because that is the property this generator guarantees. The acoustic
    seam is a diagnostic: a real-time capture opens with a few ms of latency, so
    its head/tail energies never match to the tolerance a synthesized loop would.
    """

    out_dir = Path(args.dir)
    print(f"\n{'cue':16s} {'cps':>5s} {'cycles':>7s} {'seconds':>8s}  int  "
          f"acoustic seam (rendered file, if present)")
    worst = 0
    for name, cue in CUES.items():
        if not cue.loop:
            continue
        seconds = record_seconds(cue, Options(seconds=args.seconds))
        integer = is_integer_cycles(cue, seconds)
        worst = max(worst, 0 if integer else 1)
        line = (f"{name:16s} {cue.cps:5g} {str(cue.cycles):>7s} "
                f"{seconds:8.3f}  {'yes' if integer else 'NO ':>3s}")
        wav = out_dir / f"{name}.wav"
        if wav.is_file():
            seam = _seam(engine.decode_wav(wav))
            line += (f"  step {seam.step_ratio:.3f} level {seam.level_ratio:.2f}")
        else:
            line += "  (no rendered file)"
        print(line)
    print("\nint = whole number of pattern cycles -> clean wrap (guaranteed).")
    print("acoustic seam is a diagnostic only; capture latency puts a few ms of "
          "quiet at the head, so level != 1 is expected and does not fail.\n")
    return worst


# --------------------------------------------------------------------------
# rendering verbs (need the toolchain)
# --------------------------------------------------------------------------

def _describe(render: engine.Render, artifact: dict[str, Any]) -> str:
    detail = (f"  wrote {render.wav_path.name}  {render.duration:.2f}s  "
              f"peak {render.peak_dbfs:.1f} dBFS  rms {render.rms_dbfs:.1f} dBFS")
    if "seam" in artifact:
        detail += (f"  [int_cycles {artifact['integer_cycles']}"
                   f" step {artifact['seam']['step_ratio']:.3f}]")
    return detail


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    options = _options(args)
    out_dir = Path(args.out)
    artifacts: list[dict[str, Any]] = []
    print(f"{args.cue}: {CUES[args.cue].description}")
    with recording_session() as session:
        for index in range(args.count):
            stem = args.cue if index == 0 else f"{args.cue}_v{index + 1:02d}"
            render = session.render(args.cue, options, out_dir, stem)
            artifact = _artifact(render, options, index + 1)
            artifacts.append(artifact)
            print(_describe(render, artifact))
            if args.play:
                _play(render.wav_path)
        manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                                   command="render", tc=session.tc)
    print(f"  manifest {manifest}\n")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`", file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name
    options = _options(args)
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")
    artifacts: list[dict[str, Any]] = []
    with recording_session() as session:
        for entry in pack.entries:
            for index in range(entry.count):
                stem = pack.filename(entry, index)[:-len(".wav")]
                render = session.render(entry.cue, options, out_dir, stem)
                artifact = _artifact(render, options, index + 1)
                artifacts.append(artifact)
                print(_describe(render, artifact))
                if args.play:
                    _play(render.wav_path)
        manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                                   command="pack", tc=session.tc, pack=pack.name)
    print(f"  {len(artifacts)} files; manifest {manifest}\n")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen music",
        description="Music by driving Strudel headless; pure-synth patterns.",
        epilog="examples:\n"
               "  soundgen music list\n"
               "  soundgen music info battle_loop\n"
               "  soundgen music render victory_fanfare\n"
               "  soundgen music pack board-duel\n"
               "  soundgen music verify --dir output/showcase\n"
               "  soundgen music sources --excluded\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show cues, packs, axes, toolchain")
    p_list.add_argument("--cues", action="store_true")
    p_list.add_argument("--packs", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="show one cue and its pinned pattern")
    p_info.add_argument("cue", help="cue name; see `list --cues`")
    p_info.set_defaults(func=cmd_info)

    p_sources = sub.add_parser("sources",
                               help="show the licence story and pinned patterns")
    p_sources.add_argument("--excluded", action="store_true",
                           help="also show what was considered and rejected")
    p_sources.set_defaults(func=cmd_sources)

    p_verify = sub.add_parser("verify", help="check every loop's wrap")
    p_verify.add_argument("--dir", type=Path, default=DEFAULT_OUT / "showcase",
                          help="where to look for rendered loop WAVs")
    p_verify.add_argument("--seconds", type=float, default=None,
                          help="check a specific override length")
    p_verify.set_defaults(func=cmd_verify)

    p_render = sub.add_parser("render", help="render one cue (needs toolchain)")
    p_render.add_argument("cue", help="cue name; see `list --cues`")
    p_render.add_argument("--count", type=int, default=1,
                          help="number of takes; each is a distinct capture")
    p_render.add_argument("--seconds", type=float, default=None,
                          help="override the capture length")
    p_render.set_defaults(func=cmd_render)

    p_pack = sub.add_parser("pack", help="render a named pack (needs toolchain)")
    p_pack.add_argument("pack", help="pack name; see `list --packs`")
    p_pack.set_defaults(func=cmd_pack)

    for p in (p_render, p_pack):
        p.add_argument("--seed", type=int, default=20260716,
                       help="recorded in the manifest; does not change the audio")
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--ogg", action="store_true",
                       help="also write an .ogg beside each .wav")
        p.add_argument("--play", action="store_true",
                       help="play each file after writing it")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "count", 1) < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError,
            RecorderUnavailable, RecordFailed) as exc:
        print(str(exc), file=sys.stderr)
        return 2
