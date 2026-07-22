"""Command line for the wood noise generator."""

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
from .cues import CUES, GROUPS, PACKS, Options, Pack, render

DEFAULT_OUT = Path("output") / "wood"
PROVENANCE = engine.SOURCE_DIRECTORY / "provenance.json"


def _play(path: Path) -> None:
    if not play_audio(path):
        print(
            "no audio player found (tried paplay, aplay, afplay, play, ffplay)",
            file=sys.stderr,
        )


def _options(args: argparse.Namespace, cue_name: str) -> Options:
    cue = CUES[cue_name]
    form_name = args.form
    if form_name is None:
        form_name = (
            engine.NEUTRAL_FORM
            if cue.supports(engine.NEUTRAL_FORM)
            else cue.offered_forms()[0]
        )
    return Options(form=engine.form_named(form_name), intensity=args.intensity)


def _artifact(path: Path, signal: list[float], sources: tuple[str, ...],
              cue_name: str, seed: int, variation: int,
              options: Options) -> dict[str, Any]:
    metrics = artifact_metrics(path, signal, engine.SR)
    return {
        "file": path.name,
        "cue": cue_name,
        "seed": seed,
        "variation": variation,
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        "sources": list(sources),
        "options": {"form": options.form.name,
                    "intensity": round(options.intensity, 4)},
    }


def _write_manifest(path: Path, artifacts: list[dict[str, Any]], *,
                    command: str, pack: str | None = None,
                    skipped: list[str] | None = None) -> Path:
    payload: dict[str, Any] = {
        "generator": "wood_noise_generator",
        "command": command,
        "source_audio": "CC0 recordings; see sources/provenance.json",
        "artifacts": artifacts,
    }
    if pack is not None:
        payload["pack"] = pack
    if skipped:
        # Recorded rather than silently dropped: a bank that is quietly short is
        # worse than one that says what it is missing and why.
        payload["skipped"] = skipped
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
    show_all = not (args.cues or args.forms or args.packs)
    if args.cues or show_all:
        for group in GROUPS:
            print(f"\nCues: {group}")
            for cue in CUES.values():
                if cue.group != group:
                    continue
                pending = "" if cue.available() else "   [pending sources]"
                print(f"  {cue.name:<10} {cue.description}{pending}")
                if cue.honours:
                    print(f"  {'':<10} honours: {cue.honours}")
                if cue.form_flat():
                    print(f"  {'':<10} forms:   (form-flat; renders once)")
                elif set(cue.forms) != set(engine.FORMS):
                    print(f"  {'':<10} forms:   {', '.join(cue.forms)}")
    if args.forms or show_all:
        print("\nForms")
        for form in engine.FORMS.values():
            print(f"  {form.name:<10} {form.description}")
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
    print(f"\n{cue.name}  ({cue.group})")
    print(f"  {cue.description}")
    print(f"  honours:   {cue.honours or 'nothing'}")
    print(f"  peak:      {cue.peak_db} dBFS")
    doc = (cue.render.__doc__ or "").strip()
    if doc:
        print("\n" + "\n".join("  " + line for line in doc.splitlines()))
    missing = cue.missing()
    if missing:
        print(f"\n  UNAVAILABLE - needs: {', '.join(missing)}")
        print("  see `sources --pending`")
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
    if args.pending or not args.excluded:
        for item in data.get("pending", []):
            print(f"\n  PENDING: {item['name']} - {item['license']}")
            print(f"    page:     {item['source_page']}")
            print(f"    expected: {item['expected_file']}")
            print(f"    {item['reason']}")
    if args.excluded:
        print("\nConsidered and not used:")
        for item in data["excluded"]:
            print(f"\n  {item['name']}")
            print(f"    {item['source_page']}")
            print(f"    {item['reason']}")
    print()
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    cue = CUES[args.cue]
    options = _options(args, args.cue)
    if not cue.supports(options.form.name):
        print(f"{args.cue!r} is not offered at form {options.form.name!r} "
              f"(rejected by ear 2026-07-16); offered: {', '.join(cue.forms)}",
              file=sys.stderr)
        return 2
    if cue.form_flat():
        if args.form is not None and args.form != engine.NEUTRAL_FORM:
            print(f"  note: {args.cue} is form-flat; --form {args.form} ignored")
        # Keep the manifest honest: record what was actually rendered.
        options = Options(form=engine.FORMS[engine.NEUTRAL_FORM],
                          intensity=args.intensity)
    out_dir = Path(args.out)
    artifacts: list[dict[str, Any]] = []
    print(f"{args.cue}: {cue.description}")
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        result = render(args.cue, options, np.random.default_rng(seed))
        stem = args.cue if cue.form_flat() else f"{args.cue}_{options.form.name}"
        name = f"{stem}.wav" if args.count == 1 else f"{stem}_{index + 1:02d}.wav"
        path = engine.write_wav(out_dir / name, result.samples)
        artifacts.append(_artifact(path, result.samples, result.sources,
                                   args.cue, seed, index + 1, options))
        print(f"  wrote {path}  {len(result.samples) / engine.SR:.2f}s")
        if args.play:
            _play(path)
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="render")
    print(f"  manifest {manifest}\n")
    return 0


def _render_pack(pack: Pack, out_dir: Path, seed: int,
                 play: bool) -> tuple[list[dict[str, Any]], list[str]]:
    artifacts: list[dict[str, Any]] = []
    skipped: list[str] = []
    for entry in pack.entries:
        cue = CUES[entry.cue]
        if not cue.available():
            # A partial bank is a working bank; a wrong sound would not be.
            note = f"{entry.filename} ({entry.cue}): needs {', '.join(cue.missing())}"
            if note not in skipped:
                skipped.append(note)
                print(f"  skipped {entry.filename}  ({entry.cue}: "
                      f"{', '.join(cue.missing())} not bundled)")
            continue
        for index in range(entry.count):
            file_seed = derive_seed(seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            result = render(entry.cue, entry.options,
                            np.random.default_rng(file_seed))
            path = engine.write_wav(out_dir / pack.filename(entry, index),
                                    result.samples)
            artifacts.append(_artifact(path, result.samples, result.sources,
                                       entry.cue, file_seed, index + 1,
                                       entry.options))
            print(f"  wrote {path.name}  {len(result.samples) / engine.SR:.2f}s")
            if play:
                _play(path)
    return artifacts, skipped


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`",
              file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")
    artifacts, skipped = _render_pack(pack, out_dir, args.seed, args.play)
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="pack", pack=pack.name, skipped=skipped)
    print(f"  {len(artifacts)} files; manifest {manifest}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen wood",
        description="Everything wood does, built from bundled CC0 recordings.",
        epilog="examples:\n"
               "  soundgen wood list\n"
               "  soundgen wood info break\n"
               "  soundgen wood render strike --form beam "
               "--count 5 --play\n"
               "  soundgen wood render creak --form beam "
               "--play\n"
               "  soundgen wood pack showcase\n"
               "  soundgen wood sources --excluded\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show cues, forms, and packs")
    p_list.add_argument("--cues", action="store_true")
    p_list.add_argument("--forms", action="store_true")
    p_list.add_argument("--packs", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="everything about one cue")
    p_info.add_argument("cue", help="cue name; see `list --cues`")
    p_info.set_defaults(func=cmd_info)

    p_sources = sub.add_parser("sources",
                               help="show the bundled recordings and licences")
    p_sources.add_argument("--excluded", action="store_true",
                           help="also show what was considered and rejected")
    p_sources.add_argument("--pending", action="store_true",
                           help="show recordings approved but not yet fetched")
    p_sources.set_defaults(func=cmd_sources)

    p_render = sub.add_parser("render", help="render one cue")
    p_render.add_argument("cue", help="cue name; see `list --cues`")
    p_render.add_argument("--count", type=int, default=1,
                          help="variations, each on its own seed")
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
        p.add_argument(
            "--form",
            help=(
                f"one of: {', '.join(engine.FORMS)}; defaults to plank when "
                "offered, otherwise the cue's first offered form"
            ),
        )
        p.add_argument("--intensity", type=float, default=0.9,
                       help="0..1, how hard it happens")
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
