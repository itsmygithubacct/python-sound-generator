"""Registry-driven command line for creature Foley SFX."""

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
from .cues import CUES, GROUPS, PACKS, Options, duration_for, render, stem_for

DEFAULT_OUT = Path("output") / "creature"
DEFAULT_SEED = 20260715
PROVENANCE = engine.SOURCE_DIRECTORY / "provenance.json"


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
        size=engine.size_named(args.size),
        character=engine.character_named(args.character),
        intensity=args.intensity,
        beats=args.beats,
        room=args.room,
        duration=args.duration,
    )


def _options_dict(options: Options, cue_name: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "size": options.size.name,
        "character": options.character.name,
        "intensity": options.intensity,
        "beats": options.beats,
        "room": options.room,
    }
    honoured = {
        name.strip() for name in CUES[cue_name].honours if name.strip()
    }
    return {name: values[name] for name in values if name in honoured}


def _artifact(
    path: Path,
    signal: list[float],
    sources: tuple[str, ...],
    cue_name: str,
    seed: int,
    variation: int,
    options: Options,
) -> dict[str, Any]:
    metrics = artifact_metrics(path, signal, engine.SR)
    return {
        "file": path.name,
        "cue": cue_name,
        "seed": seed,
        "variation": variation,
        "options": _options_dict(options, cue_name),
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        "sources": list(sources),
    }


def _write_manifest(
    path: Path,
    artifacts: list[dict[str, Any]],
    *,
    command: str,
    pack: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "generator": "creature_foley_generator",
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
    selected = any((args.cues, args.sizes, args.characters, args.packs))
    if args.cues or not selected:
        for group in GROUPS:
            _header(f"{group} cues")
            for cue in CUES.values():
                if cue.group == group:
                    _item(cue.name, cue.description)
                    if cue.honours:
                        print(f"  {'':20} honours: {cue.honours}")
        print()
    if args.sizes or not selected:
        _header("sizes")
        for size in engine.SIZES.values():
            _item(size.name, size.description)
        print()
    if args.characters or not selected:
        _header("characters")
        for character in engine.CHARACTERS.values():
            _item(character.name, character.description)
        print()
    if args.packs or not selected:
        _header("packs")
        for pack in PACKS.values():
            count = sum(entry.count for entry in pack.entries)
            _item(pack.name, f"{pack.description} ({count} files)")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    size = engine.size_named(args.size)
    character = engine.character_named(args.character)

    _header(f"size: {size.label}")
    print(f"  {size.description}")
    print(f"  varispeed {size.speed:.2f}x; body weight {size.weight:.2f}; "
          f"wing rate {size.wing_rate:.1f}/s")
    _header(f"character: {character.label}")
    print(f"  {character.description}")
    print(f"  speed bias {character.speed_bias:.2f}; warmth {character.warmth:+.2f}; "
          f"brightness {character.brightness:+.2f}; wet {character.wet:.2f}; "
          f"air {character.air:.2f}")
    if args.cue:
        if args.cue not in CUES:
            print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
            return 2
        cue = CUES[args.cue]
        options = Options(size=size, character=character, beats=args.beats)
        print()
        _header(f"cue: {cue.name}")
        print(f"  {cue.description}")
        print(f"  group {cue.group}; peak {cue.peak_db:.1f} dBFS; "
              f"honours {cue.honours or 'no options'}")
        print(f"  natural duration for this creature: "
              f"{duration_for(cue.name, options):.2f}s")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    if not PROVENANCE.is_file():
        print(f"missing {PROVENANCE}", file=sys.stderr)
        return 1
    payload = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    print(f"policy: {payload['source_audio_policy']}")
    print(f"retrieved: {payload['retrieved_utc']}")
    print()
    for collection in payload["collections"]:
        _header(f"{collection['name']} — {collection['author']}")
        print(f"  license {collection['license']}; {len(collection['files'])} files")
        print(f"  {collection['source_page']}")
        upstream = collection.get("upstream_source")
        if upstream:
            print(f"  derived from {upstream['name']} by {upstream['author']} "
                  f"({upstream['license']}) {upstream['source_page']}")
    reachable = set(engine.bundled_sources())
    bundled = {
        entry["bundled"]
        for collection in payload["collections"]
        for entry in collection["files"]
    }
    print()
    print(f"  {len(bundled)} bundled recordings; {len(reachable)} reachable from cues")
    missing = sorted(reachable - bundled)
    if missing:
        print(f"  MISSING from provenance: {missing}", file=sys.stderr)
        return 1
    if args.excluded:
        print()
        _header("excluded sources")
        for item in payload.get("excluded", []):
            _item(item["name"], item["reason"])
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    if args.count < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2

    options = _options_from_args(args)
    out_dir = Path(args.out)
    stem = args.stem or stem_for(args.cue, options)
    artifacts: list[dict[str, Any]] = []
    last_path: Path | None = None
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        result = render(args.cue, options, np.random.default_rng(seed))
        filename = f"{stem}.wav" if index == 0 else f"{stem}_v{index + 1:02d}.wav"
        last_path = engine.write_wav(out_dir / filename, result.samples)
        artifacts.append(_artifact(
            last_path, result.samples, result.sources, args.cue, seed,
            index + 1, options,
        ))
        print(f"  wrote {last_path}  {len(result.samples) / engine.SR:.2f}s  "
              f"[{', '.join(result.sources)}]")
    manifest = _write_manifest(out_dir / "manifest.json", artifacts, command="render")
    print(f"  manifest {manifest}")
    if args.play and last_path is not None:
        _play(last_path)
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list --packs`", file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name
    artifacts: list[dict[str, Any]] = []
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")
    for entry in pack.entries:
        for index in range(entry.count):
            seed = derive_seed(args.seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            result = render(entry.cue, entry.options, np.random.default_rng(seed))
            path = engine.write_wav(
                out_dir / pack.filename(entry, index), result.samples
            )
            artifacts.append(_artifact(
                path, result.samples, result.sources, entry.cue, seed,
                index + 1, entry.options,
            ))
            print(f"  wrote {path.name}  {len(result.samples) / engine.SR:.2f}s")
            if args.play:
                _play(path)
    manifest = _write_manifest(
        out_dir / "manifest.json", artifacts, command="pack", pack=pack.name
    )
    print(f"  {len(artifacts)} files; manifest {manifest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen creature",
        description=(
            "Generate creature vocalizations, combat Foley, wing beats, body "
            "movement, and hatching from bundled CC0 recordings."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<verb>")

    list_parser = sub.add_parser("list", help="show registries and packs")
    list_parser.add_argument("--cues", action="store_true")
    list_parser.add_argument("--sizes", action="store_true")
    list_parser.add_argument("--characters", action="store_true")
    list_parser.add_argument("--packs", action="store_true")
    list_parser.set_defaults(func=cmd_list)

    info_parser = sub.add_parser("info", help="describe the selected creature")
    info_parser.add_argument("--cue", help="also describe a cue")
    info_parser.set_defaults(func=cmd_info)

    sources_parser = sub.add_parser(
        "sources", help="show recording provenance and licensing"
    )
    sources_parser.add_argument(
        "--excluded", action="store_true", help="also show rejected sources"
    )
    sources_parser.set_defaults(func=cmd_sources)

    render_parser = sub.add_parser("render", help="render one cue")
    render_parser.add_argument("cue", help="cue name; see `list --cues`")
    render_parser.add_argument("--count", type=int, default=1)
    render_parser.add_argument("--stem", help="override the generated filename stem")
    render_parser.add_argument("--duration", type=float, help="override cue length")
    render_parser.set_defaults(func=cmd_render)

    pack_parser = sub.add_parser("pack", help="render a named artifact bank")
    pack_parser.add_argument("pack", help="pack name; see `list --packs`")
    pack_parser.set_defaults(func=cmd_pack)

    for target in (info_parser, render_parser):
        target.add_argument("--size", default="medium", choices=engine.SIZES)
        target.add_argument("--character", default="beast", choices=engine.CHARACTERS)
        target.add_argument("--beats", type=int, default=3)
    for target in (render_parser,):
        target.add_argument("--intensity", type=float, default=0.85)
        target.add_argument("--room", type=float, default=0.06)
    for target in (render_parser, pack_parser):
        target.add_argument("--seed", type=int, default=DEFAULT_SEED)
        target.add_argument("--out", type=Path, default=DEFAULT_OUT)
        target.add_argument("--play", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (OSError, ValueError) as error:
        print(f"soundgen creature: {error}", file=sys.stderr)
        return 1
