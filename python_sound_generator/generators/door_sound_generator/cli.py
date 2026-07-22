"""Pack-aware command line for recorded door and latch Foley."""

from __future__ import annotations

from ..._audio.runtime import artifact_metrics

from ..._audio.runtime import derive_seed

from ..._audio.runtime import AudioFormat
from ..._audio.runtime import write_manifest as write_shared_manifest

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from . import engine
from .cues import PACKS, Options

DEFAULT_OUT = Path("output") / "doors"
DEFAULT_SEED = 20260715


def _render(options: Options, seed: int) -> engine.AudioBuffer:
    return engine.generate_door_sound(
        action=options.action,
        material=options.material,
        intensity=options.intensity,
        age=options.age,
        duration=options.duration,
        room=options.room,
        seed=seed,
        stereo=options.stereo,
    )


def _artifact(
    path: Path,
    root: Path,
    audio: engine.AudioBuffer,
    options: Options,
    seed: int,
    variation: int,
) -> dict[str, Any]:
    samples = [sample for channel in audio.channels for sample in channel]
    metrics = artifact_metrics(path, samples, audio.sample_rate)
    return {
        "file": path.relative_to(root).as_posix(),
        "cue": options.action,
        "seed": seed,
        "variation": variation,
        "options": {
            "action": options.action,
            "material": options.material,
            "intensity": options.intensity,
            "age": options.age,
            "duration": options.duration,
            "room": options.room,
            "stereo": options.stereo,
        },
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
        "sources": list(audio.sources),
    }


def _write_manifest(
    path: Path,
    artifacts: list[dict[str, Any]],
    *,
    command: str,
    pack: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "generator": "door_sound_generator",
        "command": command,
        "source_audio": "CC0/public-domain recordings; see sources/provenance.json",
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
        audio_format=AudioFormat(
            channels=(
                2
                if any(
                    artifact.get("options", {}).get("stereo")
                    for artifact in artifact_rows
                )
                else 1
            )
        ),
        extra=payload,
    )


def cmd_list(args: argparse.Namespace) -> int:
    for pack in PACKS.values():
        count = sum(entry.count for entry in pack.entries)
        print(f"{pack.name:<15} {pack.description} ({count} files)")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in PACKS:
        print(f"unknown pack {args.pack!r}; see `list`", file=sys.stderr)
        return 2
    pack = PACKS[args.pack]
    out_dir = Path(args.out) / pack.name
    artifacts: list[dict[str, Any]] = []
    print(f"pack {pack.name}: {pack.description} -> {out_dir}")
    for entry in pack.entries:
        for index in range(entry.count):
            seed = derive_seed(args.seed, provider=__package__, cue=entry.cue, filename=entry.filename, variation=index + 1)
            audio = _render(entry.options, seed)
            path = engine.write_wav(out_dir / pack.filename(entry, index), audio)
            artifacts.append(
                _artifact(path, out_dir, audio, entry.options, seed, index + 1)
            )
            print(f"  wrote {path.relative_to(out_dir)}  {audio.duration:.3f}s")
    manifest = _write_manifest(
        out_dir / "manifest.json", artifacts, command="pack", pack=pack.name
    )
    print(f"  {len(artifacts)} files; manifest {manifest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen doors",
        description="Render game-facing packs from bundled CC0 door Foley.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list game packs")
    listing.set_defaults(func=cmd_list)
    pack = sub.add_parser("pack", help="render a named game pack")
    pack.add_argument("pack", help="pack name; see `list`")
    pack.add_argument("--seed", type=int, default=DEFAULT_SEED)
    pack.add_argument("--out", type=Path, default=DEFAULT_OUT)
    pack.set_defaults(func=cmd_pack)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original one-file CLI. Only the registry verbs opt into the
    # pack-aware parser, so existing invocations remain valid.
    if not arguments or arguments[0] not in {"list", "pack"}:
        return engine.main(arguments)
    args = build_parser().parse_args(arguments)
    try:
        return args.func(args) or 0
    except (OSError, ValueError) as error:
        print(f"soundgen doors: {error}", file=sys.stderr)
        return 1
