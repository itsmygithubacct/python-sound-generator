"""Registry-driven command line for vehicle foley SFX."""

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

from . import engine
from .cues import (
    CUES,
    GROUPS,
    PACKS,
    Options,
    render_with_sources,
    stem_for,
)


DEFAULT_OUT = Path("output") / "vehicles"
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
        vehicle=engine.vehicle_named(args.vehicle),
        surface=engine.surface_named(args.surface),
        speed=args.speed,
        duration=args.duration,
        intensity=args.intensity,
    )


def _options_dict(options: Options, cue_name: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "vehicle": options.vehicle.name,
        "surface": options.surface.name,
        "speed": options.speed,
        "duration": options.duration,
        "intensity": options.intensity,
    }
    honoured = {
        name.strip() for name in CUES[cue_name].honours if name.strip()
    }
    return {name: values[name] for name in values if name in honoured}


def _artifact(
    path: Path,
    signal: list[float],
    cue_name: str,
    seed: int,
    variation: int,
    options: Options,
    sources: tuple[str, ...] = (),
) -> dict[str, Any]:
    metrics = artifact_metrics(path, signal, engine.SR)
    artifact = {
        "file": path.name,
        "cue": cue_name,
        "loop": CUES[cue_name].loop,
        "seed": seed,
        "variation": variation,
        "options": _options_dict(options, cue_name),
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
    }
    if sources:
        artifact["renderer"] = "crash_generator"
        artifact["sources"] = list(sources)
    return artifact


def _write_manifest(
    path: Path,
    artifacts: list[dict[str, Any]],
    *,
    command: str,
    pack: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "generator": "vehicle_foley_generator",
        "command": command,
        "artifacts": artifacts,
    }
    if any("sources" in artifact for artifact in artifacts):
        payload["delegated_provenance"] = {
            "projectile_sounds": "projectile_sounds/sources/provenance.json",
            "stone_noise_generator": (
                "stone_noise_generator/sources/provenance.json"
            ),
            "wood_noise_generator": (
                "wood_noise_generator/sources/provenance.json"
            ),
            "aircraft_spacecraft_generator": (
                "aircraft_spacecraft_generator/sources/provenance.json"
            ),
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
    selected = any((args.cues, args.vehicles, args.surfaces, args.packs))
    if args.cues or not selected:
        for group in GROUPS:
            _header(f"{group} cues")
            for cue in CUES.values():
                if cue.group != group:
                    continue
                _item(cue.name, cue.description)
                detail = cue.honours or "no options"
                if cue.loop:
                    detail = f"{detail}; seamless loop"
                print(f"  {'':20} honours: {detail}")
        print()
    if args.vehicles or not selected:
        _header("vehicles")
        for profile in engine.VEHICLES.values():
            _item(profile.name, profile.description)
        print()
    if args.surfaces or not selected:
        _header("surfaces")
        for surface in engine.SURFACES.values():
            _item(surface.name, surface.description)
        print()
    if args.packs or not selected:
        _header("packs")
        for pack in PACKS.values():
            count = sum(entry.count for entry in pack.entries)
            _item(pack.name, f"{pack.description} ({count} files)")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    vehicle = engine.vehicle_named(args.vehicle)
    surface = engine.surface_named(args.surface)

    _header(f"vehicle: {vehicle.label}")
    print(f"  {vehicle.description}")
    print(f"  mass {vehicle.mass:.2f}; cavity {vehicle.cavity_frequency:.0f} Hz")
    print(f"  body {vehicle.body_frequency:.0f} Hz; "
          f"suspension {vehicle.suspension_frequency:.0f} Hz; "
          f"squeal {vehicle.squeal_frequency:.0f} Hz; "
          f"{len(vehicle.panel_modes)} panel modes")
    _header(f"surface: {surface.label}")
    print(f"  {surface.description}")
    print(f"  kind {surface.kind}; roll {surface.roll_low:.0f}-"
          f"{surface.roll_high:.0f} Hz; damping {surface.damping:.2f}")
    print(f"  grip {surface.grip:.2f}; wetness {surface.wetness:.2f}; "
          f"{surface.grain_rate:.0f} grains/s at full speed")
    if args.cue:
        if args.cue not in CUES:
            print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
            return 2
        cue = CUES[args.cue]
        print()
        _header(f"cue: {cue.name}")
        print(f"  {cue.description}")
        print(f"  group {cue.group}; peak {cue.peak_db:.1f} dBFS; "
              f"{'seamless loop' if cue.loop else 'one-shot'}; "
              f"honours {cue.honours or 'no options'}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    if args.count < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    if not 0.05 <= args.speed <= 1.0:
        print("--speed must be between 0.05 and 1", file=sys.stderr)
        return 2
    if not 0.05 <= args.intensity <= 1.0:
        print("--intensity must be between 0.05 and 1", file=sys.stderr)
        return 2
    if not engine.MIN_LOOP <= args.duration <= engine.MAX_LOOP:
        print(
            f"--duration must be between {engine.MIN_LOOP} and {engine.MAX_LOOP}",
            file=sys.stderr,
        )
        return 2

    options = _options_from_args(args)
    out_dir = Path(args.out)
    stem = args.stem or stem_for(args.cue, options)
    artifacts: list[dict[str, Any]] = []
    last_path: Path | None = None
    for index in range(args.count):
        seed = derive_seed(args.seed, provider=__package__, cue=args.cue, variation=index + 1)
        result = render_with_sources(args.cue, options, random.Random(seed))
        signal = result.samples
        filename = f"{stem}.wav" if index == 0 else f"{stem}_v{index + 1:02d}.wav"
        last_path = engine.write_wav(out_dir / filename, signal)
        artifacts.append(_artifact(
            last_path, signal, args.cue, seed, index + 1, options,
            result.sources,
        ))
        loop_note = " loop" if CUES[args.cue].loop else ""
        print(f"  wrote {last_path}  {len(signal) / engine.SR:.2f}s{loop_note}")
    manifest = _write_manifest(out_dir / "manifest.json", artifacts,
                               command="render")
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
            result = render_with_sources(
                entry.cue, entry.options, random.Random(seed)
            )
            signal = result.samples
            path = engine.write_wav(out_dir / pack.filename(entry, index), signal)
            artifacts.append(_artifact(
                path, signal, entry.cue, seed, index + 1, entry.options,
                result.sources,
            ))
            print(f"  wrote {path.name}  {len(signal) / engine.SR:.2f}s")
            if args.play:
                _play(path)
    manifest = _write_manifest(
        out_dir / "manifest.json", artifacts, command="pack", pack=pack.name
    )
    print(f"  {len(artifacts)} files; manifest {manifest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen vehicles",
        description=(
            "Generate tire roll, skids, spray, suspension, driveline, jumps, "
            "with dependency-free procedural DSP; impacts delegate to "
            "crash_generator."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<verb>")

    list_parser = sub.add_parser("list", help="show registries and packs")
    list_parser.add_argument("--cues", action="store_true")
    list_parser.add_argument("--vehicles", action="store_true")
    list_parser.add_argument("--surfaces", action="store_true")
    list_parser.add_argument("--packs", action="store_true")
    list_parser.set_defaults(func=cmd_list)

    info_parser = sub.add_parser("info", help="describe the selected models")
    info_parser.add_argument("--cue", help="also describe a cue")
    info_parser.set_defaults(func=cmd_info)

    render_parser = sub.add_parser("render", help="render one cue")
    render_parser.add_argument("cue", help="cue name; see `list --cues`")
    render_parser.add_argument("--count", type=int, default=1)
    render_parser.add_argument("--stem", help="override the generated filename stem")
    render_parser.set_defaults(func=cmd_render)

    pack_parser = sub.add_parser("pack", help="render a named artifact bank")
    pack_parser.add_argument("pack", help="pack name; see `list --packs`")
    pack_parser.set_defaults(func=cmd_pack)

    for target in (info_parser, render_parser):
        target.add_argument("--vehicle", default="sedan", choices=engine.VEHICLES)
        target.add_argument("--surface", default="asphalt", choices=engine.SURFACES)
    for target in (render_parser,):
        target.add_argument(
            "--speed", type=float, default=0.6,
            help="0.05..1 as a fraction of the vehicle's working speed",
        )
        target.add_argument(
            "--duration", type=float, default=2.5,
            help="loop length in seconds; loop cues only",
        )
        target.add_argument("--intensity", type=float, default=0.85)
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
        print(f"soundgen vehicles: {error}", file=sys.stderr)
        return 1
