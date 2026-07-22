"""Registry-driven command line for aircraft and spacecraft SFX."""

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

from . import engine
from .cues import (
    CUES,
    GROUPS,
    LOOPS,
    PACKS,
    Options,
    foley_available,
    render_with_sources,
    renderer_for,
    stem_for,
)


DEFAULT_OUT = Path("output") / "aircraft"
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
        craft=engine.craft_named(args.craft),
        powerplant=engine.powerplant_named(args.powerplant),
        power=args.power,
        speed=args.speed,
        duration=args.duration,
        intensity=args.intensity,
    )


def _options_dict(options: Options, cue_name: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "craft": options.craft.name,
        "powerplant": options.powerplant.name,
        "power": options.power,
        "speed": options.speed,
        "duration": options.duration,
        "intensity": options.intensity,
    }
    honoured = {
        name.strip()
        for name in CUES[cue_name].honours if name.strip()
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
    artifact: dict[str, Any] = {
        "file": path.name,
        "cue": cue_name,
        "loop": CUES[cue_name].loop,
        "renderer": renderer_for(cue_name),
        "seed": seed,
        "variation": variation,
        "options": _options_dict(options, cue_name),
        "duration_seconds": metrics['duration_seconds'],
        "peak_dbfs": metrics['peak_dbfs'],
        "rms_dbfs": metrics['rms_dbfs'],
        "sha256": metrics['sha256'],
    }
    if CUES[cue_name].foley:
        # A file built from recordings should say so, and say what licences it
        # inherits, without anyone having to read the code to find out.
        artifact["sources"] = list(sources)
    if CUES[cue_name].loop:
        # A loop's manifest should say what its loop point measures, not just
        # claim it has one.
        measured = loopdsp.seam(signal)
        artifact["seam"] = {
            "step_ratio": round(measured.step_ratio, 6),
            "level_ratio": round(measured.level_ratio, 6),
            "seamless": _seamless(cue_name, measured),
        }
    return artifact


def _seamless(cue_name: str, measured: loopdsp.Seam) -> bool:
    """Judge a loop point, holding rhythmic cues only to the step check.

    See :class:`aircraft_spacecraft_generator.cues.Cue` for why the level check
    does not apply to a klaxon.
    """

    if CUES[cue_name].stationary:
        return measured.is_seamless()
    return measured.step_ratio <= loopdsp.MAX_STEP_RATIO


def _write_manifest(
    path: Path,
    artifacts: list[dict[str, Any]],
    *,
    command: str,
    pack: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "generator": "aircraft_spacecraft_generator",
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
    selected = any((args.cues, args.craft, args.powerplants, args.packs))
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
                if cue.foley:
                    detail = f"{detail}; delegated to crash_generator"
                print(f"  {'':20} honours: {detail}")
        if not foley_available():
            print("  note: the delegated impact cues cannot render here; "
                  "pip install -r crash_generator/requirements.txt")
        print()
    if args.craft or not selected:
        _header("craft")
        for profile in engine.CRAFT.values():
            _item(profile.name, profile.description)
        print()
    if args.powerplants or not selected:
        _header("powerplants")
        for plant in engine.POWERPLANTS.values():
            _item(plant.name, f"{plant.description} [{plant.kind}]")
        print()
    if args.packs or not selected:
        _header("packs")
        for pack in PACKS.values():
            count = sum(entry.count for entry in pack.entries)
            _item(pack.name, f"{pack.description} ({count} files)")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    craft = engine.craft_named(args.craft)
    plant = engine.powerplant_named(args.powerplant)

    _header(f"craft: {craft.label}")
    print(f"  {craft.description}")
    print(f"  mass {craft.mass:.2f}; skin {craft.skin_frequency:.0f} Hz; "
          f"cabin {craft.cabin_frequency:.0f} Hz; "
          f"strut {craft.strut_frequency:.0f} Hz")
    print(f"  airflow {craft.wind_low:.0f}-{craft.wind_high:.0f} Hz at "
          f"coupling {craft.wind_coupling:.2f}; damping {craft.damping:.2f}; "
          f"{craft.gear} gear")
    print(f"  annunciator {craft.horn_frequency:.0f} Hz; "
          f"{len(craft.panel_modes)} panel modes")

    _header(f"powerplant: {plant.label}")
    print(f"  {plant.description}")
    print(f"  kind {plant.kind}; idles at {plant.idle:.0%} of full rate; "
          f"spins up in {plant.spin_up:.1f} s")
    if plant.kind == "piston":
        fire = plant.shaft_rate * plant.cylinders * 0.5
        print(f"  {plant.cylinders} cylinders firing at {fire:.0f} Hz at full "
              f"power; exhaust {plant.exhaust_frequency:.0f} Hz")
    if plant.spool_frequency > 0.0:
        print(f"  core {plant.spool_frequency:.0f} rev/s at full power over "
              f"{plant.stages} stages")
    if plant.blades > 0:
        blade_pass = plant.shaft_rate * plant.blades
        kind = "open propeller" if plant.open_prop else "ducted fan"
        print(f"  {plant.blades}-blade {kind} at {plant.shaft_rate:.0f} rev/s; "
              f"blade-pass {blade_pass:.0f} Hz at full power")
    if plant.chamber_frequency > 0.0:
        print(f"  chamber {plant.chamber_frequency:.0f} Hz"
              + (f"; chugs at {plant.chug_rate:.0f} Hz"
                 if plant.chug_rate > 0.0 else ""))
    print(f"  tone {plant.tone_gain:.2f} against noise {plant.noise_gain:.2f}; "
          f"roughness {plant.roughness:.2f}")

    if args.cue:
        if args.cue not in CUES:
            print(f"unknown cue {args.cue!r}; see `list --cues`",
                  file=sys.stderr)
            return 2
        cue = CUES[args.cue]
        print()
        _header(f"cue: {cue.name}")
        print(f"  {cue.description}")
        print(f"  group {cue.group}; peak {cue.peak_db:.1f} dBFS; "
              f"{'seamless loop' if cue.loop else 'one-shot'}; "
              f"honours {cue.honours or 'no options'}")
        print(f"  rendered: {renderer_for(cue.name)}"
              + ("; see sources/provenance.json" if cue.foley else
                 "; standard library only"))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in CUES:
        print(f"unknown cue {args.cue!r}; see `list --cues`", file=sys.stderr)
        return 2
    if args.count < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    for name, value in (("power", args.power), ("speed", args.speed),
                        ("intensity", args.intensity)):
        if not 0.05 <= value <= 1.0:
            print(f"--{name} must be between 0.05 and 1", file=sys.stderr)
            return 2
    if not engine.MIN_LOOP <= args.duration <= engine.MAX_LOOP:
        print(
            f"--duration must be between {engine.MIN_LOOP} and "
            f"{engine.MAX_LOOP}",
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
        filename = (
            f"{stem}.wav" if index == 0 else f"{stem}_v{index + 1:02d}.wav"
        )
        last_path = engine.write_wav(out_dir / filename, signal)
        artifacts.append(
            _artifact(
                last_path, signal, args.cue, seed, index + 1, options,
                result.sources,
            )
        )
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
            result = render_with_sources(
                entry.cue, entry.options, random.Random(seed)
            )
            signal = result.samples
            path = engine.write_wav(
                out_dir / pack.filename(entry, index), signal
            )
            artifacts.append(
                _artifact(
                    path, signal, entry.cue, seed, index + 1, entry.options,
                    result.sources,
                )
            )
            print(f"  wrote {path.name}  {len(signal) / engine.SR:.2f}s")
            if args.play:
                _play(path)
    manifest = _write_manifest(
        out_dir / "manifest.json", artifacts, command="pack", pack=pack.name
    )
    print(f"  {len(artifacts)} files; manifest {manifest}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Render loop cues and report what their loop points measure.

    "It loops" is a claim that can be checked, and a generator that makes the
    claim should be able to show its work.
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
    print(f"{'cue':12s} {'step':>8s} {'level':>8s}  result")
    failed = 0
    for name in names:
        signal = render_with_sources(
            name, options, random.Random(args.seed)
        ).samples
        measured = loopdsp.seam(signal)
        ok = _seamless(name, measured)
        failed += 0 if ok else 1
        note = "" if CUES[name].stationary else "  (rhythmic; step only)"
        print(f"{name:12s} {measured.step_ratio:8.4f} "
              f"{measured.level_ratio:8.3f}  "
              f"{'seamless' if ok else 'NOT SEAMLESS'}{note}")
    print(f"\n{len(names) - failed}/{len(names)} seamless "
          f"(step <= {loopdsp.MAX_STEP_RATIO:g}, level within "
          f"{1 / loopdsp.MAX_LEVEL_DRIFT:g}-{loopdsp.MAX_LEVEL_DRIFT:g})")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen aircraft",
        description=(
            "Generate piston, turbine, rocket and thruster loops, propeller "
            "blade-pass, airframe and cockpit vibration, speed-dependent "
            "wind and alarms with dependency-free procedural DSP; touchdown "
            "and crash compatibility delegates use crash_generator."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<verb>")

    list_parser = sub.add_parser("list", help="show registries and packs")
    list_parser.add_argument("--cues", action="store_true")
    list_parser.add_argument("--craft", action="store_true")
    list_parser.add_argument("--powerplants", action="store_true")
    list_parser.add_argument("--packs", action="store_true")
    list_parser.set_defaults(func=cmd_list)

    info_parser = sub.add_parser("info", help="describe the selected models")
    info_parser.add_argument("--cue", help="also describe a cue")
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

    # `list` is the only verb that names no model, so every other one can pick.
    for target in (info_parser, render_parser, verify_parser):
        target.add_argument("--craft", default="trainer", choices=engine.CRAFT)
        target.add_argument("--powerplant", default="flat_four",
                            choices=engine.POWERPLANTS)
    for target in (render_parser, verify_parser):
        target.add_argument(
            "--power", type=float, default=0.6,
            help="0.05..1 throttle setting, from idle to full power",
        )
        target.add_argument(
            "--speed", type=float, default=0.5,
            help="0.05..1 airspeed as a fraction of the craft's envelope",
        )
        target.add_argument(
            "--duration", type=float, default=2.5,
            help="loop length in seconds; loop cues only",
        )
        target.add_argument("--intensity", type=float, default=0.85)
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
        print(f"soundgen aircraft: {error}", file=sys.stderr)
        return 1
