#!/usr/bin/env python3
"""Combat SFX generator. Pure Python, standard library only.

    soundgen combat list                      what can be made
    soundgen combat info --weapon greatsword  how a weapon's numbers sound
    soundgen combat render full --weapon axe --target plate --play
    soundgen combat pack board-duel --out ./out/board-duel

This provider uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ..._audio.runtime import derive_seed
from ..._audio.runtime import play as play_audio
from .lib import cues as cue_lib
from .lib import display as d
from ..._audio import dsp
from .lib.materials import MATERIALS, TARGETS
from .lib.weapons import BLADED, LIMBS, WEAPONS

DEFAULT_OUT = Path("output") / "combat"
# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def build_options(args: argparse.Namespace) -> cue_lib.Options:
    return cue_lib.Options(
        weapon=cue_lib.weapon_named(args.weapon),
        target=cue_lib.target_named(args.target),
        surface=args.surface,
        armored=not args.no_armor,
    )


def stem_for(cue: cue_lib.Cue, opts: cue_lib.Options) -> str:
    """Name the file after whatever actually varies it."""
    if "target" in cue.honours:
        return f"{cue.name}_{opts.weapon.name}_{opts.target.name}"
    if "weapon" in cue.honours:
        return f"{cue.name}_{opts.weapon.name}"
    if "surface" in cue.honours:
        tail = "_armored" if opts.armored else ""
        return f"{cue.name}_{opts.surface}{tail}"
    return cue.name


def play(path: Path) -> None:
    if not play_audio(path):
        d.warn("no audio player found (tried paplay, aplay, afplay, play, ffplay)")


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    show_all = not (args.weapons or args.targets or args.cues or args.packs
                    or args.materials)
    if args.cues or show_all:
        for group in cue_lib.GROUPS:
            d.header(f"Cues: {group}")
            for cue in cue_lib.CUES.values():
                if cue.group == group:
                    d.row(cue.name, cue.description, 17)
                    if cue.honours:
                        print(f"  {' ' * 17} {d.C.DIM}honours: "
                              f"{cue.honours}{d.C.RESET}")
    if args.weapons or show_all:
        d.header("Weapons")
        for w in BLADED.values():
            d.row(w.name, f"{w.description}  "
                          f"({w.length:.2f} m, {w.mass:.2f} kg, "
                          f"{w.material.name})", 17)
        # Limbs are Weapons and --weapon takes them, but their mass means
        # something else (see lib/weapons.py), so do not print it as if it
        # were a greatsword's.
        d.header("Unarmed")
        for w in LIMBS.values():
            d.row(w.name, f"{w.description}  "
                          f"({w.mass:.2f} kg effective)", 17)
    if args.targets or show_all:
        d.header("Targets")
        for t in TARGETS.values():
            d.row(t.name, t.description, 17)
    if args.materials or show_all:
        d.header("Materials")
        for m in MATERIALS.values():
            d.row(m.name, f"{m.description}  "
                          f"(wave speed {m.wave_speed:.0f} m/s)", 17)
    if args.packs or show_all:
        d.header("Packs")
        for p in cue_lib.PACKS.values():
            total = sum(e.count for e in p.entries)
            d.row(p.name, f"{p.description}  ({total} files)", 17)
    print()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    weapon = cue_lib.weapon_named(args.weapon)
    d.header(f"{weapon.name} - {weapon.description}")
    for label, value in weapon.describe().items():
        d.item(label, value)
    d.header("Ring modes vs the reference sword")
    target = cue_lib.target_named(args.target)
    for (f0, _, _), (f1, d1, _) in zip(target.modes,
                                       weapon.scaled_modes(target.modes)):
        print(f"  {f0:8.0f} Hz -> {f1:8.0f} Hz   decay {d1 * 1000:5.1f} ms")
    print()
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if args.cue not in cue_lib.CUES:
        d.error(f"unknown cue {args.cue!r}")
        d.warn(f"try one of: {', '.join(cue_lib.CUES)}")
        return 2
    opts = build_options(args)
    cue = cue_lib.CUES[args.cue]
    stem = stem_for(cue, opts)
    d.header(f"{cue.name}: {cue.description}")
    for i in range(args.count):
        seed = derive_seed(
            args.seed,
            provider=__package__,
            cue=args.cue,
            filename=stem,
            variation=i + 1,
        )
        rng = random.Random(seed)
        signal = cue_lib.render(args.cue, opts, rng)
        name = (f"{stem}.wav" if args.count == 1
                else f"{stem}_{i + 1:02d}.wav")
        path = args.out / name
        dsp.write_wav(path, signal)
        d.wrote(str(path), len(signal) / dsp.SR)
        if args.play:
            play(path)
    print()
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.pack not in cue_lib.PACKS:
        d.error(f"unknown pack {args.pack!r}")
        d.warn(f"try one of: {', '.join(cue_lib.PACKS)}")
        return 2
    pack = cue_lib.PACKS[args.pack]
    out_dir = args.out if args.out != DEFAULT_OUT else DEFAULT_OUT / pack.name
    d.header(f"pack {pack.name}: {pack.description}")
    count = 0
    for entry in pack.entries:
        for i in range(entry.count):
            seed = derive_seed(
                args.seed,
                provider=__package__,
                cue=entry.cue,
                filename=entry.filename,
                variation=i + 1,
            )
            rng = random.Random(seed)
            signal = cue_lib.render(entry.cue, entry.opts, rng)
            path = out_dir / pack.filename(entry, i)
            dsp.write_wav(path, signal)
            d.wrote(str(path), len(signal) / dsp.SR)
            count += 1
            if args.play:
                play(path)
    d.info(f"{count} files -> {out_dir}")
    print()
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="soundgen combat",
        description="Generate combat SFX with pure Python DSP. "
                    "No dependencies.",
        epilog="examples:\n"
               "  soundgen combat list\n"
               "  soundgen combat info --weapon greatsword\n"
               "  soundgen combat render full --weapon axe --target plate "
               "--play\n"
               "  soundgen combat render strike --target mail --count 5\n"
               "  soundgen combat pack board-duel\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show cues, weapons, targets, packs")
    p_list.add_argument("--cues", action="store_true")
    p_list.add_argument("--weapons", action="store_true")
    p_list.add_argument("--targets", action="store_true")
    p_list.add_argument("--materials", action="store_true")
    p_list.add_argument("--packs", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="show a weapon's derived acoustics")
    p_info.add_argument("--weapon", default="sword")
    p_info.add_argument("--target", default="blade")
    p_info.set_defaults(func=cmd_info)

    p_render = sub.add_parser("render", help="render one cue")
    p_render.add_argument("cue", help="cue name; see `list`")
    p_render.set_defaults(func=cmd_render)

    p_pack = sub.add_parser("pack", help="render a named set of artifacts")
    p_pack.add_argument("pack", help="pack name; see `list --packs`")
    p_pack.set_defaults(func=cmd_pack)

    for p in (p_render, p_pack):
        p.add_argument("--seed", type=int, default=20260714,
                       help="base seed; renders are reproducible")
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--play", action="store_true",
                       help="play each file after writing it")
    for p in (p_render,):
        p.add_argument("--count", type=int, default=1,
                       help="variations, each on its own seed")
        p.add_argument("--weapon", default="sword",
                       help=f"one of: {', '.join(WEAPONS)}")
        p.add_argument("--target", default="blade",
                       help=f"one of: {', '.join(TARGETS)}")
        p.add_argument("--surface", default="stone",
                       help="footstep surface: stone, dirt, wood")
        p.add_argument("--no-armor", action="store_true",
                       help="drop the mail layer from footsteps and falls")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "count", 1) < 1:
        d.error("--count must be at least 1")
        return 2
    try:
        return args.func(args)
    except ValueError as exc:
        d.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
