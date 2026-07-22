#!/usr/bin/env python3
"""Sword strike renders, kept working after the move to lib/ and combat_sfx.py.

This used to be the whole project: a single file with its own DSP. It now
delegates to lib/ so there is one implementation of the filters rather than
two copies drifting apart. The CLI and the output are unchanged -- the four
variants below are byte-identical to what this script produced before, which
tests/test_render.py pins with md5 goldens.

For anything beyond these four, use combat_sfx.py:

    soundgen combat list
    soundgen combat render full --weapon greatsword --target plate
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from ..._audio import dsp
from ..._audio.runtime import legacy_seed
from .lib import strikes
from .lib.materials import BLADE, PLATE
from .lib.weapons import SWORD

DEFAULT_OUT = Path("output") / "combat"

# variant -> (peak dB, renderer). Frozen to preserve the approved renders.
VARIANTS = {
    "slash": (-10.5, lambda rng: strikes.swing(SWORD, rng, 0.36, 0.20)),
    "clash": (-9.0, lambda rng: strikes.strike(SWORD, BLADE, rng)),
    "armor": (-9.5, lambda rng: strikes.strike(SWORD, PLATE, rng)),
    "full": (-9.0, lambda rng: strikes.full_strike(SWORD, BLADE, rng)),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Render the original four sword strikes. "
                    "See combat_sfx.py for the full generator.")
    ap.add_argument("--variant", default="all", choices=[*VARIANTS, "all"])
    ap.add_argument("--count", type=int, default=1,
                    help="variations per variant, each with its own seed")
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    names = list(VARIANTS) if args.variant == "all" else [args.variant]

    for name in names:
        peak_db, render = VARIANTS[name]
        for i in range(args.count):
            seed = legacy_seed(args.seed, index=i, name=name, stride=1000)
            rng = random.Random(seed)
            signal = dsp.master(render(rng), peak_db)
            stem = (f"sword_{name}" if args.count == 1
                    else f"sword_{name}_{i + 1:02d}")
            path = args.out / f"{stem}.wav"
            dsp.write_wav(path, signal)
            print(f"wrote {path}  {len(signal) / dsp.SR:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
