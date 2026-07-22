"""Unified facade for procedural and recording-driven car engines."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from . import car_engine_generator as procedural


SAMPLE_BANK = Path(__file__).resolve().parent / "sample_banks" / "mini_cooper_s"
DEFAULT_SAMPLE_OUTPUT = Path("output") / "car-engine" / "mini_cooper_s.wav"


def _print_help(stream: TextIO | None = None) -> None:
    stream = sys.stdout if stream is None else stream
    print(
        "usage: soundgen car-engine COMMAND [ARGS]\n\n"
        "Procedural and recording-driven combustion-engine performances.\n\n"
        "commands:\n"
        "  procedural [ARGS]       synthesize an engine (legacy arguments)\n"
        "  sample [OUTPUT] [ARGS]  render with the bundled CC0 sample bank\n"
        "  inspect [BANK]          show sample-bank anchors and provenance\n"
        "  analyze RECORDING ...   estimate firing frequency and RPM\n"
        "  render-bank BANK OUTPUT ...\n"
        "                           render with another compatible bank\n\n"
        "Compatibility: arguments without a command still run the procedural "
        "renderer.\n"
        "Use `soundgen car-engine COMMAND --help` for renderer options.",
        file=stream,
    )


def _hybrid_main(args: list[str]) -> int:
    from .hybrid import main

    return main(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if args and args[0] == "procedural":
        return procedural.main(args[1:])
    if args and args[0] == "sample":
        tail = args[1:]
        if tail and not tail[0].startswith("-"):
            output, options = tail[0], tail[1:]
        else:
            output, options = str(DEFAULT_SAMPLE_OUTPUT), tail
        return _hybrid_main(["render", str(SAMPLE_BANK), output, *options])
    if args and args[0] == "inspect":
        bank = args[1:] or [str(SAMPLE_BANK)]
        return _hybrid_main(["inspect", *bank])
    if args and args[0] == "analyze":
        return _hybrid_main(args)
    if args and args[0] == "render-bank":
        return _hybrid_main(["render", *args[1:]])
    return procedural.main(args)


__all__ = ["DEFAULT_SAMPLE_OUTPUT", "SAMPLE_BANK", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
