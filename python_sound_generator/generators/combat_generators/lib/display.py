"""Display utilities: colors and formatted print helpers.

Mirrors birdsong_generators/lib/display.py so the two CLIs look alike. Colour
is dropped automatically when stdout is not a terminal, so piping to a file or
a log stays readable.
"""

from __future__ import annotations

import sys


class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


def _plain() -> None:
    for name in dir(C):
        if not name.startswith("_"):
            setattr(C, name, "")


if not sys.stdout.isatty():
    _plain()


def header(text: str) -> None:
    print(f"\n{C.BOLD}{C.CYAN}{text}{C.RESET}")


def info(text: str) -> None:
    print(f"  {C.GREEN}{text}{C.RESET}")


def warn(text: str) -> None:
    print(f"  {C.YELLOW}{text}{C.RESET}")


def error(text: str) -> None:
    print(f"  {C.RED}{text}{C.RESET}")


def item(label: str, value: str) -> None:
    print(f"  {C.BOLD}{label}{C.RESET}: {value}")


def row(name: str, description: str, width: int = 16) -> None:
    print(f"  {C.BOLD}{C.GREEN}{name:{width}s}{C.RESET} "
          f"{C.DIM}{description}{C.RESET}")


def wrote(path: str, seconds: float) -> None:
    print(f"  {C.GREEN}wrote{C.RESET} {path} {C.DIM}{seconds:.2f}s{C.RESET}")
