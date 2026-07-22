"""Public command line for the modular unified sound generator."""

from __future__ import annotations

import difflib
import json
import sys
from collections.abc import Sequence

from . import __version__
from .providers import PROVIDERS, Provider, provider_named, provider_names


def _print_help(stream: object | None = None) -> None:
    stream = sys.stdout if stream is None else stream
    print(
        "usage: soundgen COMMAND [ARGS]\n\n"
        "Unified procedural and recording-derived sound generation.\n\n"
        "commands:\n"
        "  list [--json]                 list sound-domain providers\n"
        "  info PROVIDER [--json]        describe one provider\n"
        "  doctor [--json]               import and validate every provider\n"
        "  recipe validate|render ...    validate or render a game recipe\n"
        "  run PROVIDER [ARGS...]        run a provider CLI\n"
        "  PROVIDER [ARGS...]            shorthand for `run PROVIDER`\n\n"
        "examples:\n"
        "  soundgen list\n"
        "  soundgen fire list\n"
        "  soundgen fire render lava_flow --seconds 8 --out output/lava\n"
        "  soundgen car-engine engine.wav --duration 3 --seed 7\n"
        "  soundgen recipe validate\n"
        "  soundgen recipe render cyber-runner --cue ui.cursor --out output/recipes\n"
        "  soundgen doctor\n",
        file=stream,
    )


def _wants_json(args: Sequence[str]) -> bool:
    return len(args) == 1 and args[0] == "--json"


def _cmd_list(args: Sequence[str]) -> int:
    if args and not _wants_json(args):
        print("soundgen list accepts only --json", file=sys.stderr)
        return 2
    if _wants_json(args):
        rows = [provider.as_dict() for provider in PROVIDERS]
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(provider.slug) for provider in PROVIDERS)
    for provider in PROVIDERS:
        print(f"{provider.slug:<{width}}  {provider.description}")
    print(f"\n{len(PROVIDERS)} providers")
    return 0


def _resolve(name: str) -> Provider | None:
    try:
        return provider_named(name)
    except KeyError:
        choices = difflib.get_close_matches(
            name.replace("_", "-"), provider_names(), n=3, cutoff=0.45
        )
        suggestion = f"; did you mean {', '.join(choices)}?" if choices else ""
        print(f"unknown provider {name!r}{suggestion}", file=sys.stderr)
        return None


def _cmd_info(args: Sequence[str]) -> int:
    json_mode = "--json" in args
    names = [arg for arg in args if arg != "--json"]
    if len(names) != 1 or len(args) != len(names) + int(json_mode):
        print("usage: soundgen info PROVIDER [--json]", file=sys.stderr)
        return 2
    provider = _resolve(names[0])
    if provider is None:
        return 2
    payload = provider.as_dict()
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"name:        {payload['name']}")
        print(f"module:      {payload['module']}")
        print(f"description: {payload['description']}")
        print(f"dependency:  {payload['dependency']}")
        print(f"audio files: {payload['audio_files']}")
        print(f"data files:  {payload['asset_files']}")
        print(f"data size:   {_format_bytes(int(payload['asset_bytes']))}")
        print(f"provenance:  {'yes' if payload['provenance'] else 'not applicable'}")
        if provider.aliases:
            print(f"aliases:     {', '.join(provider.aliases)}")
    return 0


def _doctor_row(provider: Provider) -> dict[str, object]:
    row: dict[str, object] = {
        "name": provider.slug,
        "module": provider.package_name,
        "description": provider.description,
        "aliases": list(provider.aliases),
        "dependency": provider.dependency,
        "source_files": 0,
        "audio_files": 0,
        "asset_files": 0,
        "asset_bytes": 0,
        "provenance": False,
        "provenance_files": 0,
        "missing_runtime_files": [],
    }
    errors: list[str] = []
    try:
        row.update(provider.as_dict())
    except Exception as error:
        errors.append(f"metadata {type(error).__name__}: {error}")
    try:
        errors.extend(provider.validation_errors())
    except Exception as error:
        errors.append(f"validation {type(error).__name__}: {error}")
    try:
        provider.load()
    except Exception as error:  # doctor reports all providers in one run
        errors.append(f"import {type(error).__name__}: {error}")
    row["status"] = "error" if errors else "ok"
    if errors:
        row["error"] = "; ".join(errors)
    return row


def _cmd_doctor(args: Sequence[str]) -> int:
    if args and not _wants_json(args):
        print("soundgen doctor accepts only --json", file=sys.stderr)
        return 2
    rows = [_doctor_row(provider) for provider in PROVIDERS]
    failed = [row for row in rows if row["status"] != "ok"]
    if _wants_json(args):
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        width = max(len(provider.slug) for provider in PROVIDERS)
        for row in rows:
            detail = row.get("error", f"{row['audio_files']} audio files")
            print(f"{row['name']:<{width}}  {row['status']:<5}  {detail}")
        print(f"\n{len(rows) - len(failed)}/{len(rows)} providers import cleanly")
    return 1 if failed else 0


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def _dispatch(name: str, args: Sequence[str]) -> int:
    provider = _resolve(name)
    if provider is None:
        return 2
    try:
        return provider.invoke(args)
    except ModuleNotFoundError as error:
        print(
            f"cannot load {provider.slug}: missing Python module {error.name!r}; "
            "install the project requirements",
            file=sys.stderr,
        )
        return 1
    except Exception as error:
        print(
            f"cannot load {provider.slug}: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if args[0] in {"-V", "--version"}:
        print(__version__)
        return 0
    command, tail = args[0], args[1:]
    if command in {"list", "providers"}:
        return _cmd_list(tail)
    if command == "info":
        return _cmd_info(tail)
    if command == "doctor":
        return _cmd_doctor(tail)
    if command == "recipe":
        from .recipes import recipe_main

        return recipe_main(tail)
    if command == "run":
        if not tail:
            print("usage: soundgen run PROVIDER [ARGS...]", file=sys.stderr)
            return 2
        return _dispatch(tail[0], tail[1:])
    return _dispatch(command, tail)


__all__ = ["main"]
