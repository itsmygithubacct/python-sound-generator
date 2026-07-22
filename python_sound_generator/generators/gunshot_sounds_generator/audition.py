"""Interactive, resumable yes/no audition for item 15.

The runner renders and plays one deterministic artifact at a time.  A case does
not advance until the listener records yes or no; replay and quit do not count as
verdicts.  Progress is checkpointed after every decision as JSON and Markdown.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from ..._audio.runtime import derive_seed, legacy_seed
from ..weapons_destruction_generator import cues as weapon_cues
from ..weapons_destruction_generator import engine as weapon_engine
from ..weapons_destruction_generator.cli import DEFAULT_SEED as WEAPONS_DEFAULT_SEED

from . import engine
from .cues import Options, render


SESSION_VERSION = 1
RESULTS_JSON = "results.json"
RESULTS_MARKDOWN = "results.md"
PLAYERS = ("paplay", "aplay", "play", "ffplay")


@dataclass(frozen=True)
class AuditionCase:
    """One stable yes/no decision in the audition queue."""

    id: str
    label: str
    criterion: str
    cue: str
    options: Options | weapon_cues.Options
    generator: str = "gunshot"
    render_seed: int | None = None

    @property
    def filename(self) -> str:
        return f"{self.id}.wav"


def _gun_options(*, weapon: str = "rifle", space: str = "outdoor",
                 suppression: str = "unsuppressed",
                 surface: str = "concrete", shots: int = 5) -> Options:
    return Options(
        weapon=engine.FIREARMS[weapon],
        space=engine.SPACES[space],
        suppression=engine.SUPPRESSIONS[suppression],
        surface=engine.SURFACES[surface],
        shots=shots,
    )


def _weapon_options(*, weapon: str = "rifle", shots: int = 5
                    ) -> weapon_cues.Options:
    return weapon_cues.Options(
        weapon=weapon_engine.FIREARMS[weapon], shots=shots
    )


AUDITION_CASES: tuple[AuditionCase, ...] = (
    # The roadmap's required gate. These are the original item 2 renders, not
    # this generator's compositions.
    AuditionCase(
        "gate_fire_rifle", "Item 2: rifle report",
        "Does the pressure event read as a convincing rifle report?",
        "fire", _weapon_options(weapon="rifle"), "weapons",
        legacy_seed(WEAPONS_DEFAULT_SEED, index=0, name="fire", stride=1000),
    ),
    AuditionCase(
        "gate_burst_smg", "Item 2: five-shot SMG burst",
        "Do all five reports and their cadence read as one convincing SMG burst?",
        "burst", _weapon_options(weapon="smg", shots=5), "weapons",
        legacy_seed(WEAPONS_DEFAULT_SEED, index=0, name="burst", stride=1000),
    ),

    # One canonical report through the full space x suppression matrix.
    *tuple(
        AuditionCase(
            f"shot_rifle_{space}_{suppression}",
            f"Rifle shot: {space}, {suppression}",
            (
                f"Does this read as a rifle fired {space}, with a genuinely "
                f"{suppression} gas/action balance?"
            ),
            "shot", _gun_options(space=space, suppression=suppression),
        )
        for space in engine.SPACES
        for suppression in engine.SUPPRESSIONS
    ),

    # The reflected field isolated, because it can hide behind the direct report.
    *tuple(
        AuditionCase(
            f"tail_rifle_{space}", f"Rifle tail only: {space}",
            f"Does the reflected field read as {space}, rather than a tapped echo?",
            "tail", _gun_options(space=space),
        )
        for space in engine.SPACES
    ),

    # Individual chaotic Foley. Casing land crosses every material library.
    AuditionCase(
        "bolt_rifle", "Rifle bolt cycle",
        "Does this read as a bolt cycling back and returning home?",
        "bolt", _gun_options(),
    ),
    AuditionCase(
        "casing_eject", "Casing ejection",
        "Does this read as a small empty casing leaving the action?",
        "casing_eject", _gun_options(),
    ),
    *tuple(
        AuditionCase(
            f"casing_land_{surface}", f"Casing landing on {surface}",
            (
                f"Does this read as an empty metal casing bouncing on {surface}, "
                "without the surface taking over?"
            ),
            "casing_land", _gun_options(surface=surface),
        )
        for surface in engine.SURFACES
    ),
    AuditionCase(
        "magazine_out", "Magazine out",
        "Does this read as a magazine catch releasing and the body clearing?",
        "magazine_out", _gun_options(),
    ),
    AuditionCase(
        "magazine_in", "Magazine in",
        "Does this read as a magazine seating and its catch engaging?",
        "magazine_in", _gun_options(),
    ),
    AuditionCase(
        "charging_handle", "Charging handle",
        "Does this read as a handle pulled back and the action released?",
        "charging_handle", _gun_options(),
    ),
    AuditionCase(
        "safety", "Safety / selector",
        "Does this read as a small selector detent, not generic dropped metal?",
        "safety", _gun_options(),
    ),
    AuditionCase(
        "trigger", "Trigger",
        "Does this read as quiet trigger linkage without a striker?",
        "trigger", _gun_options(),
    ),
    AuditionCase(
        "dry_fire", "Dry fire",
        "Does this read as a trigger and striker on an empty chamber?",
        "dry_fire", _gun_options(),
    ),

    # One representative of each reload branch.
    *tuple(
        AuditionCase(
            f"reload_{weapon}", f"Reload: {weapon}",
            f"Does this remain one physically plausible {weapon} reload sequence?",
            "reload", _gun_options(weapon=weapon),
        )
        for weapon in ("rifle", "shotgun", "revolver", "artillery")
    ),

    # The new burst composition, after its individual report/mechanisms are known.
    AuditionCase(
        "burst_smg_unsuppressed", "Complete SMG burst: unsuppressed",
        "Do the reports, actions, casings, and outdoor tail fuse into one burst?",
        "burst", _gun_options(weapon="smg", shots=5),
    ),
    AuditionCase(
        "burst_smg_suppressed", "Complete SMG burst: suppressed",
        "Does suppression move the repeated actions forward without losing the burst?",
        "burst", _gun_options(
            weapon="smg", suppression="suppressed", shots=5
        ),
    ),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_session(seed: int) -> dict:
    now = _now()
    return {
        "version": SESSION_VERSION,
        "generator": "gunshot_sounds_generator",
        "seed": seed,
        "started_utc": now,
        "updated_utc": now,
        "decisions": {},
    }


def _load_session(path: Path, seed: int, restart: bool,
                  cases: Sequence[AuditionCase] = AUDITION_CASES) -> dict:
    if restart or not path.is_file():
        return _new_session(seed)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != SESSION_VERSION:
        raise ValueError(
            f"unsupported audition session version in {path}; use --restart"
        )
    if data.get("seed") != seed:
        raise ValueError(
            f"{path} uses seed {data.get('seed')}, not {seed}; resume with "
            "that seed or use --restart"
        )
    known = {case.id for case in cases}
    unknown = set(data.get("decisions", {})) - known
    if unknown:
        raise ValueError(
            f"{path} contains obsolete cases {sorted(unknown)}; use --restart"
        )
    return data


def _markdown(session: dict, cases: Sequence[AuditionCase], out_dir: Path) -> str:
    decisions = session["decisions"]
    yes = sum(item["verdict"] == "yes" for item in decisions.values())
    no = sum(item["verdict"] == "no" for item in decisions.values())
    remaining = len(cases) - len(decisions)
    if remaining:
        status = f"IN PROGRESS — {remaining} remaining"
    elif no:
        status = f"BLOCKED — {no} rejected"
    else:
        status = "PASSED — every gate accepted"
    lines = [
        "# Gunshot Audition Results",
        "",
        f"Status: **{status}**",
        "",
        f"Seed: `{session['seed']}`  ",
        f"Updated: `{session['updated_utc']}`  ",
        f"Accepted: **{yes}** · Rejected: **{no}** · Remaining: **{remaining}**",
        "",
        "| # | Sound | Verdict | Note | Artifact |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for index, case in enumerate(cases, 1):
        decision = decisions.get(case.id)
        verdict = decision["verdict"] if decision else "pending"
        note = (decision.get("note", "") if decision else "").replace("|", "\\|")
        artifact = case.filename if (out_dir / case.filename).is_file() else "—"
        lines.append(
            f"| {index} | {case.label} | **{verdict}** | {note} | `{artifact}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _save_session(session: dict, cases: Sequence[AuditionCase],
                  out_dir: Path) -> None:
    session["updated_utc"] = _now()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / RESULTS_JSON
    json_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
    (out_dir / RESULTS_MARKDOWN).write_text(
        _markdown(session, cases, out_dir), encoding="utf-8"
    )


def _render_case(case: AuditionCase, path: Path, seed: int) -> str:
    rng = random.Random(seed)
    if case.generator == "weapons":
        assert isinstance(case.options, weapon_cues.Options)
        signal = weapon_cues.render(case.cue, case.options, rng)
    else:
        assert isinstance(case.options, Options)
        signal = render(case.cue, case.options, rng).samples
    engine.write_wav(path, signal)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def available_player(preferred: str | None = None) -> str:
    choices = (preferred,) if preferred else PLAYERS
    for player in choices:
        if player and shutil.which(player):
            return player
    requested = preferred or ", ".join(PLAYERS)
    raise ValueError(f"no audio player found ({requested})")


def play_audio(path: Path, preferred: str | None = None) -> None:
    """Play synchronously; try fallbacks when an installed backend cannot connect."""

    choices = (preferred,) if preferred else PLAYERS
    errors: list[str] = []
    for player in choices:
        if not player or not shutil.which(player):
            continue
        command = [player, str(path)]
        if player == "ffplay":
            command = [
                player, "-nodisp", "-autoexit", "-loglevel", "error", str(path)
            ]
        result = subprocess.run(
            command, check=False, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True,
        )
        if result.returncode == 0:
            return
        detail = result.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        errors.append(f"{player} exited {result.returncode}{suffix}")
    detail = "; ".join(errors) if errors else "none installed"
    raise RuntimeError(f"could not play {path}: {detail}")


def _prompt(input_fn: Callable[[str], str]) -> tuple[str, str]:
    """Return (verdict, note); replay and quit are control results."""

    while True:
        answer = input_fn(
            "Accept this sound? [y]es / [n]o / [r]eplay / [q]uit: "
        ).strip().lower()
        if answer in {"y", "yes"}:
            return "yes", ""
        if answer in {"n", "no"}:
            note = input_fn("Short rejection note (optional): ").strip()
            return "no", note
        if answer in {"r", "replay"}:
            return "replay", ""
        if answer in {"q", "quit"}:
            return "quit", ""
        print("Please enter y, n, r, or q.")


def print_queue(cases: Sequence[AuditionCase] = AUDITION_CASES) -> None:
    for index, case in enumerate(cases, 1):
        print(f"{index:2d}. {case.id:<38} {case.label}")
    print(f"\n{len(cases)} sounds total")


def print_status(out_dir: Path,
                 cases: Sequence[AuditionCase] = AUDITION_CASES) -> None:
    path = out_dir / RESULTS_JSON
    if not path.is_file():
        print(f"No audition started at {out_dir}")
        print(f"0/{len(cases)} decided")
        return
    session = json.loads(path.read_text(encoding="utf-8"))
    decisions = session.get("decisions", {})
    yes = sum(item["verdict"] == "yes" for item in decisions.values())
    no = sum(item["verdict"] == "no" for item in decisions.values())
    print(f"Audition: {out_dir}")
    print(f"{len(decisions)}/{len(cases)} decided: {yes} yes, {no} no")
    if len(decisions) < len(cases):
        next_case = next(case for case in cases if case.id not in decisions)
        print(f"Next: {next_case.label}")


def run_audition(out_dir: Path, seed: int, *, restart: bool = False,
                 preferred_player: str | None = None,
                 input_fn: Callable[[str], str] = input,
                 play_fn: Callable[[Path, str | None], None] = play_audio,
                 cases: Sequence[AuditionCase] = AUDITION_CASES) -> int:
    """Run or resume the queue; a completed queue with any ``no`` returns 1."""

    out_dir = out_dir.expanduser().resolve()
    session_path = out_dir / RESULTS_JSON
    session = _load_session(session_path, seed, restart, cases)
    if restart and out_dir.is_dir():
        for case in cases:
            (out_dir / case.filename).unlink(missing_ok=True)
    _save_session(session, cases, out_dir)

    decisions = session["decisions"]
    remaining = [case for case in cases if case.id not in decisions]
    print(f"\nGunshot audition: {len(cases)} yes/no gates")
    print(f"Progress: {len(decisions)}/{len(cases)} decided")
    print(f"Results:  {out_dir / RESULTS_MARKDOWN}")
    if not remaining:
        print("Nothing remaining. Use --restart to audition again.")
        return 1 if any(
            item["verdict"] == "no" for item in decisions.values()
        ) else 0

    # Fail before rendering if the explicitly requested player is unavailable.
    if preferred_player:
        available_player(preferred_player)

    for case in remaining:
        index = next(i for i, item in enumerate(cases, 1) if item.id == case.id)
        path = out_dir / case.filename
        case_seed = (
            case.render_seed if case.render_seed is not None
            else derive_seed(
                seed,
                provider=f"{__package__}.audition.{case.generator}",
                cue=case.cue,
                filename=case.filename,
            )
        )
        digest = _render_case(case, path, case_seed)
        while True:
            print(f"\n[{index}/{len(cases)}] {case.label}")
            print(f"Criterion: {case.criterion}")
            print(f"Artifact:  {path}")
            play_fn(path, preferred_player)
            verdict, note = _prompt(input_fn)
            if verdict == "replay":
                continue
            if verdict == "quit":
                _save_session(session, cases, out_dir)
                print("Paused. Resume with the same audition command.")
                return 0
            session["decisions"][case.id] = {
                "label": case.label,
                "criterion": case.criterion,
                "verdict": verdict,
                "note": note,
                "file": case.filename,
                "sha256": digest,
                "seed": case_seed,
                "decided_utc": _now(),
            }
            _save_session(session, cases, out_dir)
            print(f"Recorded: {verdict.upper()}")
            break

    yes = sum(item["verdict"] == "yes"
              for item in session["decisions"].values())
    no = sum(item["verdict"] == "no"
             for item in session["decisions"].values())
    print(f"\nAudition complete: {yes} accepted, {no} rejected.")
    print(f"Results: {out_dir / RESULTS_MARKDOWN}")
    return 1 if no else 0


__all__ = [
    "AUDITION_CASES", "AuditionCase", "RESULTS_JSON", "RESULTS_MARKDOWN",
    "available_player", "play_audio", "print_queue", "print_status",
    "run_audition",
]
