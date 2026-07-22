"""Cue and pack registries for the music generator.

Every cue is a named Strudel pattern (JS/mini-notation), a tempo (``cps``), and
either an integer cycle count (loops) or a capture length (stings). The pattern
text *is* the source of truth for this generator: it is pinned in-package and
hashed, because a real-time browser capture cannot be pinned by its output
bytes (see the determinism note in ``engine.py`` and the README).

Two families of cue live here:

* Three original project-authored tracks -- ``battle_loop``,
  ``thinking_loop``, and ``victory_fanfare`` -- first composed and signed off
  by ear on 2026-07-07. Their exact pattern hashes retain that provenance.

* Three more, written here as pure-synth beds and stings for the rest of a
  game's music states: a ``menu_theme``, a ``tension_loop``, a ``defeat_sting``.

Licence cleanliness: every pattern uses only synthesised sources --
``s("sine"/"sawtooth"/"triangle"/"square")`` and noise
(``"white"/"pink"/"brown"``). No ``s("bd")``, ``s("piano")``, or any
dough-samples sample bank appears anywhere, so no sample-bank licence attaches
to the output. This is enforced by a test.
"""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from ..._audio.models import VariationFilenames

from dataclasses import dataclass


# --------------------------------------------------------------------------
# cue model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cue:
    """A named Strudel pattern plus how long to record it.

    ``cycles`` is set for loops and is the whole reason a loop wraps cleanly:
    recording an integer number of pattern cycles means the tempo grid lands
    exactly on the loop point (``seconds = cycles / cps``). One-shots leave it
    ``None`` and set ``seconds`` directly, long enough for the release to ring
    out before ``hush()``.
    """

    name: str
    group: str
    description: str
    cps: float
    cycles: int | None
    seconds: float
    loop: bool
    pattern: str
    source: str
    honours: Honours | str | tuple[str, ...] = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


def _loop(name: str, description: str, cps: float, cycles: int,
          pattern: str, source: str) -> Cue:
    return Cue(name, "loop", description, cps, cycles,
               round(cycles / cps, 6), True, pattern, source,
               "seconds (overrides the integer-cycle length)")


def _sting(name: str, description: str, cps: float, seconds: float,
           pattern: str, source: str) -> Cue:
    return Cue(name, "sting", description, cps, None,
               seconds, False, pattern, source, "seconds (capture length)")


# --------------------------------------------------------------------------
# the three original patterns accepted on 2026-07-07
#
# The original captures used nominal lengths of 24 s / 20 s / 7 s. The two
# loops are recorded here as an integer number of cycles instead (12 cycles =
# 24.0 s at 0.5 cps; 6 cycles = 21.43 s at 0.28 cps, the whole-cycle length
# nearest the nominal 20 s), so the per-cycle content wraps clean at the loop
# point. The fanfare is a one-shot and keeps its 7 s.
# --------------------------------------------------------------------------

_ORIGINAL = "original project composition (accepted 2026-07-07 by ear)"

BATTLE_LOOP = _loop(
    "battle_loop",
    "driving minor-key battle bed: bass, saw ostinato, triangle arpeggio",
    0.5, 12,
    """
setcps(0.5)

let roots = note("<d2 bb1 c2 a1>/4")

stack(
  roots.s("sine").gain(0.40).attack(0.02).release(0.18).lpf(125),
  note("<d3 ~ d3 f3 bb2 ~ c3 a2>/8").s("sawtooth").gain(0.23).attack(0.01).release(0.10).lpf(520).pan("<0.35 0.65>/4"),
  note("[d4 f4 a4 f4] [bb3 d4 f4 d4] [c4 e4 g4 e4] [a3 c4 e4 c4]").s("triangle").gain(0.24).attack(0.006).release(0.11).delay(0.16).pan("<0.2 0.8>/8"),
  s("brown*8").gain("0.024 0.010 0.015 0.010 0.022 0.010 0.018 0.010").hpf(1100).lpf(4200)
).gain(1.0)
""",
    _ORIGINAL,
)

THINKING_LOOP = _loop(
    "thinking_loop",
    "slow contemplative bed: deep sine root, drifting triangle, sine sparkle",
    0.28, 6,
    """
setcps(0.28)

stack(
  note("<d2 a1 bb1 f2>/8").s("sine").gain(0.24).attack(0.5).release(2.2).lpf(140),
  note("<d4 f4 a4 c5>/16").s("triangle").gain(0.11).attack(0.08).release(1.1).delay(0.3).pan("<0.25 0.75>/8"),
  note("[d5 ~ ~ a4] [f5 ~ ~ c5] [a5 ~ ~ f5] [c6 ~ ~ a5]").s("sine").gain(0.055).attack(0.03).release(0.9).delay(0.42).pan("<0.8 0.2>/8"),
  s("pink").gain(0.015).hpf(700).lpf(1800)
).gain(0.72)
""",
    _ORIGINAL,
)

VICTORY_FANFARE = _sting(
    "victory_fanfare",
    "bright rising fanfare: triangle chords, square lead, saw bass, one-shot",
    0.72, 7.0,
    """
setcps(0.72)

stack(
  note("[d4 f4 a4 d5] [f4 a4 c5 f5] [g4 bb4 d5 g5] [a4 c5 e5 a5]")
    .s("triangle").gain(0.26).attack(0.015).release(0.22).lpf(1800),
  note("[d5 ~ a4 d5] [f5 ~ c5 f5] [g5 ~ d5 g5] [a5 c6 d6 ~]")
    .s("square").gain(0.18).attack(0.01).release(0.16).hpf(450).lpf(2600).pan(0.35),
  note("[d3 ~ d3 ~] [f3 ~ f3 ~] [g3 ~ g3 ~] [a3 ~ d4 ~]")
    .s("sawtooth").gain(0.16).attack(0.01).release(0.12).lpf(460),
  s("white*2").gain("0.035 0.02").hpf(2000).lpf(8000)
).gain(0.82)
""",
    _ORIGINAL,
)


# --------------------------------------------------------------------------
# new pure-synth cues, in the same D-minor world as the original tracks so a
# game can move between them without a key change. All synthesised sources.
# --------------------------------------------------------------------------

_INPKG = "music_sound_generator (in-package, pure-synth)"

MENU_THEME = _loop(
    "menu_theme",
    "calm title/menu bed: slow sine pad, gentle triangle arpeggio, sine bell",
    0.5, 8,
    """
setcps(0.5)

stack(
  note("<d2 a1 bb1 f2>/8").s("sine").gain(0.26).attack(0.6).release(2.6).lpf(150),
  note("<[d4 a4 f4 a4] [c4 g4 e4 g4] [bb3 f4 d4 f4] [f4 c5 a4 c5]>")
    .s("triangle").gain(0.14).attack(0.02).release(0.5).delay(0.35).lpf(1600).pan("<0.35 0.65>/4"),
  note("<d5 ~ a5 ~ f5 ~ c5 ~>").s("sine").gain(0.06).attack(0.01).release(1.2).delay(0.5).pan("<0.7 0.3>/8"),
  s("pink").gain(0.012).hpf(900).lpf(2600)
).gain(0.7)
""",
    _INPKG,
)

TENSION_LOOP = _loop(
    "tension_loop",
    "suspense bed: low sine drone, slow triangle pulse, dissonant saw stab",
    0.5, 8,
    """
setcps(0.5)

stack(
  note("d1").s("sine").gain(0.30).attack(0.01).release(1.9).lpf(90),
  note("[d2 ~ ~ ~ d2 ~ ~ ~]").s("triangle").gain(0.12).attack(0.005).release(0.18).lpf(300),
  note("<eb4 ~ ~ ~>/2").s("sawtooth").gain(0.05).attack(0.4).release(1.4).lpf(900).delay(0.4).pan(0.6),
  s("brown").gain(0.03).hpf(40).lpf(400),
  s("white*4").gain("0.006 0.002 0.004 0.002").hpf(3000).lpf(9000)
).gain(0.62)
""",
    _INPKG,
)

DEFEAT_STING = _sting(
    "defeat_sting",
    "dark descending defeat motif: falling saw run, triangle answer, low sine",
    0.5, 6.5,
    """
setcps(0.5)

stack(
  note("<[a3 f3 d3 a2] [f3 d3 a2 f2]>").s("sawtooth").gain(0.20).attack(0.006).release(0.4).lpf(760).pan("<0.4 0.6>"),
  note("<[f4 ~ d4 ~] [d4 ~ bb3 ~]>").s("triangle").gain(0.13).attack(0.01).release(0.55).lpf(1500).delay(0.22),
  note("<d2 a1>").s("sine").gain(0.30).attack(0.02).release(2.4).lpf(125),
  s("brown*4").gain("0.02 0.012 0.015 0.008").hpf(240).lpf(1500)
).gain(0.72)
""",
    _INPKG,
)


START_FANFARE = _sting(
    "start_fanfare",
    "short bright game-start fanfare: square-lead brass call, triangle chords",
    0.8, 2.6,
    """
setcps(0.8)

stack(
  note("[a4 a4 a4 d5] [f5 a5 ~ ~]")
    .s("square").gain(0.24).attack(0.008).release(0.20).hpf(420).lpf(3200).pan(0.5),
  note("[d4 ~ ~ ~] [d4 f4 a4 ~]")
    .s("triangle").gain(0.20).attack(0.012).release(0.42).lpf(2000),
  note("[d3 ~ ~ ~] [d3 ~ a3 ~]")
    .s("sawtooth").gain(0.16).attack(0.01).release(0.32).lpf(500),
  s("white*2").gain("0.03 0.015").hpf(2000).lpf(8000)
).gain(0.85)
""",
    _INPKG,
)


WIN_FANFARE = _sting(
    "win_fanfare",
    "short triumphant win fanfare: square-lead brass rise to a held tonic",
    0.75, 2.7,
    """
setcps(0.75)

stack(
  note("[a4 d5 f5 a5] [d6 ~ ~ ~]")
    .s("square").gain(0.24).attack(0.008).release(0.55).hpf(420).lpf(3200).pan(0.5),
  note("[d4 f4 a4 d5] [d5 a4 f4 d5]")
    .s("triangle").gain(0.20).attack(0.012).release(0.6).lpf(2000),
  note("[d3 ~ a3 ~] [d3 ~ ~ ~]")
    .s("sawtooth").gain(0.16).attack(0.01).release(0.5).lpf(500),
  s("white*2").gain("0.03 0.02").hpf(2000).lpf(8000)
).gain(0.72)
""",
    _INPKG,
)


CUES: dict[str, Cue] = {
    cue.name: cue
    for cue in (
        BATTLE_LOOP,
        THINKING_LOOP,
        VICTORY_FANFARE,
        START_FANFARE,
        WIN_FANFARE,
        MENU_THEME,
        TENSION_LOOP,
        DEFEAT_STING,
    )
}

GROUPS = ("loop", "sting")


# --------------------------------------------------------------------------
# packs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PackEntry:
    filename: str
    cue: str
    count: int = 1


@dataclass(frozen=True)
class Pack(VariationFilenames):
    name: str
    description: str
    entries: tuple[PackEntry, ...]



# A compact board-game music bank using the original three filenames.
BOARD_DUEL = Pack(
    name="board-duel",
    description="thinking, battle, and victory music for a board-game duel",
    entries=(
        PackEntry("thinking_loop", "thinking_loop"),
        PackEntry("battle_loop", "battle_loop"),
        PackEntry("victory_fanfare", "victory_fanfare"),
    ),
)

SHOWCASE = Pack(
    name="showcase",
    description="one of every cue, for auditioning the whole library",
    entries=tuple(PackEntry(name, name) for name in CUES),
)

PACKS = {pack.name: pack for pack in (BOARD_DUEL, SHOWCASE)}
