"""The cue registry: every UI and game-state sound this project can render.

A cue is a named renderer plus the level it masters to. Everything the CLI can
produce is in CUES or PACKS, so `list` is generated from the same tables the
renderer uses and cannot drift out of date.

**The pitch language here is deliberately not powerup_generator's.**

That generator's design is "a rising perfect interval reads as *gain*" — every
pickup uses one, because a pickup is a reward. Reuse that here and every menu
keypress sounds like picking up a coin, which is both wrong and exhausting.

UI and game state are not about reward. They are about *where you are*: is this
thing open or closed, pending or resolved, running or over. So this bank is
built on **tension and resolution around a tonic**, which is the move Western
music already uses to mean exactly that:

- **Neutral** (`menu_move`, `score_tick`) is a single note with no interval at
  all. These fire hundreds of times a session. An interval is a message, and a
  message you are forced to re-read on every keypress becomes noise you
  learn to hate. The absence of an interval is the design, not an omission.
- **Affirmative** (`menu_accept`, `checkpoint`, `resume`, `lap_complete`)
  resolves *up to the tonic*: the dominant G rising to C. This is the core
  motion of the authentic cadence — the sound of arriving.
- **Withdrawing** (`menu_cancel`, `pause`) is that same motion reversed, tonic
  falling back to the dominant below. Note this is *not* the "wrong" sound:
  backing out of a menu is not an error, and a bank that punishes it
  teaches the player to fear its own UI.
- **Wrong** (`menu_invalid`) is a falling minor 2nd — deliberately the same
  idiom as `powerup_generator`'s `denied`, because it is the single most
  reliably "wrong"-sounding move available, and inventing a second
  vocabulary for the same meaning would only make the two banks fight.
- **Unresolved tension** (`countdown_tick`, `warning`) sits on the dominant and
  refuses to resolve. A countdown that resolved on every tick would have
  nothing left to spend when it reaches zero.
- **Release** (`countdown_go`, `round_start`) is the tonic finally arriving.
- **Terminal** (`win`, `lose`, `game_over`) are full cadences; see below.

Everything is tuned around C major, matching `powerup_generator`, so the two
banks can be used in one game without clashing and both layer under music.
Transposing the lot is a matter of adding a constant to the MIDI numbers here.
"""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..powerup_generator import engine as e
from ..powerup_generator import gestures as g
from ..powerup_generator.styles import Style

# MIDI reference: C4 = 60 (middle C), C5 = 72, C6 = 84.
C4, D4, EB4, E4, F4, G4, AB4, A4, B4 = 60, 62, 63, 64, 65, 67, 68, 69, 71
C5, D5, EB5, E5, F5, G5, AB5, A5, B5 = 72, 74, 75, 76, 77, 79, 80, 81, 83
C6, D6, E6, F6, G6, A6, B6 = 84, 86, 88, 89, 91, 93, 95
C7, E7 = 96, 100

# Scale degrees of the major scale, in semitones. Used by the cues that honour
# `--step` so a climbing tally or combo stays *in key* instead of walking up in
# chromatic semitones, which would fight any music underneath it.
_MAJOR = (0, 2, 4, 5, 7, 9, 11)

# A combo or tally that keeps climbing eventually runs out of usable register.
# Past two octaves the notes stop reading as distinct pitches and start reading
# as the same thin whistle, so the ladder saturates rather than continuing into
# frequencies nobody's speakers reproduce.
MAX_STEP = 14


def scale_step(root: float, degrees: int) -> float:
    """Transpose `root` by `degrees` steps of the C major scale.

    Negative degrees walk downward; `divmod` handles the wrap, so -1 is the
    seventh below rather than an error.
    """
    octave, index = divmod(int(degrees), 7)
    return root + 12 * octave + _MAJOR[index]


def _clamp_step(step: int) -> int:
    return max(0, min(int(step), MAX_STEP))


Rng = np.random.Generator
Renderer = Callable[[Style, Rng, int], np.ndarray]


@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Renderer
    peak_db: float
    honours: Honours | str | tuple[str, ...] = "style, seed"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


# --------------------------------------------------------------------------
# menu — the highest-frequency sounds in the bank, and the quietest
# --------------------------------------------------------------------------

def _menu_move(style: Style, rng: Rng, step: int) -> np.ndarray:
    """One neutral blip. No interval, because this fires on every keypress."""
    # Deliberately mid-register and short. High enough to cut through, low
    # enough not to be piercing at the repetition rate this thing runs at.
    out = g.blip(style, G5, 0.032, rng, brightness=0.7, with_body=False)
    total = len(out) / e.SR
    mixed = e.silence(total)
    e.add_at(mixed, 0.0, out)
    e.add_at(mixed, 0.0, g.accent_at(style, min(0.02, total), rng,
                                     brightness=0.45), 0.35)
    return mixed


def _menu_accept(style: Style, rng: Rng, step: int) -> np.ndarray:
    """Dominant resolving up to the tonic: a rising fourth, G5 -> C6."""
    return g.arpeggio(style, [G5, C6], rng, note_dur=0.048, gap=0.042,
                      brightness=1.0)


def _menu_cancel(style: Style, rng: Rng, step: int) -> np.ndarray:
    """The accept motion reversed: C6 -> G5. Backing out, not failing."""
    return g.arpeggio(style, [C6, G5], rng, note_dur=0.048, gap=0.042,
                      brightness=0.65)


def _menu_invalid(style: Style, rng: Rng, step: int) -> np.ndarray:
    """A falling minor second, dry and doubled. The one unambiguous 'no'."""
    fall = g.arpeggio(style, [B4, B4 - 1], rng, note_dur=0.042, gap=0.038,
                      brightness=0.4, with_body=False)
    total = len(fall) / e.SR
    out = e.silence(total)
    e.add_at(out, 0.0, fall)
    # A dry mechanical edge on the onset so it reads as a rejected keypress
    # rather than a sad little tune.
    e.add_at(out, 0.0, g.noise_crack(min(0.05, total), rng, 1_400.0, 5_500.0,
                                     decay=0.010), 0.4)
    return out


# --------------------------------------------------------------------------
# state — pause/resume and the countdown
# --------------------------------------------------------------------------

def _close_over(signal: np.ndarray, start_hz: float,
                end_hz: float) -> np.ndarray:
    """Sweep a lowpass across a finished buffer.

    Used by pause/resume. A pause is not just a falling interval — it is the
    sound of the game's audio being *taken away*, and a closing filter is what
    that sounds like. Applied after the style has had its say, so it dulls
    whatever the style produced rather than replacing it.
    """
    duration = len(signal) / e.SR
    cutoff = e.ramp(duration, start_hz, end_hz, curve=0.8)
    return e.resonant_sweep(signal, np.clip(cutoff, 60.0, e.SR * 0.45), q=0.9,
                            mode="lowpass")


def _pause(style: Style, rng: Rng, step: int) -> np.ndarray:
    """C6 falling to G5 with the filter closing: the sound withdrawing."""
    sweep = g.glide(style, C6, G5, 0.26, rng, brightness=0.7, curve=1.25,
                    with_body=True)
    return _close_over(sweep, 6_500.0, 700.0)


def _resume(style: Style, rng: Rng, step: int) -> np.ndarray:
    """The pause reversed, filter opening: the sound coming back."""
    sweep = g.glide(style, G5, C6, 0.24, rng, brightness=0.9, curve=0.75,
                    with_body=True)
    return _close_over(sweep, 700.0, 7_500.0)


def _countdown_tick(style: Style, rng: Rng, step: int) -> np.ndarray:
    """The dominant, struck once, going nowhere. Three of these precede `go`.

    Every tick is the *same* pitch on purpose. A countdown that stepped down in
    pitch would spend its tension early and leave `countdown_go` with
    nothing to resolve.
    """
    out = g.blip(style, G5, 0.075, rng, brightness=0.95, with_body=True)
    total = len(out) / e.SR
    mixed = e.silence(total)
    e.add_at(mixed, 0.0, out)
    e.add_at(mixed, 0.0, g.accent_at(style, min(0.03, total), rng,
                                     brightness=0.8), 0.6)
    return mixed


def _countdown_go(style: Style, rng: Rng, step: int) -> np.ndarray:
    """C6: the tonic the ticks have been circling, finally arriving."""
    hit = g.blip(style, C6, 0.30, rng, brightness=1.3, with_body=True)
    total = len(hit) / e.SR + 0.10
    out = e.silence(total)
    e.add_at(out, 0.0, hit)
    # An octave underneath, for the weight the ticks never had.
    e.add_at(out, 0.0, g.blip(style, C5, 0.26, rng, brightness=0.8), 0.45)
    return e.mix(out, g.sparkle_tail(style, total, rng, 5_000.0))


# --------------------------------------------------------------------------
# feedback — tally, combo, checkpoint, warning
# --------------------------------------------------------------------------

def _score_tick(style: Style, rng: Rng, step: int) -> np.ndarray:
    """A tiny bright blip. Honours --step so a tally can climb the scale.

    The quietest cue in the bank: a score tally fires this dozens of times in a
    row, and anything with presence becomes a machine gun.
    """
    note = scale_step(C6, _clamp_step(step))
    return g.blip(style, note, 0.026, rng, brightness=0.85, with_body=False)


def _combo_step(style: Style, rng: Rng, step: int) -> np.ndarray:
    """A combo hit that climbs the scale and brightens as it goes.

    Honours --step. The brightening is the point: pitch alone stops reading as
    escalation once the notes get high, so the timbre has to escalate with it.
    """
    level = _clamp_step(step)
    note = scale_step(C5, level)
    # 0 -> 0.85, saturating toward ~1.6 at the top of the ladder.
    brightness = 0.85 + 0.75 * (level / MAX_STEP)
    out = g.blip(style, note, 0.085, rng, brightness=brightness,
                 with_body=True)
    total = len(out) / e.SR
    mixed = e.silence(total)
    e.add_at(mixed, 0.0, out)
    # Sparkle only enters in the upper half of the ladder, so a 2-hit combo and
    # a 12-hit combo are different events rather than the same one louder.
    if level > MAX_STEP // 2:
        glitter = g.sparkle_tail(style, total, rng, 6_000.0, 14_000.0)
        mixed = e.mix(mixed, glitter * (level / MAX_STEP))
    return mixed


def _checkpoint(style: Style, rng: Rng, step: int) -> np.ndarray:
    """G5 -> C6 -> E6: the accept motion, carried through into the third.

    Deliberately the same opening move as `menu_accept`. A checkpoint *is* an
    accept — the game agreeing you got here — just a bigger one, so it says the
    same thing and then keeps going.
    """
    out = g.arpeggio(style, [G5, C6, E6], rng, note_dur=0.070, gap=0.062,
                     brightness=1.05, accel=0.92)
    return e.mix(out, g.sparkle_tail(style, len(out) / e.SR, rng, 4_500.0))


def _warning(style: Style, rng: Rng, step: int) -> np.ndarray:
    """B5 and F5 alternating — the tritone that already lives in C major.

    B-F is the diminished fifth between the leading tone and the subdominant.
    It is the most unstable interval available *without leaving the key*, so it
    reads as an alarm while still layering under C-major music. A tritone
    borrowed from outside the key would sound more urgent and fit nothing.

    Distinct from `powerup_generator`'s `powerup_warning`, which is three
    pulses on one pitch and means "your powerup is expiring". This one means
    "something is wrong with the game state".
    """
    period = 0.115
    pulse = 0.070
    count = 4
    total = period * (count - 1) + pulse * style.tail_scale + 0.03
    out = e.silence(total)
    for i in range(count):
        note = B5 if i % 2 == 0 else F5
        voice = g.blip(style, note, pulse, rng, brightness=1.0,
                       with_body=False)
        e.add_at(out, i * period, voice, 1.0)
    return out


# --------------------------------------------------------------------------
# round — the shape of a single round: start, lap, finish
# --------------------------------------------------------------------------

def _round_start(style: Style, rng: Rng, step: int) -> np.ndarray:
    """C5-E5-G5-C6: a bugle call on the tonic triad. An opening, not a release.

    The distinction from `countdown_go` is real and worth keeping.
    `countdown_go` is a *resolution*: it only means anything after
    `countdown_tick` has built the tension it discharges. `round_start` is an
    *announcement*, and has to work cold, with no setup. So it is a call (an
    arpeggio that states the key) rather than a single arriving note.
    """
    out = g.arpeggio(style, [C5, E5, G5, C6], rng, note_dur=0.080, gap=0.070,
                     brightness=1.15, accel=0.90)
    return e.mix(out, g.sparkle_tail(style, len(out) / e.SR, rng))


def _lap_complete(style: Style, rng: Rng, step: int) -> np.ndarray:
    """A quick rise to the tonic.

    A checkpoint's smaller sibling: "noted, keep going".
    """
    return g.arpeggio(style, [E5, G5, C6], rng, note_dur=0.052, gap=0.046,
                      brightness=1.0, accel=0.90)


def _finish(style: Style, rng: Rng, step: int) -> np.ndarray:
    """A rising run over two octaves. Crossing the line — but not a verdict.

    `finish` and `win` are not the same event and should not be the same sound.
    You cross the line, and *then* you find out where you placed. A game that
    plays `win` at the line has told the player the result before it knows it.
    """
    out = g.arpeggio(style, [C5, E5, G5, C6, E6, G6], rng, note_dur=0.058,
                     gap=0.052, brightness=1.2, accel=0.90)
    return e.mix(out, g.sparkle_tail(style, len(out) / e.SR, rng, 5_000.0))


# --------------------------------------------------------------------------
# terminal — the verdicts, as cadences
# --------------------------------------------------------------------------

def _win(style: Style, rng: Rng, step: int) -> np.ndarray:
    """A real authentic cadence: G major resolving to C major. V -> I.

    Not an arpeggio-plus-triad like `powerup_generator`'s `quest_complete` — a
    fanfare and a cadence are different claims. A fanfare says "well done"; a
    cadence says "this is finished". Only one of those is true at a win.
    """
    dominant = g.chord(style, [G5, B5, D6], 0.30, rng, brightness=1.0)
    tonic = g.chord(style, [C6, E6, G6, C7], 0.70, rng, brightness=1.25)
    total = 0.30 + 0.78
    out = e.silence(total)
    e.add_at(out, 0.0, dominant, 0.80)
    e.add_at(out, 0.28, tonic, 1.0)
    return e.mix(out, g.sparkle_tail(style, total, rng))


def _lose(style: Style, rng: Rng, step: int) -> np.ndarray:
    """The win cadence's world, in the parallel minor, falling not arriving.

    C6 -> Ab5 -> Eb5 is the C minor triad descending: the same tonic as `win`,
    borrowed into the parallel minor. That relationship is the design: a
    player hears the key they were winning in, gone dark, without being told.
    """
    fall = g.arpeggio(style, [C6, AB5, EB5], rng, note_dur=0.130, gap=0.115,
                      brightness=0.5, accel=1.10)
    total = len(fall) / e.SR
    out = e.silence(total)
    e.add_at(out, 0.0, fall)
    return _close_over(out, 5_000.0, 1_100.0)


def _game_over(style: Style, rng: Rng, step: int) -> np.ndarray:
    """The `lose` descent, slower, an octave down, landing on the low tonic.

    The longest and lowest cue in the bank. `lose` is a round ending; this
    is the run ending, and the difference is register and patience.
    """
    fall = g.arpeggio(style, [C5, AB4, EB4, C4], rng, note_dur=0.190,
                      gap=0.175, brightness=0.35, accel=1.12, with_body=True)
    total = len(fall) / e.SR + 0.20
    out = e.silence(total)
    e.add_at(out, 0.0, fall)
    # A low sine settling underneath the last note: the floor the run stops on.
    settle = e.osc("sine", e.midi_to_freq(C4 - 12), 0.55)
    e.add_at(out, 0.175 * 2.6, settle * e.exp_env(0.55, 0.20, 0.004), 0.40)
    return _close_over(out, 3_500.0, 500.0)


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

_STEP = "style, seed, step"

CUES: dict[str, Cue] = {
    # menu
    "menu_move": Cue("menu_move", "menu",
                     "one neutral blip, no interval; fires constantly",
                     _menu_move, -20.0),
    "menu_accept": Cue("menu_accept", "menu",
                       "rising fourth to the tonic; the dominant resolving",
                       _menu_accept, -13.0),
    "menu_cancel": Cue("menu_cancel", "menu",
                       "the accept motion reversed; backing out, not failing",
                       _menu_cancel, -14.0),
    "menu_invalid": Cue("menu_invalid", "menu",
                        "falling minor second, dry; the one unambiguous 'no'",
                        _menu_invalid, -13.0),
    # state
    "pause": Cue("pause", "state",
                 "falling fourth with the filter closing; sound withdrawing",
                 _pause, -12.0),
    "resume": Cue("resume", "state",
                  "the pause reversed, filter opening; sound returning",
                  _resume, -12.0),
    "countdown_tick": Cue("countdown_tick", "state",
                          "the dominant struck once, unresolved; 3 of these",
                          _countdown_tick, -11.0),
    "countdown_go": Cue("countdown_go", "state",
                        "the tonic arriving; what the ticks were circling",
                        _countdown_go, -8.0),
    # feedback
    "score_tick": Cue("score_tick", "feedback",
                      "tiny blip for a climbing tally; quietest in the bank",
                      _score_tick, -19.0, honours=_STEP),
    "combo_step": Cue("combo_step", "feedback",
                      "climbs the scale and brightens as the combo escalates",
                      _combo_step, -12.0, honours=_STEP),
    "checkpoint": Cue("checkpoint", "feedback",
                      "the accept motion carried on into the third",
                      _checkpoint, -10.0),
    "warning": Cue("warning", "feedback",
                   "B-F tritone alternating; the alarm native to C major",
                   _warning, -9.0),
    # round
    "round_start": Cue("round_start", "round",
                       "bugle call on the tonic triad; an opening, works cold",
                       _round_start, -8.0),
    "lap_complete": Cue("lap_complete", "round",
                        "quick rise to the tonic; 'noted, keep going'",
                        _lap_complete, -10.0),
    "finish": Cue("finish", "round",
                  "rising run over two octaves; the line, not the verdict",
                  _finish, -7.5),
    # terminal
    "win": Cue("win", "terminal",
               "an authentic cadence, V-I; 'this is finished'", _win, -7.0),
    "lose": Cue("lose", "terminal",
                "the win cadence in the parallel minor, falling", _lose, -9.0),
    "game_over": Cue("game_over", "terminal",
                     "the lose descent, slower and an octave down", _game_over,
                     -8.5),
}

GROUPS = ("menu", "state", "feedback", "round", "terminal")


@dataclass(frozen=True)
class PackEntry:
    cue: str
    count: int = 1
    # A pack for a specific game sets these two; the review banks below leave
    # them alone. `filename` is the stem that game's loader asks for, which is
    # almost never the cue name (a fighter's menu blip is `menu.wav`, the cue
    # behind it is `menu_move`) — setting it also switches the entry to the
    # shared variation contract, see `pack_stem`. `style` pins the timbre the
    # bank was signed off in, so rendering a game's pack needs no flags to come
    # out the way that game shipped.
    filename: str | None = None
    style: str | None = None


def pack_stem(entry: PackEntry, index: int) -> str:
    """Name one file of a pack entry, without the extension.

    Cue-named entries keep this bank's own `_v01..` numbering. Entries that name
    a game's filename take the convention every consumer of these banks already
    implements: bare stem for the first variant, `_v02` onward. Their loaders
    walk the variants until one is missing (`variant_filename()` in each game's
    src/sound.c), so a bank numbered from `_v01` has no first file and loads as
    nothing at all — the game falls back to its own synth and the bank is silent
    rather than broken, which is the version of this mistake you don't notice.
    combat_generators/lib/cues.py calls the same rule 'v_suffix'.
    """
    if entry.filename is None:
        if entry.count == 1:
            return entry.cue
        return f"{entry.cue}_v{index + 1:02d}"
    if index == 0:
        return entry.filename
    return f"{entry.filename}_v{index + 1:02d}"


PACKS: dict[str, tuple[PackEntry, ...]] = {
    # One of everything — the review bank.
    "showcase": tuple(PackEntry(name) for name in CUES),
    # The cues that repeat within a single session, with variations to rotate
    # through. Retriggering one identical menu blip is what makes a UI bank
    # sound cheap, and these fire often enough for it to show.
    "hud": (
        PackEntry("menu_move", 4),
        PackEntry("menu_accept", 2),
        PackEntry("menu_cancel", 2),
        PackEntry("menu_invalid", 2),
        PackEntry("score_tick", 4),
        PackEntry("countdown_tick", 3),
    ),
    # Drop-in bank for Arcade Fighter's four non-combat sounds; the fighting
    # itself comes from unarmed_foley_generator, not from here. Filenames and
    # counts are what its src/sound.c already loads. `menu` is menu_move and not
    # something with more character on purpose: it fires on every keypress of a
    # character select, and this bank's whole argument is that the sound you
    # hear most should carry the least. arcade suits the genre and is also what
    # the combat bank next to it was rendered in.
    "arcade-fighter": (
        PackEntry("menu_move", 3, "menu", "arcade"),
        PackEntry("menu_accept", 2, "select", "arcade"),
        PackEntry("round_start", 2, "round", "arcade"),
        PackEntry("win", 2, "win", "arcade"),
    ),
    # Drop-in bank for Offroad Racer's UI and race-state cues; the driving
    # sounds are vehicle_foley_generator's. Two of these were auditioned rather
    # than read off the name:
    #
    # `start` is countdown_go and not round_start. round_start is a bugle that
    # works cold, and this one is not cold — it lands on the third of three
    # countdown ticks, and countdown_go is written as the tonic those ticks were
    # circling. Played after `countdown`, round_start starts a new thought;
    # countdown_go finishes the one already running.
    #
    # `buy` is the weakest match in the bank and is worth knowing about: it is
    # a shop purchase, which wants a transaction sound, and this bank has no
    # coin or cash cue (deliberately — see the module docstring on why UI does
    # not borrow powerup_generator's reward language). menu_accept is honest
    # about what it is, a confirm, and a confirm is most of what a purchase is.
    "offroad-racer": (
        PackEntry("menu_move", 3, "menu", "arcade"),
        PackEntry("countdown_tick", 2, "countdown", "arcade"),
        PackEntry("countdown_go", 2, "start", "arcade"),
        PackEntry("lap_complete", 2, "lap", "arcade"),
        PackEntry("menu_accept", 2, "buy", "arcade"),
        PackEntry("finish", 3, "finish", "arcade"),
    ),
    # Drop-in bank for Creature Rancher's menu and verdicts. acoustic because the
    # game is a ranch and its other banks are recorded foley; an arcade blip
    # over that would be the only synthetic thing in the mix. win/lose are the
    # parallel-major/minor pair, which is the point of taking both from one
    # bank rather than two.
    "creature-rancher": (
        PackEntry("menu_move", 4, "move", "acoustic"),
        PackEntry("menu_accept", 3, "confirm", "acoustic"),
        PackEntry("win", 3, "win", "acoustic"),
        PackEntry("lose", 3, "lose", "acoustic"),
    ),
    # Drop-in bank for Lunar Lander's three cockpit cues. `beep` is the
    # altimeter/telemetry blip, which is countdown_tick: a single unresolved
    # note struck repeatedly is exactly what an instrument reading out is, and
    # it stays out of the way of `warning` above it.
    "lunar-lander": (
        PackEntry("menu_move", 3, "menu", "scifi"),
        PackEntry("countdown_tick", 3, "beep", "scifi"),
        PackEntry("warning", 2, "warning", "scifi"),
    ),
    # Artillery Terrain's five non-combat/menu-state files. Its ten physical cues
    # come from projectile_sounds and weapons_destruction_generator. The shop
    # purchase is a checkpoint gesture rather than another menu accept: both
    # confirm an action, but only the former carries the small earned-reward
    # lift a successful transaction needs.
    "artillery-terrain": (
        PackEntry("win", 1, "win", "arcade"),
        PackEntry("menu_move", 1, "menu_move", "arcade"),
        PackEntry("menu_accept", 1, "menu_select", "arcade"),
        PackEntry("checkpoint", 1, "buy", "arcade"),
        PackEntry("menu_invalid", 1, "deny", "arcade"),
    ),
    # Brick Breaker's state/UI subset. Paddle/brick/metal contacts and the
    # destruction/launch/powerup cues are supplied by their physical banks.
    "brick-breaker": (
        PackEntry("menu_move", 1, "menu", "arcade"),
        PackEntry("checkpoint", 1, "powerup", "arcade"),
        PackEntry("lose", 1, "lose", "arcade"),
        PackEntry("finish", 1, "clear", "arcade"),
    ),
    # Non-physical semantic events for arena-fps-engine. The Warden's actual
    # alert voice comes from creature_foley_generator; this pack owns only the
    # interface confirmation and the pickup reward.
    "arena-fps": (
        PackEntry("menu_accept", 1, "ui_accept", "scifi"),
        PackEntry("checkpoint", 1, "world_pickup", "scifi"),
    ),
    # Fishing state is readable without inventing physical Foley the source
    # bank does not have: bite is unresolved attention, hook set resolves,
    # escape withdraws, line snap is invalid/error, and shark arrival is the
    # longer danger warning. Fever remains an earned checkpoint transition.
    "aquarium": (
        PackEntry("checkpoint", 1, "frenzy", "arcade"),
        PackEntry("countdown_tick", 1, "hook_bite", "arcade"),
        PackEntry("menu_accept", 1, "hook_set", "arcade"),
        PackEntry("menu_cancel", 1, "fish_escape", "arcade"),
        PackEntry("menu_invalid", 1, "line_snap", "arcade"),
        PackEntry("warning", 1, "shark_alert", "arcade"),
    ),
}


def render(cue: Cue, style: Style, rng: Rng,
           step: int = 0) -> np.ndarray:
    """Render one cue and master it to its own level.

    `step` is honoured only by the cues that declare it in `honours`; the rest
    accept and ignore it, so the signature stays uniform.
    """
    signal = cue.render(style, rng, step)
    return e.master(signal, cue.peak_db, drive=style.drive)
