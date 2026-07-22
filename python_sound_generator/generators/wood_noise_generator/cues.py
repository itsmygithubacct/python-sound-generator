"""Cue and pack registries for the wood noise generator.

The registry is deliberately shaped around what wood *is* rather than around
what happens to it, because that is the provider boundary: a
material generator owns contact, ring, fracture and debris; an event generator
owns the composition. So `strike` is a plank being struck, and its consumer decides
whether a bullet or a hammer struck it.

That cut is also why there are no game packs here. Every other generator in this
repository ships them, and their absence is a design statement rather than an
omission: no game has a `wood_strike.wav` slot, because a game's bank is
event-shaped. Wood reaches downstream games through providers that compose these
layers into events. A game pack here would
be this generator guessing at an event, which is exactly what the material/event
cut exists to prevent.
"""

from __future__ import annotations

from ..._audio.models import Honours, normalize_honours

from ..._audio.models import VariationFilenames

from ..._audio.models import RenderResult as Render

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import engine


@dataclass(frozen=True)
class Options:
    """Every supported render axis; cues ignore fields they do not honour."""

    form: engine.WoodForm = engine.FORMS["plank"]
    intensity: float = 0.9




ALL_FORMS: tuple[str, ...] = tuple(engine.FORMS)


@dataclass(frozen=True)
class Cue:
    name: str
    group: str
    description: str
    render: Callable[..., tuple[np.ndarray, list[str]]]
    peak_db: float
    honours: Honours | str | tuple[str, ...] = ""
    # Recordings this cue cannot run without. Empty means "whatever the source
    # families offer"; a non-empty tuple is checked before rendering so a cue
    # waiting on a hand-fetched master fails with a reason rather than a
    # traceback.
    requires: tuple[str, ...] = ()
    # Which forms this cue is offered at. Not every cue spans every form -- that
    # was the finding of the 2026-07-16 audition, and the matrix used to pretend
    # otherwise. A twig is too small to groan, settle, or scatter; a crack does
    # not survive being retuned. An empty tuple means the cue is form-flat: it
    # renders once, ignoring form entirely. See the per-cue notes below and
    # `docs/technique-selection.md`.
    forms: tuple[str, ...] = ALL_FORMS

    def missing(self) -> tuple[str, ...]:
        return engine.missing_sources(self.requires) if self.requires else ()

    def available(self) -> bool:
        return not self.missing()

    def form_flat(self) -> bool:
        return not self.forms

    def supports(self, form_name: str) -> bool:
        return self.form_flat() or form_name in self.forms

    def offered_forms(self) -> tuple[str, ...]:
        """The forms to render in a pack: the real set, or one neutral render."""
        return self.forms if self.forms else (engine.NEUTRAL_FORM,)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "honours", normalize_honours(self.honours)
        )


# The form sets below are by-ear verdicts from 2026-07-16, not guesses. The
# rejected combinations are recorded in each cue's note so a future tuning pass
# knows they were heard and cut, not overlooked.
CUES: dict[str, Cue] = {
    "strike": Cue(
        "strike", "contact", "wood struck without breaking; hollow, no pitch",
        engine.strike, -9.0, "form, intensity",
    ),
    "drop": Cue(
        # twig rejected by ear: a stick is too light to land, lift, and settle
        # the way this cue is about. It reads as a heavier object.
        "drop", "contact", "a piece landing and settling; wood bounces",
        engine.drop, -10.0, "form, intensity",
        forms=("plank", "crate", "beam"),
    ),
    "break": Cue(
        "break", "fracture", "wood failing: the break, the tear, the pieces",
        engine.fracture, -8.0, "form, intensity",
    ),
    "splinter": Cue(
        # Form-flat by ear: only the untransformed render passed. A crack is a
        # precise-template sound -- the ear knows it exactly, so any varispeed or
        # damping to suit a form reads as fake. beam, crate and twig were all
        # rejected; plank (no varispeed, no tilt) is what ships, for every form.
        "splinter", "fracture", "cracking under strain, still in one piece",
        engine.splinter, -11.0, "intensity",
        forms=(),
    ),
    "debris": Cue(
        # twig and plank rejected: a light, sparse scatter of small wood does not
        # read as wood. The two heavier forms, denser and lower, carry it.
        "debris", "fracture", "the pieces scattering and coming to rest",
        engine.debris, -11.0, "form, intensity",
        forms=("crate", "beam"),
    ),
    "creak": Cue(
        # twig rejected: a stick cannot store the energy to creak; it snaps,
        # which is `break`. A creak needs a timber long enough to groan.
        "creak", "stress", "timber under load; not a hinge - see door_sound_generator",
        engine.creak, -14.0, "form, intensity",
        forms=("plank", "crate", "beam"),
    ),
    "scrape": Cue(
        # Forms unpruned: scrape has not been auditioned (its masters are
        # pending), so its form set is not yet a by-ear verdict like the others.
        "scrape", "stress", "wood dragged across a surface; sustained friction",
        engine.scrape, -12.0, "form, intensity",
        requires=engine.SCRAPE_SOURCES,
    ),
}

GROUPS = ("contact", "fracture", "stress")


def render(cue_name: str, options: Options,
           rng: np.random.Generator) -> Render:
    """Render one cue, mastered and ready to write."""

    if cue_name not in CUES:
        raise ValueError(
            f"unknown cue {cue_name!r}; expected one of {sorted(CUES)}"
        )
    cue = CUES[cue_name]
    if not cue.supports(options.form.name):
        raise ValueError(
            f"{cue_name!r} is not offered at form {options.form.name!r}; "
            f"it was rejected by ear on 2026-07-16. Offered forms: "
            f"{', '.join(cue.forms)}."
        )
    form = engine.FORMS[engine.NEUTRAL_FORM] if cue.form_flat() else options.form
    signal, sources = cue.render(form, rng, options.intensity)
    return Render(engine.master(signal, cue.peak_db), tuple(sources))


# --------------------------------------------------------------------------
# packs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PackEntry:
    filename: str
    cue: str
    count: int
    options: Options = Options()


@dataclass(frozen=True)
class Pack(VariationFilenames):
    name: str
    description: str
    entries: tuple[PackEntry, ...]



def _options(form: str, **overrides) -> Options:
    return Options(form=engine.FORMS[form], **overrides)


def _showcase_entries() -> tuple[PackEntry, ...]:
    """One render of every cue at every form it is *offered* at.

    Not every form: the 2026-07-16 audition cut the combinations that had no
    physical referent, and this pack follows the cue registry rather than the
    full cross-product. A form-flat cue (`splinter`) appears once, with no form
    suffix, because rendering it four times would write four identical files.
    """
    entries = []
    for name, cue in CUES.items():
        if cue.form_flat():
            entries.append(PackEntry(name, name, 1, Options()))
        else:
            for form in cue.forms:
                entries.append(PackEntry(f"{name}_{form}", name, 1,
                                         _options(form)))
    return tuple(entries)


# One of everything the audition kept. This is the audition bank: the roadmap's
# standing instruction is that nothing here is decided correctly in advance of
# hearing it, so the pack that makes hearing all of it one command earns its
# place more than a speculative game bank would.
SHOWCASE = Pack(
    name="showcase",
    description="one of every cue at every form it survived, for auditioning",
    entries=_showcase_entries(),
)

# The material layers item 11 composes, at the counts a projectile impact wants
# to rotate between. Not a game bank -- a *library* bank, and the reason this
# generator exists upstream of that one. Every entry is a form the audition
# kept: no wood_debris_plank (rejected), splinter is form-flat.
LIBRARY = Pack(
    name="library",
    description="the wood layers projectile_sounds and crash_generator consume",
    entries=(
        PackEntry("wood_strike_plank", "strike", 4, _options("plank")),
        PackEntry("wood_strike_beam", "strike", 4, _options("beam")),
        PackEntry("wood_break_plank", "break", 4, _options("plank")),
        PackEntry("wood_break_crate", "break", 4, _options("crate")),
        PackEntry("wood_splinter", "splinter", 4, Options()),
        PackEntry("wood_debris_crate", "debris", 4, _options("crate")),
        PackEntry("wood_debris_beam", "debris", 4, _options("beam")),
    ),
)

PACKS = {pack.name: pack for pack in (SHOWCASE, LIBRARY)}
