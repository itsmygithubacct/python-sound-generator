"""Music for games, rendered by driving Strudel (an external tool) headless.

This generator does not do DSP in Python. It owns a registry of music cues --
each a pure-synth Strudel pattern -- and renders them by writing a recorder page
into the strudel workspace, starting the vite dev server, driving system Chrome
via puppeteer to record the Web Audio graph, and converting the capture to WAV
with ffmpeg. The recorder integration is project-authored and invokes the
separately installed Strudel toolchain through its public API.
"""

from .cues import CUES, GROUPS, PACKS, Cue, Pack, PackEntry
from .engine import (
    SR,
    MusicToolchain,
    Options,
    RecordFailed,
    RecorderUnavailable,
    Render,
    Session,
    ensure_workspace,
    pattern_sha256,
    record_seconds,
    recording_session,
    render,
    write_wav,
)

__all__ = [
    "CUES",
    "GROUPS",
    "PACKS",
    "SR",
    "Cue",
    "MusicToolchain",
    "Options",
    "Pack",
    "PackEntry",
    "RecordFailed",
    "RecorderUnavailable",
    "Render",
    "Session",
    "ensure_workspace",
    "pattern_sha256",
    "record_seconds",
    "recording_session",
    "render",
    "write_wav",
]
