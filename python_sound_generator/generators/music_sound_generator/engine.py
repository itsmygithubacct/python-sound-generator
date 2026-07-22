"""The hybrid render engine: Python orchestrating Strudel, not DSP in Python.

Every other generator in this project synthesises or reshapes audio in NumPy.
This one does not. It drives Strudel -- the live-coding tool checked out at
``$STRUDEL_DIR`` -- through a project-authored recorder integration, and the
audio is the *output of running that tool*:

    write recorder files + cue payloads into the strudel workspace
        -> start the vite dev server (if not already up)
        -> puppeteer-core launches system Chrome headless and loads the page
        -> the page records the Web Audio graph to webm via MediaRecorder
        -> ffmpeg converts webm -> 44100 Hz mono 16-bit WAV
        -> read the WAV back for peak/RMS and an as-rendered SHA

Why this is not linked into Python
----------------------------------
Strudel is AGPL-3.0-or-later. This generator never imports or bundles it: it
launches the tool as an external process over a local dev server and captures
what it plays. The pattern text is ours (in ``cues.py``); the synthesis engine
is Strudel's, invoked at arm's length.

Determinism -- read this before trusting a hash
-----------------------------------------------
A MediaRecorder real-time capture is **not** byte-deterministic. Two takes of
the same pattern differ by scheduler jitter and opus encoding, so there is no
seed that makes the output bytes reproducible, and this module never claims
one. What is pinned instead is the *input*: the exact Strudel pattern text
(hashed, in-package) and the configured toolchain. The WAV's SHA-256 is written to
the manifest as an **as-rendered record**, not a golden to diff against.

Loops: seamlessness from cycle math, not a crossfade
----------------------------------------------------
A loop cue is recorded for an integer number of pattern cycles
(``seconds = cycles / cps``), so the tempo grid lands exactly on the loop
point. There is no crossfade -- the join is clean because the clock is. The
``verify`` command reports both the cycle math (which we control) and the
acoustic seam of a rendered file (which carries real-time-capture latency at
its head; see the note there).
"""

from __future__ import annotations

from ..._audio.assets import resolve_asset

from ..._audio.foley_numpy import write_wav_scipy as write_wav  # noqa: F401

import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.io import wavfile

from .cues import CUES, Cue

SR = 44_100

PACKAGE_DIR = Path(__file__).resolve().parent
RECORDER_DIR = PACKAGE_DIR / "recorder"
SOURCE_DIRECTORY = PACKAGE_DIR / "sources"

RECORDER_TEMPLATE = RECORDER_DIR / "record_template.html"
RECORDER_DRIVER = RECORDER_DIR / "record_cue.mjs"

# The one servable location outside the package. Additive: created at render
# time under the strudel workspace so the vite dev server can serve it.
WORKSPACE_SUBDIR = "music_sound_generator_assets"

DEFAULT_STRUDEL_DIR = Path.home() / ".local" / "share" / "strudel"
DEFAULT_CHROME_BIN = Path("/usr/bin/google-chrome")
DEFAULT_PORT = 5398


class RecorderUnavailable(RuntimeError):
    """The external Strudel toolchain is missing a piece.

    Raised with a message that names exactly what is absent, the same way the
    NumPy generators fail loudly when a dependency is not importable -- this
    generator's dependency just happens to be a browser and a dev server rather
    than a wheel.
    """


class RecordFailed(RuntimeError):
    """A recording pass ran but did not produce a usable webm."""


# --------------------------------------------------------------------------
# render options and result
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Options:
    """Every supported render axis; cues ignore fields they do not honour.

    Music driven by a fixed pattern has few axes. ``seconds`` overrides the
    capture length (for a loop that means overriding the integer-cycle default,
    which the manifest then flags). ``seed`` changes nothing about the audio --
    the pattern is fixed and the capture is not seeded -- but it is accepted and
    recorded so the CLI matches its siblings; ``--count`` re-records, giving
    genuinely distinct real-time takes.
    """

    seconds: float | None = None
    emit_ogg: bool = False
    seed: int = 0


@dataclass(frozen=True)
class Render:
    """One rendered take: the files written and what they measured."""

    cue: str
    wav_path: Path
    webm_path: Path
    ogg_path: Path | None
    samples: np.ndarray
    seconds: float
    duration: float
    cps: float
    cycles: int | None
    loop: bool
    peak_dbfs: float
    rms_dbfs: float
    wav_sha256: str
    webm_sha256: str
    pattern_sha256: str
    source: str


# --------------------------------------------------------------------------
# pinning the pattern (the input we actually control)
# --------------------------------------------------------------------------

def pattern_sha256(cue: Cue) -> str:
    """SHA-256 of the exact pattern text. This is what is pinned in-package."""

    return hashlib.sha256(cue.pattern.encode("utf-8")).hexdigest()


def record_seconds(cue: Cue, options: Options) -> float:
    """How long to record: an override, else the cue's own length."""

    if options.seconds is not None:
        if options.seconds <= 0:
            raise ValueError(f"seconds must be positive; got {options.seconds}")
        return float(options.seconds)
    return float(cue.seconds)


def is_integer_cycles(cue: Cue, seconds: float) -> bool:
    """Does `seconds` cover a whole number of pattern cycles at this tempo?

    True by construction for a loop at its default length; this exists to catch
    a `--seconds` override that would break the clean wrap.
    """

    cycles = seconds * cue.cps
    return abs(cycles - round(cycles)) < 1e-6


# --------------------------------------------------------------------------
# toolchain configuration and validation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MusicToolchain:
    """The external dependencies, all configurable, all validated up front."""

    strudel_dir: Path
    chrome_bin: Path
    port: int
    node: str
    ffmpeg: str
    vite_bin: Path | None

    @classmethod
    def from_env(cls) -> "MusicToolchain":
        strudel_dir = Path(os.environ.get("STRUDEL_DIR", str(DEFAULT_STRUDEL_DIR)))
        chrome_bin = Path(os.environ.get("CHROME_BIN", str(DEFAULT_CHROME_BIN)))
        port = int(os.environ.get("STRUDEL_PORT", str(DEFAULT_PORT)))
        return cls(
            strudel_dir=strudel_dir,
            chrome_bin=chrome_bin,
            port=port,
            node=shutil.which("node") or "node",
            ffmpeg=shutil.which("ffmpeg") or "ffmpeg",
            vite_bin=_find_vite_bin(strudel_dir),
        )

    @property
    def workspace_assets(self) -> Path:
        return self.strudel_dir / WORKSPACE_SUBDIR

    def problems(self) -> list[str]:
        """Everything missing, so the caller can report all of it at once."""

        issues: list[str] = []
        if not self.strudel_dir.is_dir():
            issues.append(
                f"STRUDEL_DIR {self.strudel_dir} is not a directory "
                f"(set STRUDEL_DIR to the strudel workspace)"
            )
        else:
            if self.vite_bin is None or not self.vite_bin.is_file():
                issues.append(
                    f"vite is not installed under {self.strudel_dir}/node_modules "
                    f"(run the pnpm install from the strudel findings doc)"
                )
            if not (self.strudel_dir / "node_modules" / "puppeteer-core").exists():
                issues.append(
                    f"puppeteer-core is not installed under "
                    f"{self.strudel_dir}/node_modules"
                )
            if not (self.strudel_dir / "node_modules" / "@strudel" / "web").exists():
                issues.append(
                    f"@strudel/web is not linked under "
                    f"{self.strudel_dir}/node_modules/@strudel"
                )
        if shutil.which(self.node) is None and not Path(self.node).is_file():
            issues.append("node was not found on PATH")
        if not self.chrome_bin.exists():
            issues.append(
                f"CHROME_BIN {self.chrome_bin} does not exist "
                f"(set CHROME_BIN to a Chrome/Chromium binary)"
            )
        if shutil.which(self.ffmpeg) is None and not Path(self.ffmpeg).is_file():
            issues.append("ffmpeg was not found on PATH")
        if not RECORDER_TEMPLATE.is_file():
            issues.append(f"packaged recorder template missing: {RECORDER_TEMPLATE}")
        if not RECORDER_DRIVER.is_file():
            issues.append(f"packaged recorder driver missing: {RECORDER_DRIVER}")
        return issues

    def validate(self) -> None:
        issues = self.problems()
        if issues:
            listing = "\n".join(f"  - {issue}" for issue in issues)
            raise RecorderUnavailable(
                "music_sound_generator renders by driving Strudel (an external "
                "tool), and part of that toolchain is missing:\n"
                f"{listing}\n"
                "See the unified project's music-provider setup notes."
            )


def _find_vite_bin(strudel_dir: Path) -> Path | None:
    direct = strudel_dir / "node_modules" / "vite" / "bin" / "vite.js"
    if direct.is_file():
        return direct
    matches = sorted(
        (strudel_dir / "node_modules" / ".pnpm").glob(
            "vite@*/node_modules/vite/bin/vite.js"
        )
    )
    return matches[0] if matches else None


# --------------------------------------------------------------------------
# workspace: recorder files + one payload per cue
# --------------------------------------------------------------------------

def ensure_workspace(tc: MusicToolchain,
                     cues: dict[str, Cue] = CUES) -> Path:
    """Write the servable recorder + cue payloads into the strudel workspace.

    This is the only write outside the package, and it is additive: a folder
    that did not exist, containing a copy of the packaged recorder template and
    one ``<cue>.cue.js`` per cue. The cue payload sets ``window.__CUE__`` so the
    recorder loads a pattern without any dev-server JSON-transform surprises.
    """

    dst = tc.workspace_assets
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RECORDER_TEMPLATE, dst / "record.html")
    shutil.copyfile(RECORDER_DRIVER, dst / "record_cue.mjs")
    for cue in cues.values():
        payload = {
            "name": cue.name,
            "code": cue.pattern,
            "seconds": record_seconds(cue, Options()),
        }
        (dst / f"{cue.name}.cue.js").write_text(
            "window.__CUE__ = " + json.dumps(payload) + ";\n",
            encoding="utf-8",
        )
    return dst


# --------------------------------------------------------------------------
# vite lifecycle
# --------------------------------------------------------------------------

def vite_is_up(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def start_vite(tc: MusicToolchain, timeout: float = 40.0) -> subprocess.Popen:
    """Start the vite dev server and wait until it is accepting connections."""

    if tc.vite_bin is None:
        raise RecorderUnavailable("vite binary not found under the strudel workspace")
    proc = subprocess.Popen(
        [tc.node, str(tc.vite_bin), "--host", "127.0.0.1", "--port", str(tc.port)],
        cwd=str(tc.strudel_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RecorderUnavailable(
                f"vite exited early (code {proc.returncode}); check the strudel "
                f"workspace at {tc.strudel_dir}"
            )
        if vite_is_up(tc.port):
            return proc
        time.sleep(0.25)
    stop_vite(proc)
    raise RecorderUnavailable(f"vite did not come up on port {tc.port} within {timeout}s")


def stop_vite(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# --------------------------------------------------------------------------
# one recording pass
# --------------------------------------------------------------------------

def record_webm(tc: MusicToolchain, cue_name: str, seconds: float,
                dest_dir: Path, out_name: str) -> Path:
    """Drive puppeteer to record one cue to ``dest_dir/out_name.webm``."""

    dest_dir.mkdir(parents=True, exist_ok=True)
    driver = tc.workspace_assets / "record_cue.mjs"
    env = dict(os.environ)
    env["STRUDEL_PORT"] = str(tc.port)
    env["CHROME_BIN"] = str(tc.chrome_bin)
    result = subprocess.run(
        [tc.node, str(driver), cue_name, f"{seconds}", str(dest_dir), out_name],
        cwd=str(tc.strudel_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=max(90.0, seconds + 75.0),
    )
    webm = dest_dir / f"{out_name}.webm"
    if result.returncode != 0 or not webm.is_file():
        tail = "\n".join(result.stderr.strip().splitlines()[-12:])
        raise RecordFailed(
            f"recording {cue_name!r} failed (exit {result.returncode}).\n{tail}"
        )
    if webm.stat().st_size == 0:
        raise RecordFailed(f"recording {cue_name!r} produced an empty webm")
    return webm


def convert_wav(tc: MusicToolchain, webm: Path, wav: Path) -> Path:
    """webm -> 44100 Hz mono signed 16-bit PCM WAV, via ffmpeg."""

    wav.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [tc.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(webm), "-ar", str(SR), "-ac", "1",
         "-c:a", "pcm_s16le", str(wav)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0 or not wav.is_file():
        raise RecordFailed(f"ffmpeg wav conversion failed:\n{result.stderr.strip()}")
    return wav


def convert_ogg(tc: MusicToolchain, webm: Path, ogg: Path) -> Path:
    ogg.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [tc.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(webm), "-c:a", "libvorbis", "-q:a", "5", str(ogg)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0 or not ogg.is_file():
        raise RecordFailed(f"ffmpeg ogg conversion failed:\n{result.stderr.strip()}")
    return ogg


# --------------------------------------------------------------------------
# reading the render back
# --------------------------------------------------------------------------

def decode_wav(path: Path) -> np.ndarray:
    """A mono float64 view of a rendered WAV, in [-1, 1]."""

    path = resolve_asset(path)
    sample_rate, pcm = wavfile.read(path)
    if int(sample_rate) != SR:
        raise ValueError(f"{path.name} is {sample_rate} Hz; expected {SR}")
    data = np.asarray(pcm)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype.kind == "i":
        info = np.iinfo(data.dtype)
        return data.astype(np.float64) / max(abs(info.min), info.max)
    if data.dtype.kind == "f":
        return data.astype(np.float64)
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    raise ValueError(f"unsupported WAV dtype {data.dtype} in {path}")


def _dbfs(value: float) -> float:
    return round(20.0 * float(np.log10(max(value, 1e-12))), 3)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()




# --------------------------------------------------------------------------
# a recording session: one workspace, one vite, many takes
# --------------------------------------------------------------------------

class Session:
    """A live recording context. Each ``render`` is one browser capture."""

    def __init__(self, toolchain: MusicToolchain,
                 cues: dict[str, Cue] = CUES) -> None:
        self.tc = toolchain
        self.cues = cues

    def render(self, cue_name: str, options: Options, dest_dir: Path,
               stem: str) -> Render:
        if cue_name not in self.cues:
            raise ValueError(
                f"unknown cue {cue_name!r}; expected one of {sorted(self.cues)}"
            )
        cue = self.cues[cue_name]
        seconds = record_seconds(cue, options)
        dest_dir = Path(dest_dir)

        webm = record_webm(self.tc, cue_name, seconds, dest_dir, stem)
        wav = convert_wav(self.tc, webm, dest_dir / f"{stem}.wav")
        ogg = None
        if options.emit_ogg:
            ogg = convert_ogg(self.tc, webm, dest_dir / f"{stem}.ogg")

        samples = decode_wav(wav)
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
        return Render(
            cue=cue_name,
            wav_path=wav,
            webm_path=webm,
            ogg_path=ogg,
            samples=samples,
            seconds=seconds,
            duration=round(len(samples) / SR, 6),
            cps=cue.cps,
            cycles=cue.cycles,
            loop=cue.loop,
            peak_dbfs=_dbfs(peak),
            rms_dbfs=_dbfs(rms),
            wav_sha256=sha256_file(wav),
            webm_sha256=sha256_file(webm),
            pattern_sha256=pattern_sha256(cue),
            source=cue.source,
        )


@contextmanager
def recording_session(toolchain: MusicToolchain | None = None,
                      cues: dict[str, Cue] = CUES) -> Iterator[Session]:
    """Validate the toolchain, stage the workspace, and start vite if needed.

    If a vite server is already listening on the port it is reused and left
    running; only a server this session starts is stopped on exit.
    """

    tc = toolchain or MusicToolchain.from_env()
    tc.validate()
    ensure_workspace(tc, cues)
    started = None
    if not vite_is_up(tc.port):
        started = start_vite(tc)
    try:
        yield Session(tc, cues)
    finally:
        if started is not None:
            stop_vite(started)


def render(cue_name: str, options: Options | None = None, *,
           dest_dir: str | Path, stem: str | None = None,
           toolchain: MusicToolchain | None = None) -> Render:
    """Render one cue end to end, opening a session for the single take.

    Convenience for one-offs. The CLI opens a single session and reuses it
    across a whole pack rather than paying the vite/workspace cost per file.
    """

    options = options or Options()
    with recording_session(toolchain) as session:
        return session.render(cue_name, options, Path(dest_dir),
                              stem or cue_name)
