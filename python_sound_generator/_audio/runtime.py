"""Shared CLI runtime services: seeds, metadata, manifests, and playback."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = 1
SEED_SCHEME = "stable-v2"
LEGACY_SEED_SCHEME = "legacy-v1"


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int = 44_100
    channels: int = 1
    sample_width: int = 2
    encoding: str = "signed PCM WAV"

    @property
    def description(self) -> str:
        layout = "mono" if self.channels == 1 else "stereo"
        return (
            f"{layout} {self.sample_rate} Hz signed "
            f"{self.sample_width * 8}-bit PCM WAV"
        )

    def as_dict(self) -> dict[str, int | str]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "encoding": self.encoding,
        }


@dataclass(frozen=True, slots=True)
class AudioStats:
    frames: int
    duration_seconds: float
    peak: float
    rms: float


def audio_stats(samples: Sequence[float], sample_rate: int = 44_100) -> AudioStats:
    """Measure finite sample sequences without requiring NumPy."""

    frames = len(samples)
    peak = 0.0
    energy = 0.0
    for raw in samples:
        sample = float(raw)
        if not math.isfinite(sample):
            raise ValueError("audio contains non-finite samples")
        peak = max(peak, abs(sample))
        energy += sample * sample
    return AudioStats(
        frames=frames,
        duration_seconds=frames / sample_rate,
        peak=peak,
        rms=math.sqrt(energy / max(1, frames)),
    )


def file_sha256(path: str | Path) -> str:
    """Hash a completed output artifact in bounded chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(
    path: str | Path,
    samples: Sequence[float],
    *,
    cue: str,
    seed: int,
    variation: int,
    root: str | Path | None = None,
    sources: Sequence[str] = (),
    options: Mapping[str, Any] | None = None,
    loop: bool | None = None,
    seam: Mapping[str, Any] | None = None,
    sample_rate: int = 44_100,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common portion of one rendered-artifact manifest entry."""

    target = Path(path)
    relative = target.relative_to(Path(root)) if root is not None else Path(target.name)
    stats = audio_stats(samples, sample_rate)
    record: dict[str, Any] = {
        "file": relative.as_posix(),
        "cue": cue,
        "seed": seed,
        "variation": variation,
        "duration_seconds": round(stats.duration_seconds, 6),
        "peak": round(stats.peak, 8),
        "rms": round(stats.rms, 8),
        "sha256": file_sha256(target),
        "sources": list(sources),
    }
    if options is not None:
        record["options"] = dict(options)
    if loop is not None:
        record["loop"] = bool(loop)
    if seam is not None:
        record["seam"] = dict(seam)
    if extra:
        record.update(extra)
    return record


def artifact_metrics(
    path: str | Path,
    samples: Sequence[float],
    sample_rate: int = 44_100,
) -> dict[str, float | str]:
    """Return the repeated duration/level/checksum block used by CLIs."""

    stats = audio_stats(samples, sample_rate)
    return {
        "duration_seconds": round(stats.duration_seconds, 6),
        "peak_dbfs": round(20.0 * math.log10(max(stats.peak, 1e-12)), 3),
        "rms_dbfs": round(20.0 * math.log10(max(stats.rms, 1e-12)), 3),
        "sha256": file_sha256(path),
    }


def _canonical_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize historical artifact shapes without discarding extensions."""

    row = dict(artifact)
    legacy_loop = row.get("loop")
    if isinstance(legacy_loop, Mapping):
        row["loop"] = True
        row.setdefault("seam", dict(legacy_loop))
    else:
        row["loop"] = bool(legacy_loop)
    row.setdefault("sources", [])
    return row


def manifest_payload(
    *,
    generator: str,
    command: str,
    artifacts: Sequence[Mapping[str, Any]],
    audio_format: AudioFormat = AudioFormat(),
    seed_scheme: str = SEED_SCHEME,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the canonical manifest envelope with extension fields."""

    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator": generator,
        "command": command,
        "format": audio_format.description,
        "audio": audio_format.as_dict(),
        "seed_scheme": seed_scheme,
    }
    extension = dict(extra or {})
    provenance_keys = (
        "source_audio",
        "source_licensing",
        "source_provenance",
        "delegated_provenance",
        "synthesis",
    )
    provenance = {
        key: extension[key] for key in provenance_keys if key in extension
    }
    if provenance and "provenance" not in extension:
        extension["provenance"] = provenance
    if extension:
        reserved = set(payload) | {"artifacts"}
        collision = reserved.intersection(extension)
        if collision:
            names = ", ".join(sorted(collision))
            raise ValueError(f"manifest extension overwrites reserved fields: {names}")
        payload.update(extension)
    payload["artifacts"] = [_canonical_artifact(item) for item in artifacts]
    if any(item.get("sources") for item in payload["artifacts"]):
        payload["asset_catalog"] = "assets/catalog.json"
    return payload


def write_manifest(
    path: str | Path,
    *,
    generator: str,
    command: str,
    artifacts: Sequence[Mapping[str, Any]],
    audio_format: AudioFormat = AudioFormat(),
    seed_scheme: str = SEED_SCHEME,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a canonical, deterministically formatted JSON manifest."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_payload(
        generator=generator,
        command=command,
        artifacts=artifacts,
        audio_format=audio_format,
        seed_scheme=seed_scheme,
        extra=extra,
    )
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def derive_seed(
    base_seed: int,
    *,
    provider: str,
    cue: str,
    filename: str = "",
    variation: int = 1,
) -> int:
    """Derive a stable seed whose value does not depend on pack order."""

    if variation < 1:
        raise ValueError("variation must be at least 1")
    identity = json.dumps(
        {
            "base_seed": int(base_seed),
            "cue": cue,
            "filename": Path(filename).as_posix() if filename else "",
            "provider": provider,
            "variation": variation,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.blake2b(identity, digest_size=8, person=b"sndgen-v2").digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def legacy_seed(
    base_seed: int,
    *,
    index: int,
    name: str = "",
    stride: int = 1009,
) -> int:
    """Reproduce the pre-v2 positional seed convention when compatibility matters."""

    return int(base_seed) + int(index) * stride + len(name)


_PLAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("paplay", ()),
    ("aplay", ()),
    ("afplay", ()),
    ("play", ("-q",)),
    ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "error")),
)


def play(path: str | Path) -> bool:
    """Play an audio file with the first available system player."""

    target = Path(path).expanduser().resolve()
    for executable, arguments in _PLAYERS:
        player = shutil.which(executable)
        if player is None:
            continue
        completed = subprocess.run(
            [player, *arguments, str(target)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            return True
    return False


__all__ = [
    "AudioFormat",
    "AudioStats",
    "LEGACY_SEED_SCHEME",
    "MANIFEST_SCHEMA_VERSION",
    "SEED_SCHEME",
    "artifact_record",
    "artifact_metrics",
    "audio_stats",
    "derive_seed",
    "file_sha256",
    "legacy_seed",
    "manifest_payload",
    "play",
    "write_manifest",
]
