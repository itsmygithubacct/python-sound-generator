"""Dependency-free audio buffers and PCM16 WAV output."""

from __future__ import annotations

import array
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


Channel = Sequence[float]


@dataclass
class AudioBuffer:
    """One or two channels of floating-point PCM plus source provenance."""

    sample_rate: int
    channels: tuple[Channel, ...]
    sources: tuple[str, ...] = ()

    @property
    def frame_count(self) -> int:
        return len(self.channels[0]) if self.channels else 0

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def duration(self) -> float:
        return self.frame_count / self.sample_rate

    def validation_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.sample_rate <= 0:
            errors.append("sample rate must be positive")
        if self.channel_count not in (1, 2):
            errors.append("WAV output supports one or two channels")
        if self.frame_count < 1:
            errors.append("audio buffer is empty")
        if any(len(channel) != self.frame_count for channel in self.channels):
            errors.append("all channels must have the same number of frames")
        if any(
            not math.isfinite(float(sample))
            for channel in self.channels
            for sample in channel
        ):
            errors.append("audio contains non-finite samples")
        return tuple(errors)


def write_pcm16(path: str | Path, audio: AudioBuffer) -> Path:
    """Write an audio buffer as little-endian signed 16-bit PCM WAV."""

    errors = audio.validation_errors()
    if errors:
        raise ValueError(errors[0])

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = array.array("h")
    append = frames.append
    for frame in range(audio.frame_count):
        for channel in audio.channels:
            sample = max(-1.0, min(1.0, float(channel[frame])))
            append(round(sample * 32_767.0))
    if sys.byteorder != "little":
        frames.byteswap()

    with wave.open(str(output), "wb") as wav_file:
        wav_file.setnchannels(audio.channel_count)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(frames.tobytes())
    return output


def write_mono_pcm16(
    path: str | Path,
    samples: Sequence[float],
    sample_rate: int = 44_100,
) -> Path:
    """Convenience wrapper for the common mono case."""

    return write_pcm16(path, AudioBuffer(sample_rate, (samples,)))


__all__ = ["AudioBuffer", "Channel", "write_mono_pcm16", "write_pcm16"]
