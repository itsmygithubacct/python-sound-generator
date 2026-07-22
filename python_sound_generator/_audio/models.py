"""Small shared data contracts used across otherwise independent generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable


SamplesT = TypeVar("SamplesT")
OptionsT = TypeVar("OptionsT")


class Honours(tuple[str, ...]):
    """Canonical option-name tuple with friendly legacy string conversion."""

    def __new__(cls, values: str | tuple[str, ...] = ()) -> "Honours":
        if isinstance(values, str):
            values = tuple(
                item.strip() for item in values.split(",") if item.strip()
            )
        return super().__new__(cls, values)

    def __str__(self) -> str:
        return ", ".join(self)


def normalize_honours(values: str | tuple[str, ...]) -> Honours:
    return Honours(values)


@dataclass(frozen=True)
class RenderResult(Generic[SamplesT]):
    """Rendered PCM together with the logical recordings used to make it."""

    samples: SamplesT
    sources: tuple[str, ...] = ()


@runtime_checkable
class RenderLike(Protocol[SamplesT]):
    """Structural interface accepted by shared CLI artifact helpers."""

    samples: SamplesT
    sources: tuple[str, ...]


@runtime_checkable
class PackEntryLike(Protocol):
    """Structural interface shared by domain-specific pack entry records."""

    filename: str
    cue: str
    count: int


class VariationFilenames:
    """Shared ``name.wav`` / ``name_v02.wav`` naming convention."""

    @staticmethod
    def filename(entry: PackEntryLike, index: int) -> str:
        if index < 0:
            raise ValueError("variation index cannot be negative")
        if index == 0:
            return f"{entry.filename}.wav"
        return f"{entry.filename}_v{index + 1:02d}.wav"


@dataclass(frozen=True)
class PackEntry(Generic[OptionsT]):
    """The common four-field pack entry used by most sound domains."""

    filename: str
    cue: str
    count: int
    options: OptionsT


@dataclass(frozen=True)
class Pack(VariationFilenames, Generic[OptionsT]):
    """A named collection of common pack entries."""

    name: str
    description: str
    entries: tuple[PackEntry[OptionsT], ...]

    @staticmethod
    def variation_filename(filename: str, variation: int, count: int) -> str:
        if count <= 1:
            return filename
        path, dot, suffix = filename.rpartition(".")
        if not dot:
            return f"{filename}_{variation:02d}"
        return f"{path}_{variation:02d}.{suffix}"


__all__ = [
    "Pack",
    "PackEntry",
    "PackEntryLike",
    "RenderLike",
    "RenderResult",
    "VariationFilenames",
    "Honours",
    "normalize_honours",
]
