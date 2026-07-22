"""Small programmatic API for the unified sound generator."""

from __future__ import annotations

from collections.abc import Sequence

from .providers import PROVIDERS, Provider, provider_named


def providers() -> tuple[Provider, ...]:
    """Return the immutable provider catalog without importing DSP engines."""
    return PROVIDERS


def get_provider(name: str) -> Provider:
    """Resolve a canonical provider name or compatibility alias."""
    return provider_named(name)


def run(name: str, args: Sequence[str] = ()) -> int:
    """Run one provider with its native CLI arguments.

    Unknown providers raise ``KeyError`` and provider import errors remain visible
    to callers. A provider's normal ``argparse`` exit is converted to a return
    code, matching the top-level command-line dispatcher.
    """
    return get_provider(name).invoke(args)


__all__ = ["get_provider", "providers", "run"]
