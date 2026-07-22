"""Private namespace containing all bundled sound-domain providers."""

from __future__ import annotations

from pathlib import Path


PROVIDER_ROOT = Path(__file__).resolve().parent


__all__ = ["PROVIDER_ROOT"]
