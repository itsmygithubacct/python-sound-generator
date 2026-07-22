"""One public interface for the modular sound-generator collection."""

from __future__ import annotations

from .api import get_provider, providers, run

__version__ = "0.2.0"

__all__ = ["__version__", "get_provider", "providers", "run"]
