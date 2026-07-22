"""Compatibility entry point for direct execution and simple imports."""

try:
    from .engine import *  # noqa: F401,F403
    from .engine import main
except ImportError:  # Allows: python door_sound_generator/door_sound_generator.py
    from engine import *  # type: ignore  # noqa: F401,F403
    from engine import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())
