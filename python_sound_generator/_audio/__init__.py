"""Shared audio infrastructure with no eager optional dependencies.

Import concrete backends explicitly.  Keeping this package initializer empty is
intentional: standard-library generators must not import NumPy or SciPy merely
because another backend is installed alongside them.
"""

__all__: tuple[str, ...] = ()
