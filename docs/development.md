# Installing, verifying and building Python Sound Generator

Installation, the verification suite and reproducible release builds, moved here from the README.

## Install

Python 3.11 or newer is required.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

For a source checkout with dependencies already installed, the module form also
works directly:

```sh
python3 -m python_sound_generator list
```

## Verification

```sh
make verify
make build
python -m twine check dist/*
```

Set `SOURCE_DATE_EPOCH` to a non-negative Unix timestamp when producing release
artifacts. The wheel and source distribution then use canonical archive times
and ownership metadata, so rebuilding the same tree with the same locked build
environment produces byte-identical artifacts.

Run `make check-reproducible-sdist` to build the complete source archive twice
and verify its bytes, entry order, timestamps, and ownership metadata.

`verify` runs Ruff, the complete test suite, and the 32-provider health check.
CI repeats those checks on Python 3.11 through 3.14 and validates both built
distributions. The unified tests pin registry uniqueness, namespace-safe lazy loading,
compatibility aliases, DSP PCM characterization hashes, stable seeds, manifest
schema, shared-buffer output, content-addressed asset integrity, all-provider
imports and help commands, packaged-data integrity, and end-to-end WAV renders
through the facade—including a default-output render from the household
appliance Foley provider.
