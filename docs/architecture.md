# Python Sound Generator architecture

The provider registry, the shared audio layer and what the distribution contains, moved here from the README together with its original introduction.

One modular command line for 32 game-audio domains. It combines deterministic
procedural DSP with approved CC0/public-domain Foley while keeping each domain
isolated behind a lazy provider registry. The external Strudel music recorder
is the documented exception to byte-for-byte deterministic rendering.

```text
soundgen CLI
    │
    ├── lazy provider registry
    │     ├── fire / water / wood / stone / …
    │     ├── combat / weapons / projectiles / gunshots
    │     └── RPG / quirky RPG / ninja / cyberpunk / ambience / UI / voices
    │
    └── neutral python_sound_generator._audio layer
          ├── standard-library DSP + loop/rotor models
          ├── NumPy/SciPy recording transforms
          ├── PCM, CLI runtime, manifests, and seed derivation
          └── content-addressed WAV assets + logical provenance aliases
```

Only the selected provider is imported. Listing the catalog does not load NumPy,
SciPy, source WAVs, or the other DSP engines. Provider modules stay beneath the
`python_sound_generator.generators` namespace, so dispatch does not publish
generic names such as `lib` or legacy provider package names into `sys.modules`.

## Shared audio architecture

The public package owns discovery, aliases, lazy loading, diagnostics, and
dispatch. Consolidated provider implementations live under
`python_sound_generator/generators/` and keep their domain-specific registries,
renderers, mastering choices, and source-selection policy. Reusable mechanics
live under `python_sound_generator/_audio/`:

- `dsp.py`, `chiptune.py`, `loops.py`, and `rotor.py` are standard-library-only;
- `foley_numpy.py` owns recording I/O, trimming, resampling, placement, loop
  construction, immutable sample caching, and NumPy PCM output;
- `foley_bank.py` owns logical source records, deterministic selection,
  safe provider paths, varispeed, filtering, and tail control for hybrid domains;
- `foley_cli.py` owns consistent source inventories and provenance checksum
  diagnostics for recording-backed command-line providers;
- `validation.py` owns finite bounded-value checks shared by public controls;
- `pcm.py` owns the common mono/stereo `AudioBuffer` and dependency-free writer;
- `runtime.py` owns playback, artifact statistics, manifest envelopes, and
  stable named seed derivation;
- `models.py` supplies small structural contracts and shared variation naming;
- `assets.py` resolves provider-level logical names through the shared catalog.

The old `generators.foley_core`, `combat_generators.lib.dsp`, and
`environment_ambience_generator.loopdsp` paths remain compatibility imports.
Importing the neutral standard-library modules never imports NumPy or SciPy.

Adding a domain requires two bounded changes:

1. add its runtime package and approved logical source names;
2. add one lazy `Provider` entry in `python_sound_generator/providers.py`.

No provider is imported to register another provider, and a broken optional
domain cannot prevent catalog discovery.

## What is included

The distribution includes runtime Python modules, approved source WAVs, source
provenance JSON, and the shared Foley core. It deliberately excludes generated
outputs, audition renders, caches, superseded research candidates, test debris,
and design/session notes.

Duplicate recordings are stored once under `python_sound_generator/assets/audio/`
by SHA-256. `assets/catalog.json` maps every original provider-relative filename
to that immutable blob and its original provenance ledger. Unique recordings
remain in their provider bank. Providers continue to report and emit logical
source names, so consolidation does not transfer authorship or licensing
ownership. The bundled recording policy is CC0 or public domain; the ledgers
remain the source of truth for each byte. Full-length Foley intake masters are
local reproducibility inputs and are deliberately excluded from Git and built
packages.
