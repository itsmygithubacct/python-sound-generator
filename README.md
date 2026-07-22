# Python Sound Generator

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

## Unified CLI

```sh
soundgen list
soundgen info fire
soundgen doctor

soundgen fire list
soundgen fire render lava_flow --seconds 8 --out output/lava
soundgen ui pack aquarium --out output/aquarium
soundgen rpg render coin_pickup --theme fantasy --count 3
soundgen rpg pack core --theme retro --out output/rpg
soundgen cyberpunk render scan_ping --style electric --count 3
soundgen cyberpunk pack core --style noir --play
soundgen quirky-rpg pack core --style suburban --play
soundgen recipe validate
soundgen recipe render cyber-runner --cue ui.cursor --out output/recipes
soundgen ninja render katana_clash --style arcade --count 3
soundgen ninja pack core --style shadow --play
soundgen car-engine output/engine.wav --duration 4 --engine v8 --seed 7
soundgen car-engine sample output/mini.wav --duration 4 --seed 7
soundgen car-engine inspect
```

`soundgen PROVIDER ...` passes the remaining arguments unchanged to that
provider's established CLI. This preserves each domain's meaningful axes rather
than forcing wood forms, firearm mechanisms, musical styles, vehicle surfaces,
and appliance speeds into one misleading universal option schema. Use
`soundgen PROVIDER --help` for that provider's exact interface.

The packaged `soundgen.recipe.v1` file defines four generic example cue banks:
Fantasy Adventure, Cyber Runner, Fantasy Quest, and Suburban Odyssey. Rendering delegates
to the real provider CLIs and writes an aggregate manifest without timestamps
or absolute source paths. Validation checks the provider's real cue and
style/theme catalogs, bounds recipe sizes and counts, rejects output traversal
and symlink escapes, and verifies each rendered WAV against its provider
manifest before writing aggregate manifest checksums. The current recipes are
procedural-only. If sampled Foley is later added, validation accepts only
CC0/public-domain records with valid source and license URLs, an ISO access
date, creator, non-empty attribution record, and original SHA-256.

The car provider keeps its original procedural shorthand and adds `sample` for
the bundled CC0 recording bank, `inspect` for its RPM/load anchors and
provenance, and `render-bank` for compatible external banks.

Provider module names are accepted as compatibility aliases:

```sh
soundgen fire_sound_generator list
soundgen projectile_sounds list
```

Machine-readable catalog and health output are available with `--json`:

```sh
soundgen list --json
soundgen info appliances --json
soundgen doctor --json
```

`doctor` verifies every entrypoint, required runtime file, JSON asset, and WAV
container before reporting provider import health. Catalog rows also distinguish
conventional `sources/` files from audio stored in other runtime banks.

The same lazy boundary is available to Python callers:

```python
from python_sound_generator import get_provider, providers, run

print([provider.slug for provider in providers()])
print(get_provider("household-appliance").description)
status = run("fire", ["render", "lava_flow", "--seconds", "8"])
```

`run()` preserves each provider's native arguments and returns its process-style
status code. Unknown provider names raise `KeyError`; import and render errors
remain visible to the caller.

## RPG sound library

The `rpg` provider adds 64 deterministic one-shots for a complete party-RPG
sound vocabulary: menus, inventory, equipment, items, physical battle, elemental
magic, summons, progression, field mechanisms, vehicles, saving, and rest.
Every cue can be rendered in `fantasy`, `dark`, or `retro` themes. The `core`
pack emits one of every cue in group directories; `inventory` and `progression`
emit variation banks for repeated events.

```sh
soundgen rpg list
soundgen rpg sources
soundgen rpg info level_up --theme dark
soundgen rpg render chest_open --theme fantasy --foley-mix 0.75 --seed 42
soundgen rpg pack core --theme retro --out output/rpg
soundgen rpg pack core --theme fantasy --play  # audition every cue in order
```

Python callers can synthesize samples without going through the CLI. Importing
the dedicated module opts into its DSP engine without making top-level provider
discovery eager:

```python
from python_sound_generator.rpg import generate_with_sources, write_wav

result = generate_with_sources(
    "potion_drink", theme="fantasy", foley_mix=0.75, seed=42
)
write_wav("output/potion_drink.wav", result.samples)
print(result.sources)
```

Generated WAVs are mono, 44.1 kHz, signed 16-bit PCM. Render and pack commands
also write a manifest containing cue options, deterministic seeds, duration,
levels, SHA-256 checksums, and exact logical source names. Its default
`--foley-mix 0` path remains procedural and standard-library-only. A nonzero
mix adds approved physical layers to seven cues while progression, dialogue,
encounter, save, and rest identities remain synthesized.

## Cyberpunk game sound library

The `cyberpunk` provider supplies 64 deterministic one-shots across interface,
network, security, cyberware, world, combat, stealth, matrix, and magic groups.
Its `electric`, `noir`, and `corporate` styles preserve event meanings while
changing pitch, bandwidth, instability, distortion, and low-frequency weight.
The synthesis combines FM-like tones, packet pulses, quantised oscillators,
filtered radio static, sub impacts, and deterministic glitch processing.

```sh
soundgen cyberpunk list
soundgen cyberpunk sources
soundgen cyberpunk info hack_success --style corporate
soundgen cyberpunk render emp_blast --style noir --count 3 --seed 42
soundgen cyberpunk render implant_activate --foley-mix 0.7 --seed 42
soundgen cyberpunk pack netrunner --style electric --out output/cyberpunk
soundgen cyberpunk pack core --style electric --play  # audition all 64 in order
```

The `core` pack writes one cue per organised group directory. `netrunner`
provides variations for network and hacking events; `street-tech` covers
terminals, holograms, implants, credits, objectives, communications, and drone
alerts. All commands emit stable-v2 seeds and schema-v1 manifests.

```python
from python_sound_generator.cyberpunk import generate, write_wav

samples = generate("implant_activate", style="electric", seed=42)
write_wav("output/implant_activate.wav", samples)
```

The default provider path remains standard-library-only. `--foley-mix 0..1`
optionally adds the existing mechanical-key recording to `terminal_key` and
one of three real servo impulses to `implant_activate`. Network, hacking, EMP,
radio, hologram, credit, objective, and drone-rate cues remain procedural.

## Quirky 16-bit RPG sound library

The `quirky-rpg` provider supplies 64 original one-shots for an offbeat
early-1990s console RPG: compact menus, dialogue reactions, everyday items,
psychic abilities, battle and field feedback, character-role actions, town
services, conditions, telephones, doors, saving, party changes, and surreal
events. Its `suburban`, `psychedelic`, and `lofi` styles change the waveform
palette, pitch instability, bandwidth, bit depth, sample hold, echo, and
transient weight while keeping each gameplay event recognizable.

```sh
soundgen quirky-rpg list
soundgen quirky-rpg sources
soundgen quirky-rpg info strange_event --style psychedelic
soundgen quirky-rpg render psi_cast --style psychedelic --count 3 --seed 42
soundgen quirky-rpg render critical_hit --foley-mix 0.65 --seed 42
soundgen quirky-rpg pack hometown --style suburban --out output/quirky-rpg
soundgen quirky-rpg pack core --style suburban --play  # audition all 64 in order
```

The `core` pack writes every cue once in group directories. `hometown` provides
48 variations for repeated interface, dialogue, item, and world events;
`psychic` and `battle` each provide 15 focused variations. Render and pack
commands write stable-v2 seeds and schema-v1 manifests.

```python
from python_sound_generator.quirky_rpg import generate, write_wav

samples = generate("strange_event", style="psychedelic", seed=42)
write_wav("output/strange_event.wav", samples)
```

This library defaults to original procedural synthesis. A nonzero Foley mix
reuses approved cloth, gear, body-impact, panel, and latch recordings for five
physical cues, then filters, bit-reduces, and sample-holds every recorded layer
into the cartridge-like palette. It does not copy melodies, samples, or named
assets from any existing game.

## Ninja game sound library

The `ninja` provider supplies 30 deterministic cues across movement, blades,
projectiles, tools, and stealth-world feedback. Its `shadow`, `arcade`, and
`cinematic` styles move between restrained gameplay readability, bright compact
exaggeration, and darker weight. The `core`, `infiltration`, `arsenal`, and
`tools` packs contain 30, 33, 36, and 18 files respectively.

```sh
soundgen ninja list
soundgen ninja sources
soundgen ninja info takedown --style cinematic
soundgen ninja render katana_clash --style arcade --count 3 --seed 42
soundgen ninja pack arsenal --style shadow --out output/ninja
soundgen ninja pack core --style shadow --play  # audition all 30 in order
```

This provider is intentionally hybrid. Approved CC0 recordings supply chaotic
physical details the ear recognizes immediately: cloth, body contact, timber,
stone, light metal, panel scraping, and latches. Procedural synthesis owns air
motion, spinning rates, blade resonance, smoke and flash energy, alerts, and
the success cadence. `--foley-mix 0..1` controls recorded-layer contribution;
every manifest names the exact logical source files it consumed.

No WAV is copied into the ninja package. Its 35-source inventory resolves to
existing provider banks or content-addressed shared assets, while
[`sources/provenance.json`](python_sound_generator/generators/ninja_sound_generator/sources/provenance.json)
links each reuse to its provenance-owning provider.

```python
from python_sound_generator.ninja import generate_with_sources, write_wav

result = generate_with_sources("sliding_door", style="shadow", seed=42)
write_wav("output/sliding_door.wav", result.samples)
print(result.sources)
```

### Implemented Foley review

The ninja audit also checked the three preceding procedural domains. Its
recommendations are now implemented as opt-in layers; their musical and
abstract cues remain synthesized.

| Library | Foley-enabled cues | Implementation |
| --- | --- | --- |
| `rpg` | `item_drop`, `item_equip`, `coin_pickup`, `purchase`, `potion_drink`, `chest_open`, `door_unlock` | Real coins, cork, swallow, chest, cloth/gear, material contacts, and lock; 25-source inventory. |
| `cyberpunk` | `terminal_key`, `implant_activate` | One reused key contact plus three provider-owned servo impulses; four-source inventory. |
| `quirky-rpg` | `equip`, `player_hit`, `enemy_hit`, `critical_hit`, `door_transition` | Thirteen reused sources with mandatory cartridge-style filtering and reduction; no duplicated WAV bank. |

All three retain `--foley-mix 0` as the default, so existing procedural renders
and lightweight imports remain unchanged. Their `sources` commands expose the
complete CC0 ledgers, resolution diagnostics, and checksum verification for
every provider-owned excerpt. The ninja provider exposes the same diagnostics.

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

## Output

Provider CLIs retain their exact output filenames. Pass `--out`
where a provider supports it to choose a destination. By default, generated
files go to `output/<provider>/` beneath the current working directory, never
into the installed package. A few standalone procedural providers use a
positional output filename; their `--help` text documents it.

Render and pack manifests use schema version 1. The common envelope includes a
structured audio format, the legacy human-readable format string, and
`seed_scheme: "stable-v2"`; provider-specific provenance and options remain
extension fields. Artifact `loop` is always boolean and seam measurements live
under `seam`. Stable-v2 seeds derive from provider, cue, logical filename, and
variation, so inserting or reordering a pack entry does not change existing
sounds. The Python runtime retains `legacy_seed()` for callers that must
reconstruct the former positional convention.

### External music toolchain

The `music` provider drives Strudel as an external process; Strudel is not linked
or bundled. Set `STRUDEL_DIR` to a prepared Strudel checkout with Vite,
Puppeteer Core, and `@strudel/web`, and optionally set `CHROME_BIN` and
`STRUDEL_PORT`. Node, Chrome/Chromium, and FFmpeg must be available. Read-only
commands such as `music list`, `music info`, and cycle-math verification work
without that toolchain.

## License

Project-authored code, documentation, procedural synthesis definitions, and
music patterns are available under the [MIT License](LICENSE). Bundled
recordings retain their recorded CC0/public-domain terms; see
[ASSET_LICENSES.md](ASSET_LICENSES.md) and each provider's provenance ledger.

## Verification

```sh
make verify
make build
python -m twine check dist/*
```

`verify` runs Ruff, the complete test suite, and the 32-provider health check.
CI repeats those checks on Python 3.11 through 3.14 and validates both built
distributions. The unified tests pin registry uniqueness, namespace-safe lazy loading,
compatibility aliases, DSP PCM characterization hashes, stable seeds, manifest
schema, shared-buffer output, content-addressed asset integrity, all-provider
imports and help commands, packaged-data integrity, and end-to-end WAV renders
through the facade—including a default-output render from the household
appliance Foley provider.
