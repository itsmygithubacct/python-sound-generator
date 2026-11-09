# Python Sound Generator

Python Sound Generator is one command line, `soundgen`, that renders game sound
effects for 32 audio domains from deterministic procedural synthesis and
approved CC0/public-domain Foley recordings.

It is not part of the Plebian-OS install or its content catalog. The Kilix for
Windows companion lists it in its application catalog
(`kilix apps install python-sound-generator`, alias `soundgen`).
[pleb-tower](https://github.com/itsmygithubacct/pleb-tower#readme) and
[kilix-land-desktop](https://github.com/itsmygithubacct/kilix-land-desktop#readme)
render their sound effects through it when they regenerate their sound banks
(`make audio`); their normal builds do not run it.

## What it does

- Renders fire, water, wood, stone, weapons, projectiles, gunshots, vehicles,
  appliances, creatures, voices, ambience, UI and more: 32 providers behind one
  lazy registry.
- Provides four themed cue libraries (RPG, cyberpunk, quirky 16-bit RPG and
  ninja) with styles, variation packs, and an opt-in Foley mix.
- Writes WAV files with a schema-v1 manifest of options, stable seeds, levels,
  SHA-256 checksums and logical source names.
- Derives seeds from provider, cue, filename and variation, so adding a pack
  entry never changes existing sounds.
- Renders packaged recipe cue banks that delegate to the real provider CLIs.
- Checks its own health (`soundgen doctor`) and reports catalogs as JSON.
- Imports only the provider you ask for: listing the catalog loads no NumPy,
  SciPy or WAV data.

## Quick start

Python 3.11 or newer is required.

```sh
git clone https://github.com/itsmygithubacct/python-sound-generator.git
cd python-sound-generator
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'

soundgen list                                      # the 32 providers
soundgen fire render lava_flow --seconds 8 --out output/lava
soundgen rpg pack core --theme fantasy --play      # audition all 65 RPG cues
```

With the dependencies already installed, `python3 -m python_sound_generator list`
works from a source checkout without installing.

## Usage

| Command | What it does |
|---|---|
| `soundgen list [--json]` | every provider with a one-line description |
| `soundgen info PROVIDER [--json]` | one provider's catalog details |
| `soundgen doctor [--json]` | checks every entry point, runtime file, JSON asset and WAV container |
| `soundgen PROVIDER ...` | runs that provider's own CLI with the remaining arguments unchanged |
| `soundgen PROVIDER --help` | that provider's exact options |
| `soundgen recipe validate`, `soundgen recipe render NAME` | validate or render the packaged recipe cue banks |
| `soundgen car-engine sample`, `inspect`, `render-bank` | the car provider's recording bank, RPM/load anchors and external banks |

The themed libraries share the verbs `list`, `sources`, `info CUE`,
`render CUE` and `pack NAME`, with options such as `--seed`, `--count`,
`--out`, `--play`, a style or theme, and `--foley-mix 0..1`; other providers
keep their own interfaces. Output goes to `output/<provider>/` under the current
directory unless `--out` says otherwise (a few providers take a positional
output filename).

From Python:

```python
from python_sound_generator import get_provider, providers, run

status = run("fire", ["render", "lava_flow", "--seconds", "8"])
```

## How it works

```
soundgen CLI ──▶ lazy provider registry (python_sound_generator/providers.py)
                   └─ imports only the selected provider from generators/
                         └─ shared python_sound_generator._audio layer:
                              standard-library DSP, loops and rotors,
                              optional NumPy/SciPy Foley transforms,
                              PCM writer, manifests, stable-v2 seeds,
                              content-addressed WAV assets
```

Each provider keeps its own registry, renderer, mastering and source-selection
policy, and keeps its own command-line axes instead of a forced common schema.
Duplicate recordings are stored once by SHA-256 under
`python_sound_generator/assets/audio/`, and `assets/catalog.json` maps every
logical source name back to its provider and provenance ledger. The default
`--foley-mix 0` path is procedural and standard-library-only; the external
Strudel music recorder is the one exception to byte-for-byte deterministic
output.

Adding a domain takes a runtime package with approved source names plus one
lazy `Provider` entry. Details: [docs/architecture.md](docs/architecture.md).

## Documentation

- [docs/architecture.md](docs/architecture.md): the provider registry, the
  shared `_audio` layer, adding a domain, and what the distribution includes.
- [docs/cli.md](docs/cli.md): the unified CLI, recipes, JSON output, the Python
  facade, output and manifests, and the Strudel music toolchain.
- [docs/libraries.md](docs/libraries.md): the RPG, cyberpunk, quirky 16-bit RPG
  and ninja libraries, and the Foley review.
- [docs/development.md](docs/development.md): installation, `make verify`,
  reproducible builds and CI.
- [ASSET_LICENSES.md](ASSET_LICENSES.md): licensing and provenance of bundled
  recordings and rendered output.
- [foley_intake/README.md](foley_intake/README.md): the Foley intake masters
  used to reproduce packaged excerpts.

## Related repositories

- [pleb-tower](https://github.com/itsmygithubacct/pleb-tower#readme): its
  `tools/generate_audio.py` renders the game's sound effects through these
  providers; the recorded provenance is in its `assets/audio/provenance.json`.
- [kilix-land-desktop](https://github.com/itsmygithubacct/kilix-land-desktop#readme):
  its `tools/generate_audio.py` renders the Land desktop's sound effects through
  these providers.
- [kilix](https://github.com/itsmygithubacct/kilix#readme): the Kilix for
  Windows companion's application catalog pins and installs it.

## Part of Plebian-OS

Python Sound Generator is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
It is not installed by Plebian-OS; Pleb Tower and the Land desktop regenerate
their sound banks with it, and it is an installable app in the Kilix for Windows
companion. The map of every
repository and how it ships is on the
[components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## License

Project-authored code, documentation, procedural synthesis definitions, and
music patterns are available under the [MIT License](LICENSE). Bundled
recordings retain their recorded CC0/public-domain terms; see
[ASSET_LICENSES.md](ASSET_LICENSES.md) and each provider's provenance ledger.
