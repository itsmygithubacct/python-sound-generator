# The soundgen command line

The unified CLI, the Python facade, output locations, manifests and the external music toolchain, moved here from the README. Each provider's own options are listed by `soundgen PROVIDER --help`.

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
