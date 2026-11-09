# Game sound libraries

The four themed cue libraries (RPG, cyberpunk, quirky 16-bit RPG and ninja) and the Foley review that covers them, moved here from the README.

## RPG sound library

The `rpg` provider adds 65 deterministic one-shots for a complete party-RPG
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
soundgen rpg pack core --theme fantasy --play  # audition all 65 cues in order
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
mix adds approved physical layers to fourteen cues, including modelled weapon
movement and recorded body/creature impacts, while progression, dialogue,
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
optionally adds specialist firearm reports and mechanisms, modelled
close-combat movement and ricochet, and approved key, servo, body, metal, and
stone contacts to thirteen physical cues. Network, hacking, EMP, radio,
hologram, credit, objective, and drone-rate cues remain procedural.

## Quirky 16-bit RPG sound library

The `quirky-rpg` provider supplies 65 original one-shots for an offbeat
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
soundgen quirky-rpg pack core --style suburban --play  # audition all 65 in order
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
adds modelled blunt-weapon movement and reuses approved cloth, body-impact,
wood-contact, creature, panel, and latch recordings for ten physical cues,
then filters, bit-reduces, and sample-holds each borrowed layer into the
cartridge-like palette. It does not copy melodies, samples, or named assets
from any existing game.

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
[`sources/provenance.json`](../python_sound_generator/generators/ninja_sound_generator/sources/provenance.json)
links each reuse to its provenance-owning provider.

```python
from python_sound_generator.ninja import generate_with_sources, write_wav

result = generate_with_sources("sliding_door", style="shadow", seed=42)
write_wav("output/sliding_door.wav", result.samples)
print(result.sources)
```

### Implemented Foley review

The hybrid-domain audit covers the four libraries below. Their physical
events have opt-in modelled or recording-backed layers while musical and
abstract identities remain synthesized.

| Library | Foley-enabled cues | Implementation |
| --- | --- | --- |
| `rpg` | Fourteen item, equipment, battle, and mechanism cues | Modelled sword movement plus approved coins, cork, swallow, chest, cloth/gear, material, body, creature, and lock contacts; 39-source inventory. |
| `cyberpunk` | Thirteen hardware and combat cues | Specialist firearm, combat-motion, and ricochet models plus approved key, servo, body, metal, and stone contacts; 42-source inventory. |
| `quirky-rpg` | Ten equipment, battle, and world cues | Modelled blunt-weapon movement plus 24 reused cloth, body, wood, creature, panel, and latch sources with cartridge treatment; no duplicated WAV bank. |
| `computer` | `mouse_click` | Separate CC0 mouse-button press and release contacts assembled with their physical timing; two provider-owned sources. |

All three retain `--foley-mix 0` as the default, so existing procedural renders
and lightweight imports remain unchanged. Their `sources` commands expose the
complete CC0 ledgers, resolution diagnostics, and checksum verification for
every provider-owned excerpt. The ninja provider exposes the same diagnostics.
