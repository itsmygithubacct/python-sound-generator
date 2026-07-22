# Foley master intake

This staging area records the five original CC0 WAV masters selected and
integrated on 2026-07-22 after reviewing the RPG, cyberpunk, and quirky-RPG
libraries. The public Freesound previews were used only for research; the
runtime excerpts were derived from the verified lossless downloads.

## Reproducing the intake

Log in to Freesound in your browser, open each page, and choose **Download**:

1. [565799 — Coins by dvdfu](https://freesound.org/people/dvdfu/sounds/565799/)
2. [434143 — Cork pop.wav by mfearnley](https://freesound.org/people/mfearnley/sounds/434143/)
3. [573654 — Wooden Chest Open.wav by The_Frisbee_of_Peace](https://freesound.org/people/The_Frisbee_of_Peace/sounds/573654/)
4. [411219 — Drink and gulp by AndersHolm](https://freesound.org/people/AndersHolm/sounds/411219/)
5. [824121 — ServoMotor.wav by derjuli](https://freesound.org/people/derjuli/sounds/824121/)

Put the unchanged `.wav` files in
`foley_intake/downloads/`. Freesound normally prefixes downloaded names with
the sound ID; please retain that ID if your browser changes a filename. Do not
send account credentials or cookies.

The completed integration:

- verified RIFF/WAVE metadata and SHA-256 checksums for all five masters;
- selected and bundled 15 lossless-derived runtime excerpts;
- recorded exact edit regions and hashes in provider provenance ledgers;
- added opt-in Foley while retaining the procedural-only default path;
- reused existing approved recordings where no new master was needed; and
- passed the complete tests, provider doctor, package-content checks, and
  installed-wheel smoke renders.

The selected-source metadata and final cue mappings are recorded in
[`manifest.json`](manifest.json).
