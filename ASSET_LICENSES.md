# Asset licensing and provenance

The MIT License in [`LICENSE`](LICENSE) covers this project's source code,
documentation, original procedural synthesis definitions, and original music
patterns.

Bundled third-party recordings are separate works. They were selected under a
CC0 or public-domain-only policy and retain the terms recorded in their
provider-owned `sources/provenance.json` files. The shared
`python_sound_generator/assets/catalog.json` maps deduplicated WAV blobs back
to those provenance ledgers. The MIT License does not claim ownership of those
recordings or replace their recorded terms.

Audio rendered solely from this project's synthesis code and original patterns
is dedicated by the project author to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode), to the
extent the author holds copyright or related rights in that output. Hybrid
renders may also contain bundled CC0/public-domain recordings; their exact
inputs remain listed in each render manifest.

The five full-length Foley masters described by `foley_intake/manifest.json`
are reproducibility inputs, not distributed assets. They are deliberately
ignored under `foley_intake/downloads/`; only selected, checksum-pinned runtime
excerpts are packaged.

## Music toolchain

The music patterns are original compositions by the project author and are
licensed under MIT with the rest of the source. Three patterns were first
composed on 2026-07-07 for an earlier MIT-licensed game project; their exact
text hashes are retained in the music provider's provenance ledger.

The optional music recorder invokes [Strudel](https://strudel.cc/) from a
separately installed checkout. Strudel is licensed AGPL-3.0-or-later, but no
Strudel source code or sample bank is copied or bundled here. The packaged
recorder consists of project-authored integration code that calls Strudel's
public API, and all included patterns use synthesized oscillators and noise
rather than third-party samples.

This file is a practical inventory, not legal advice. The per-provider
provenance ledgers remain authoritative for individual audio files.
