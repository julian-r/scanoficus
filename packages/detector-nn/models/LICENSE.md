# DocQuadNet weights — license

Files: `docquadnet.json` (graph, with the name, length and SHA-256 of its weights and a
self-test reference) and `docquadnet-<sha256 prefix>.bin` (fp16 weights). The weights and the
graph nodes are unchanged from `export/docquadnet/` of the scanoficus-ml repository (run
`e06-final`); `scripts/package_model.py` adds the `weights` and `selfTest` records. They are
used by `@scanoficus/detector-nn`.

The weights were trained by us from scratch (no pretrained weights) and are released under the
BSD 3-Clause License (`../LICENSE`), Copyright (c) 2026, Julian Rath.

The training data includes frames of the SmartDoc 2015 dataset, whose license requires
attribution. This notice must be kept with the weights:

> SmartDoc 2015 Ch. 1 **tuning split only** (datasheet, magazine, patent documents 001–004 on
> all 5 backgrounds; every 3rd frame = 3,392 frames, downscaled to 640×360) — CC BY 4.0 —
> © Chazalon et al., ICDAR 2015 SmartDoc Challenge 1
> (github.com/jchazalon/smartdoc15-ch1-dataset); attribution must be kept with the weights.
> License: https://creativecommons.org/licenses/by/4.0/

The other training inputs need no attribution: procedural pages, CC0 textures and HDRIs from
Poly Haven, and OFL fonts that were only rendered into images. Hashes of all training inputs
are in `data/training-manifest.json` and `data/assets-manifest.json` of the scanoficus-ml
repository.

SHA-256 of the weights: `9bc8c22c6a4cdf9a955c18e9a0f7c44e75843df7deacc3fae06224b465c33185` (also in `docquadnet.json`).
