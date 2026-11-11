# DocQuadNet weights — license

Bundles (one folder each; `scripts/package_model.py` adds the `weights` and `selfTest` records
to the graphs and writes `bundle.json`; the weights and graph nodes are unchanged from the
scanoficus-ml exports):

- `docquadnet/`: round 1, `docquadnet.json` and `docquadnet-<sha256 prefix>.bin`, from
  `export/docquadnet/` (run `e06-final`);
- `docquadnet2/`: round 2, the main network `docquadnet2.*` (`export/docquadnet2/`, run
  `r2-m2-ft`) and the corner refiner `docquadnet2-refiner.*` (`export/docquadnet2-refiner/`, run
  `r2-rf1`).

They are used by `@scanoficus/detector-nn`.

The weights were trained by us from scratch (no pretrained weights) and are released under the
BSD 3-Clause License (`../LICENSE`), Copyright (c) 2026, Julian Rath.

The training data of both bundles includes frames of the SmartDoc 2015 dataset, whose license
requires attribution. This notice must be kept with the weights:

> SmartDoc 2015 Ch. 1 **tuning split only** (datasheet, magazine, patent documents 001–004 on
> all 5 backgrounds; every 3rd frame = 3,392 frames, downscaled to 640×360) — CC BY 4.0 —
> © Chazalon et al., ICDAR 2015 SmartDoc Challenge 1
> (github.com/jchazalon/smartdoc15-ch1-dataset); attribution must be kept with the weights.
> License: https://creativecommons.org/licenses/by/4.0/

Round 2 (`docquadnet2/`) was also trained on the training part of the Dizaj photos (CC BY 4.0).
This notice must be kept with those weights:

> Dizaj document photos, Mendeley Data, version 3 (DOI 10.17632/x3nm4cxr83.3), training part
> (524 photos) — CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

The other training inputs need no attribution: procedural pages, CC0 textures and HDRIs from
Poly Haven, and OFL fonts that were only rendered into images. Hashes of all training inputs
are in `data/training-manifest.json` and `data/assets-manifest.json` of the scanoficus-ml
repository.

The SHA-256 of each weights file is in its graph JSON (`weights.sha256`).
