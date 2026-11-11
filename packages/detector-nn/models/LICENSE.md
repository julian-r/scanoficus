# DocQuadNet weights — license

Bundles (one folder each; `scripts/package_model.py` adds the `weights` and `selfTest` records
to the graphs and writes `bundle.json`; the weights and graph nodes are unchanged from the
scanoficus-ml exports):

- `docquadnet/`: round 1, `docquadnet.json` and `docquadnet-<sha256 prefix>.bin`, from
  `export/docquadnet/` (run `e06-final`);
- `docquadnet2/`: round 2, the main network `docquadnet2.*` (`export/docquadnet2/`, run
  `r2-m2-ft`) and the corner refiner `docquadnet2-refiner.*` (`export/docquadnet2-refiner/`, run
  `r2-rf1`);
- `docquadnet3/`: round 3, the round-2 main network (the files in `docquadnet2/`), the corner
  refiner with corner scores `docquadnet3-refiner.*` (`export/docquadnet3-refiner/`, run
  `r3-rf3-conf`, fine-tuned from `r2-rf1`), `decision.json` (`export/docquadnet3/`, unchanged)
  and `coco-credits.csv` (the COCO images below).

They are used by `@scanoficus/detector-nn`.

The weights were trained by us from scratch (no pretrained weights) and are released under the
BSD 3-Clause License (`../LICENSE`), Copyright (c) 2026, Julian Rath.

The training data of all bundles includes frames of the SmartDoc 2015 dataset, whose license
requires attribution. This notice must be kept with the weights:

> SmartDoc 2015 Ch. 1 **tuning split only** (datasheet, magazine, patent documents 001–004 on
> all 5 backgrounds; every 3rd frame = 3,392 frames, downscaled to 640×360) — CC BY 4.0 —
> © Chazalon et al., ICDAR 2015 SmartDoc Challenge 1
> (github.com/jchazalon/smartdoc15-ch1-dataset); attribution must be kept with the weights.
> License: https://creativecommons.org/licenses/by/4.0/

Rounds 2 and 3 (`docquadnet2/`, `docquadnet3/`) were also trained on the training part of the
Dizaj photos (CC BY 4.0). This notice must be kept with those weights:

> "A New Image Dataset for Document Corner Localization" by Shima Dizaj, Mendeley Data v3,
> https://doi.org/10.17632/x3nm4cxr83.3, licensed under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/). Changes: photos resized, cropped, rotated and
> augmented for training; labels converted. Used: the training part (524 photos).
>
> Note (Mendeley Data): the dataset's license does not cover content shown in the photos, which
> can belong to third parties.

Round 3 (`docquadnet3/`: its refiner and decision) also used COCO photos. This notice and
`docquadnet3/coco-credits.csv` must be kept with those files:

> Images from COCO train2017 (Lin et al., "Microsoft COCO: Common Objects in Context", ECCV
> 2014; https://cocodataset.org; annotations CC BY 4.0, used to select the images), originally
> from Flickr under CC BY 2.0 (https://creativecommons.org/licenses/by/2.0/) or "no known
> copyright restrictions" (https://flickr.com/commons/usage/): 6,329 photos without documents;
> 3,921 trained the round-3 refiner (corner crops), 1,637 fitted the presence decision and 771
> were used for validation. Changes: cropped (corner crops), resized, rotated and augmented. The
> list with each image's COCO id, role, license and Flickr source URL is
> `docquadnet3/coco-credits.csv`. The Flickr photographers are the authors; COCO does not
> record their names.

The other training inputs need no attribution: procedural pages, CC0 textures and HDRIs from
Poly Haven, and OFL fonts that were only rendered into images. Hashes of all training inputs
are in `data/training-manifest.json` and `data/assets-manifest.json` of the scanoficus-ml
repository.

The SHA-256 of each weights file is in its graph JSON (`weights.sha256`).
