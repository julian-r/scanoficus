# DocQuadNet weights — license

Bundles in this package (one folder each; `scripts/package_model.py` adds the `weights` and
`selfTest` records to the graphs and writes `bundle.json`; the weights and graph nodes are
unchanged from the scanoficus-ml exports):

- `docquadnet/`: round 1, `docquadnet.json` and `docquadnet-<sha256 prefix>.bin`, from
  `export/docquadnet/` (run `e06-final`);
- `docquadnet4/`: round 4, the main network `docquadnet4.*` (`export/docquadnet4/`, run `r4-m6`,
  fine-tuned from round 2's `r2-m2-ft`), the corner refiner with corner scores
  `docquadnet4-refiner.*` (`export/docquadnet4-refiner/`, run `r4-rf6`, fine-tuned from round
  3's `r3-rf3-conf`), `decision.json` (`export/docquadnet4/`, plus the weights' hashes) and
  `coco-credits.csv` (the COCO images below).

Rounds 2 and 3 are not in the npm package; the demo keeps them (`apps/demo/models/`, with their
own `LICENSE.md`).

The weights were trained by us from scratch (no pretrained weights; later rounds fine-tune our
earlier ones) and are released under the BSD 3-Clause License (`../LICENSE`), Copyright (c)
2026, Julian Rath.

The training data of both bundles includes frames of the SmartDoc 2015 dataset, whose license
requires attribution. This notice must be kept with the weights:

> SmartDoc 2015 Ch. 1 **tuning split only** (datasheet, magazine, patent documents 001–004 on
> all 5 backgrounds; every 3rd frame = 3,392 frames, downscaled to 640×360) — CC BY 4.0 —
> © Chazalon et al., ICDAR 2015 SmartDoc Challenge 1
> (github.com/jchazalon/smartdoc15-ch1-dataset); attribution must be kept with the weights.
> License: https://creativecommons.org/licenses/by/4.0/

Round 4 (`docquadnet4/`; its main network and refiner continue round 2 and 3 training) was also
trained on the training part of the Dizaj photos (CC BY 4.0). This notice must be kept with
those weights:

> "A New Image Dataset for Document Corner Localization" by Shima Dizaj, Mendeley Data v3,
> https://doi.org/10.17632/x3nm4cxr83.3, licensed under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/). Changes: photos resized, cropped, rotated and
> augmented for training; labels converted. Used: the training part (524 photos).
>
> Note (Mendeley Data): the dataset's license does not cover content shown in the photos, which
> can belong to third parties.

Round 4 (`docquadnet4/`: its refiner and decision) also used COCO photos (the same images and
roles as round 3; round 4 added only synthetic scenes from the sources below). This notice and
`docquadnet4/coco-credits.csv` must be kept with those files:

> Images from COCO train2017 (Lin et al., "Microsoft COCO: Common Objects in Context", ECCV
> 2014; https://cocodataset.org; annotations CC BY 4.0, used to select the images), originally
> from Flickr under CC BY 2.0 (https://creativecommons.org/licenses/by/2.0/) or "no known
> copyright restrictions" (https://flickr.com/commons/usage/): 6,329 photos without documents;
> 3,921 trained the refiner (corner crops; round 3, and round 4's refiner fine-tuned from it),
> 1,637 fitted the presence decision and 771 were used for validation. Changes: cropped (corner
> crops), resized, rotated and augmented. `docquadnet4/coco-credits.csv` lists each image's COCO id,
> role, license, Flickr photo page (https://flic.kr/p/…) and, for the 6,223 CC BY 2.0 photos,
> the author and title as Flickr shows them today (oEmbed, fetched November 2026): 5,150 found,
> 1,073 no longer on Flickr (author and title unknown; the COCO source URL is listed). 106
> photos with "no known copyright restrictions" need no author. Some owners have changed the
> license on Flickr since COCO collected the photos (column `flickr_license_now`); the license
> under which COCO obtained them (column `license`) applies to the copies used here.

The other training inputs need no attribution: procedural pages, CC0 textures and HDRIs from
Poly Haven, and OFL fonts that were only rendered into images. Hashes of all training inputs
are in `data/training-manifest.json` and `data/assets-manifest.json` of the scanoficus-ml
repository.

The SHA-256 of each weights file is in its graph JSON (`weights.sha256`).
