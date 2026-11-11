# Detector benchmarks on public datasets

This page measures the classic detector (`detectDocument` with default options) on real,
publicly available video frames and on photos without documents. It covers the detector before
tuning (v1, commit `5d6b0a3`) and after tuning on a fixed tuning split (v2). The baseline
sections further down describe v1 and were the starting point of the tuning.

## Summary

Default options, long side 384, SmartDoc Jaccard index (JI). Held-out data was not used for
any decision (see the [protocol](#method)):

| data (frames)                                          | v1 (before) | v2 (after) | v2, best candidate without threshold |
| ------------------------------------------------------ | ----------: | ---------: | -----------------------------------: |
| **SmartDoc held-out documents (12,267)**               |   **0.652** |  **0.786** |                                0.877 |
| **SmartDoc held-out documents, backgrounds 1–4**       |   **0.728** |  **0.877** |                                0.968 |
| SmartDoc tuning documents (12,622)                     |       0.620 |      0.758 |                                0.869 |
| SmartDoc all (24,889)                                  |       0.636 |      0.772 |                                0.873 |
| **MIDV-500 subset, held out (3,000)**                  |   **0.789** |  **0.622** |                                0.786 |
| **MIDV-2019 subset, held out (1,200)**                 |   **0.690** |  **0.587** |                                0.755 |
| **Photos without a document (785): share with a quad** |  **20.4 %** |  **6.5 %** |                                    – |

- On held-out SmartDoc documents, v2 raises JI from 0.652 to 0.786 (backgrounds 1–4: 0.728
  to 0.877) and returns a quad on 6.5 % instead of 20.4 % of photos of desks, tables and
  screens without a document. Without a score threshold, its best candidate reaches 0.968 on
  held-out backgrounds 1–4 (backgrounds 1–3: 0.977–0.990), close to the best classic methods
  (0.97–0.99). Background 4 (white on white, glare) is at 0.557 (0.897 without threshold),
  and background 5 (clutter) stays at 0.
- **MIDV regressed** (0.789 → 0.622 and 0.690 → 0.587). The calibrated threshold rejects many
  ID cards that are held in the hand or partly outside the frame, because their quads score
  lower (frame-border sides, fingers over an edge). The precision of the returned MIDV quads
  rose (IoU ≥ 0.9 among returned: 82 % → 91 % on MIDV-500), the recall fell. Without a
  threshold, MIDV-500 dropped from 0.820 to 0.786 and MIDV-2019 rose from 0.747 to 0.755. A
  diagnostic ablation (on held-out data, not used for any decision) attributes the MIDV-500
  ranking loss to the weaker size preference (see the [experiment log](#experiment-log)). The
  tuning split holds only SmartDoc, whose documents cover about 10 % of the frame. It needs
  hand-held, larger and partly visible documents before the size preference and the threshold
  can be tuned without trading MIDV for SmartDoc.
- The score is calibrated: `DEFAULT_MIN_SCORE` (0.75) returns a quad on 6.5 % of the negatives,
  and 97–98 % of the returned SmartDoc quads are right (MIDV: 88 %). At `CONFIDENT_SCORE`
  (0.85) the negatives rate is 1.8 %, and 99.9–100 % of the SmartDoc quads are right (MIDV:
  95.5 %). See
  [Score semantics](#score-semantics-and-hybrid-escalation).
- Detector time per frame at 384: 5.0 ms (v1: 4.6 ms in the same session), single thread,
  Node 26 on an Apple M5 Pro. `@scanoficus/core` grew from 22.5 to 23.0 KB min+gz.

## Datasets

Datasets live outside the repository, in `$SCANOFICUS_DATASETS` (default
`~/datasets/scanoficus/<dataset>/`). They are never committed. `pnpm bench:fetch` downloads,
verifies and unpacks them. Downloads resume after an interruption, and finished steps are
skipped on re-runs. The script needs `curl`, `tar`, `unzip`, `shasum` and `openssl`.

```sh
pnpm bench:fetch                                  # all (≈ 1 GB + 2 × ≈ 7.5 GB + 1.2 GB download)
pnpm bench:fetch --datasets smartdoc              # SmartDoc only
pnpm bench:fetch --datasets negatives             # no-document negatives (COCO subset)
pnpm bench:fetch --datasets midv500 --midv-types 01,02,03 --keep-archives
```

| dataset                                                                                          | content used                                                                                                                                          | license / terms                                                                                                                                                                               | access                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [SmartDoc 2015 Challenge 1](https://github.com/jchazalon/smartdoc15-ch1-dataset), release v2.0.0 | All 150 clips (5 backgrounds × 30 documents of 6 types), 24,889 frames of 1920×1080, GT quad per frame                                                | CC BY 4.0 (`LICENCE` in the archive). Cite Burie et al., ICDAR 2015. "Original images remain the property of their respective authors."                                                       | `frames.tar.gz` (1.0 GB) from the GitHub release, verified with the published SHA-256. No registration.                                                                          |
| [MIDV-500](https://arxiv.org/abs/1807.05786)                                                     | 10 of 50 ID document types (every 5th: 01, 06, 11, …, 46). All 10 clips per type (5 conditions × 2 phones), 30 frames each: 3,000 frames of 1080×1920 | `license.txt` on the FTP server: CC BY-SA 2.5. Source images are public-domain or openly licensed Wikimedia images (see `documents.pdf`). Cite Arlazarov et al., Computer Optics 43(5), 2019. | `ftp://smartengines.com/midv-500/dataset/<type>.zip` (≈ 0.65 GB each), verified with `md5.txt`. Anonymous FTP, no registration. Only `images/` and `ground_truth/` are unpacked. |
| [MIDV-2019](https://arxiv.org/abs/1910.04009)                                                    | The same 10 types. All 4 clips per type (distorted / low light × 2 phones), 30 frames each: 1,200 frames of 2160×3840                                 | `license.txt` on the FTP server: CC BY-SA 2.5. Cite Bulatov et al., ICMV 2019 (Proc. SPIE 11433).                                                                                             | `ftp://smartengines.com/midv-500/extra/midv-2019/dataset/<type>.zip` (≈ 0.75 GB each), verified with `md5.txt`. No registration.                                                 |

| No-document negatives (COCO 2017 subset, see `packages/core/bench/negatives.ts`) | 785 photos (of 800 selected: 230 val2017, 570 train2017; mostly 640×480) of tables, desks, rooms, screens, laptops, keyboards, phones and remotes without any annotated book, paper, cardboard or banner and without a caption that mentions printed matter. 15 images with a frame added in photo editing (uniform 3-px rim on all sides) are skipped | Annotations CC BY 4.0 (COCO). Images: Flickr, only those under CC BY 2.0, CC BY-SA 2.0, CC BY-ND 2.0 or "no known copyright restrictions" (all permit evaluation use). Used locally only, never redistributed. | Panoptic + caption annotation zips (0.86 + 0.25 GB, only for the selection), then the 800 images (119 MB) from `images.cocodataset.org`. `manifest.json` lists ids, licenses and Flickr URLs. |

No dataset needed a registration. MIDV-2020 (a 124 GB university mirror behind a form and
license acceptance) was not used. The MIDV subset is fixed (every 5th document type) rather
than random, so it is reproducible. It covers passports, ID cards and driving licences of 10
countries.

## Method

**Pipeline.** For every clip, one ffmpeg process decodes each frame at full resolution to 8-bit
gray (`-pix_fmt gray`, the luma of the decoded image, which is within ±1 of the library's
`rgbaToGray`). The library's own `downscaleGray` (area averaging) then reduces the frame to
long side 256, 384 (the default) and 512, and `detectDocument(small, { maxSide, candidates:
true }, context)` runs with default options. Detections are mapped back to frame coordinates.
In the live pipeline the browser downscales with `drawImage` before the worker gets the
frame, so only the `detectDocument` call is timed. `candidates: true` only records the
ranked runner-ups (at most 5) for the analysis below and does not change the result.

**Tracked variant.** The library's `QuadTracker` (default options) runs over the per-frame
detections of each clip in order, and its smoothed or held quad is evaluated. Frame timestamps
are nominal: 50 ms per SmartDoc frame (the clips run at about 17–20 fps with a variable frame
rate) and 100 ms per MIDV frame (MIDV frames were extracted at 10 fps).

**Metrics.**

- **SmartDoc JI.** This is the official ICDAR 2015 SmartDoc Challenge 1 measure (Burie et al.
  2015; reference code `eval_seg.py` in
  [smartdoc15-ch1-eval](https://github.com/jchazalon/smartdoc15-ch1-eval)). The homography
  that maps the ground-truth quad onto the document's reference rectangle (the model size,
  e.g. 2100×2970 for A4 at 0.1 mm per unit) is applied to the detected quad. The score is the
  Jaccard index (IoU) of the projected detection and the reference rectangle. Frames without
  a detection, or whose projected detection self-intersects, score 0. The dataset score is the
  mean over all frames. Because the IoU is computed in the document plane, it is independent
  of the document's size in the image. For MIDV, the reference rectangle is the pixel size of
  the printed template (`images/<type>.tif`). This is the variant used by Tropin et al. for
  MIDV-500. The implementation (`packages/core/bench/metrics.ts`) matches the official
  algorithm (OpenCV homography plus polygon clipping) to within 4·10⁻⁷ on 2,793 real
  detection/GT pairs. Its unit tests use reference values from the official algorithm.
- **IoU.** Image-space IoU of the parts of the GT and the detected quad that lie inside the
  frame (MIDV corners may lie outside). A frame without a detection has IoU 0, so
  **IoU ≥ 0.9 / ≥ 0.95** are success rates over all frames.
- **Corner error.** Mean distance between corresponding corners (best cyclic relabeling) as a
  percentage of the image diagonal, with its 90th percentile, over frames with a detection.
- **found.** Share of frames with a detection.
- **ms.** Time of the `detectDocument` call on the downscaled frame. The timing numbers on this
  page come from a separate single-threaded run (`--workers 1 --step 10`). Timings from the
  parallel runs are inflated by up to 1.5× by the 16 concurrent workers.

**Diagnostic run.** A second run (`--min-area 0.02 --min-score 0`, long side 384) reports the
best candidate for every frame, including small and low-scoring ones. Failures of the
default run can then be split into causes (see [Failure analysis](#failure-analysis)). This
run is for diagnosis only. The benchmark numbers use the defaults.

**Runtime of the benchmark.** The whole SmartDoc set (24,889 frames × 3 sizes) takes 70 s with
16 worker threads. MIDV-500 takes 16 s and MIDV-2019 8 s. No frame sampling was needed, so
every frame is evaluated.

**Evaluation protocol (tuning / held-out split).** Detector changes must not be fitted to the
frames they are judged on, so the real data is split (`packages/core/bench/splits.ts`):

- SmartDoc is split by document type. Each of the 6 types has 5 documents, each filmed on all 5
  backgrounds. The tuning split holds datasheet, magazine and patent (15 documents, 75 clips,
  12,622 frames). The held-out split holds letter, paper and tax (15 documents, 75 clips,
  12,267 frames). No document is in both, and both contain every background.
- MIDV-500 and MIDV-2019 are held out entirely (other documents, devices and conditions).
- The no-document negatives are a separate false-positive check. They are used to calibrate
  score thresholds, never to rank candidates on positives.

Detector changes are developed on the tuning split only: failure montages (`--split tune`),
diagnostic runs and parameter choices. Held-out numbers are reported, not optimized. Every
change is logged in the [experiment log](#experiment-log) with its tuning result before the
held-out result is looked at.

**False-positive rate.** On the negatives, any returned quad is a false positive. The report
gives the share of images with a quad, and with a quad of score ≥ 0.6 / 0.7 / 0.8.

### Running it

```sh
pnpm bench                       # all datasets, sizes 256,384,512, default options → eval-out/bench/full/
pnpm bench --datasets smartdoc --sizes 384 --min-area 0.02 --min-score 0 --run relaxed
pnpm bench --datasets smartdoc --workers 1 --step 10 --run timing
pnpm bench --split tune --sizes 384 --run my-variant --options '{"minArea":0.04}'  # tuning split only
pnpm bench:report --diagnostic relaxed    # markdown tables → packages/core/eval-out/bench/full/report.md
pnpm bench:report --runs full,my-variant,nn-model@256   # side by side → eval-out/bench/compare.md
pnpm bench:montage --dataset smartdoc --mode worst --split tune  # or --mode failures; PNG pages + legend
```

`--detector <name>` selects a detector registered in `packages/core/bench/detectors.ts`
(default `classic`, the library's `detectDocument`), and `--options` passes it JSON options.
Each run directory holds one approach. Other detectors (e.g. a neural network evaluated
outside this repository) can write their predictions into their own run directory in the same
per-clip JSON format (`packages/core/bench/results.ts`), and `bench:report --runs` scores them
in the same way.

Results are stored per clip (`eval-out/bench/<run>/<dataset>/<clip>.json`). An interrupted
run resumes and skips finished clips. Delete a run directory to recompute it. Montages
show the ground truth in green, the detection in red and the best runner-up candidate in thin
blue.

## Score semantics and hybrid escalation

`Detection.score` is in [0, 1]. It is the share of each side's length that is supported by
edge pixels along the side (0.6 × length-weighted mean + 0.4 × weakest side), times 0.85 per
side on the frame border, times `(area / 10 %)^0.25` for quads below 10 % of the frame, times
`(inside / surrounding band)^0.5` if the inside is darker than the band around it. Two
exported constants are calibrated at long side 384 on the tuning split and the negatives
(the held-out columns confirm them):

| `minScore`                 | negatives with a quad | SmartDoc tuning: right among returned / recall | SmartDoc held-out: right / recall | MIDV (both): right / recall |
| -------------------------- | --------------------: | ---------------------------------------------: | --------------------------------: | --------------------------: |
| 0.5                        |                49.6 % |                                91.1 % / 83.1 % |                   91.2 % / 82.5 % |             68.9 % / 67.6 % |
| 0.6                        |                27.5 % |                                93.3 % / 82.3 % |                   92.4 % / 82.5 % |             73.5 % / 67.1 % |
| 0.7                        |                13.5 % |                                96.5 % / 79.2 % |                   94.2 % / 80.7 % |             82.4 % / 63.3 % |
| 0.75 (`DEFAULT_MIN_SCORE`) |                 6.5 % |                                98.4 % / 76.2 % |                   96.7 % / 77.4 % |             88.2 % / 57.7 % |
| 0.8                        |                 3.2 % |                                99.7 % / 69.5 % |                   99.2 % / 72.3 % |             93.2 % / 47.3 % |
| 0.85 (`CONFIDENT_SCORE`)   |                 1.8 % |                                99.9 % / 56.5 % |                  100.0 % / 62.3 % |             95.5 % / 34.5 % |
| 0.9                        |                 1.0 % |                               100.0 % / 34.2 % |                  100.0 % / 40.3 % |             98.0 % / 17.2 % |

"Right" is JI ≥ 0.9 (SmartDoc) or SmartDoc-style JI ≥ 0.9 (MIDV); recall is the share of all
frames with a right quad (diagnostic run with `minArea 0.02`). The rates are per frame. Live,
the `QuadTracker` and the auto-capture stability window (8 stable frames, then 1 s stable and
sharp) suppress isolated false quads further.

A hybrid scanner can call `detectDocument(gray, { minScore: 0 })` and accept the classic quad
when its score is at least `CONFIDENT_SCORE`, and run a learned detector on the other frames
(no quad or a lower score). On SmartDoc held-out frames that hands about 38 % of the frames to
the learned detector, on MIDV about 64 %.

## Results after tuning (v2)

Runs `classic-v2` (all datasets, sizes 256/384/512, default options) and `classic-v2-relaxed`
(384, `minArea 0.02, minScore 0`, diagnostic). Long side 384 unless noted.

### By split

| subset (frames)                    | version | JI    |  found | IoU ≥ 0.9 | IoU ≥ 0.95 | corner err | p90 err |
| ---------------------------------- | ------- | ----- | -----: | --------: | ---------: | ---------: | ------: |
| SmartDoc tuning (12,622)           | v1      | 0.620 | 66.8 % |    59.0 % |     57.7 % |     1.58 % |  3.40 % |
|                                    | v2      | 0.758 | 77.4 % |    76.1 % |     74.3 % |     0.24 % |  0.28 % |
| SmartDoc held-out (12,267)         | v1      | 0.652 | 70.2 % |    63.4 % |     63.3 % |     1.93 % |  0.33 % |
|                                    | v2      | 0.786 | 80.0 % |    77.5 % |     77.0 % |     0.20 % |  0.20 % |
| SmartDoc held-out, bg 1–4 (10,989) | v1      | 0.728 | 78.4 % |    70.8 % |     70.6 % |     1.92 % |  0.32 % |
|                                    | v2      | 0.877 | 89.3 % |    86.5 % |     86.0 % |     0.20 % |  0.20 % |
| SmartDoc all (24,889)              | v1      | 0.636 | 68.5 % |    61.1 % |     60.4 % |     1.75 % |  2.10 % |
|                                    | v2      | 0.772 | 78.7 % |    76.8 % |     75.7 % |     0.22 % |  0.25 % |
| MIDV-500, held out (3,000)         | v1      | 0.789 | 87.8 % |    72.2 % |     69.5 % |     3.06 % |  6.66 % |
|                                    | v2      | 0.622 | 65.7 % |    59.7 % |     58.8 % |     1.26 % |  1.42 % |
| MIDV-2019, held out (1,200)        | v1      | 0.690 | 85.8 % |    59.7 % |     58.6 % |     4.82 % | 22.85 % |
|                                    | v2      | 0.587 | 64.4 % |    53.3 % |     52.2 % |     1.82 % |  2.49 % |

SmartDoc JI per background and split:

| split: version                                | bg 01 | bg 02 | bg 03 | bg 04 | bg 05 | bg 1–4 |   all |
| --------------------------------------------- | ----: | ----: | ----: | ----: | ----: | -----: | ----: |
| tuning: v1                                    | 0.924 | 0.618 | 0.835 | 0.260 | 0.003 |  0.691 | 0.620 |
| tuning: v2                                    | 0.973 | 0.869 | 0.977 | 0.450 | 0.000 |  0.845 | 0.758 |
| tuning: v2 best candidate without threshold   | 0.982 | 0.969 | 0.989 | 0.848 | 0.122 |  0.954 | 0.869 |
| **held-out: v1**                              | 0.960 | 0.659 | 0.836 | 0.316 | 0.000 |  0.728 | 0.652 |
| **held-out: v2**                              | 0.960 | 0.892 | 0.990 | 0.557 | 0.000 |  0.877 | 0.786 |
| held-out: v2 best candidate without threshold | 0.982 | 0.977 | 0.990 | 0.897 | 0.099 |  0.968 | 0.877 |

Other sizes (all SmartDoc frames, JI): 256: 0.585 → 0.680, 512: 0.658 → 0.799. With the
`QuadTracker` at 384: 0.670 → 0.791.

MIDV-500 per condition (v1 → v2): table 0.990 → 0.984, keyboard 0.978 → 0.955, clutter
0.884 → 0.637, hand 0.755 → 0.436, partial (out of frame) 0.338 → 0.095. MIDV-2019: low
light 0.981 → 0.920, distorted 0.399 → 0.254.

### No-document negatives

785 COCO photos of tables, desks, rooms, screens, laptops and keyboards (see
[Datasets](#datasets)); share of photos with a returned quad, at 384:

| version            | any quad (default `minScore`) | score ≥ 0.6 | score ≥ 0.7 | score ≥ 0.8 | score ≥ 0.85 |
| ------------------ | ----------------------------: | ----------: | ----------: | ----------: | -----------: |
| v1 (minScore 0.5)  |                        20.4 % |       7.0 % |       3.1 % |       1.0 % |        1.0 % |
| v2 (minScore 0.75) |                         6.5 % |       6.5 % |       6.5 % |       3.2 % |        1.8 % |

v2 scores are on a different scale (no polarity requirement, weaker size preference), so
equal thresholds are not comparable across versions; compare the default columns. The
remaining v2 false positives are mostly real rectangles: laptop and TV screens, microwave
doors, cabinet fronts and picture frames.

### Synthetic eval (300 seeds per class, default options)

| class                            | v1 IoU ≥ 0.9 | v2 IoU ≥ 0.9 |
| -------------------------------- | -----------: | -----------: |
| easy, perspective, noisy, shadow |        100 % |        100 % |
| low contrast                     |         85 % |         69 % |
| border                           |         99 % |         96 % |
| textured                         |       97.7 % |       98.3 % |
| none (false positives)           |        6.3 % |        9.3 % |

The low-contrast class loses mostly to the calibrated threshold (79 % of its documents are
found as the best candidate without a threshold, 86 % in v1) and to the weaker size preference
(text blocks inside the synthetic page are not penalized for being smaller).

### Runtime

Single thread, every 10th SmartDoc frame (2,555 frames), v1 and v2 measured back to back in
one session (another job was using the machine, so absolute numbers are about 0.3 ms above
the v1 measurement below):

| long side | v1 mean / p95 ms | v2 mean / p95 ms |
| --------- | ---------------: | ---------------: |
| 256       |        2.4 / 3.7 |        2.6 / 3.9 |
| 384       |        4.6 / 5.7 |        5.0 / 6.4 |
| 512       |        7.6 / 9.3 |       8.1 / 10.0 |

### Failure causes after tuning (384)

| dataset  | failures | right quad ranked below a wrong one | right quad below `minScore` | right quad only among relaxed runner-ups | no right candidate |
| -------- | -------: | ----------------------------------: | --------------------------: | ---------------------------------------: | -----------------: |
| SmartDoc |    5,783 |                               4.6 % |                      26.0 % |                                    7.5 % |             61.9 % |
| – bg 01  |      145 |                               5.5 % |                      23.4 % |                                    2.8 % |             68.3 % |
| – bg 02  |      668 |                               5.8 % |                      67.7 % |                                   11.5 % |             15.0 % |
| – bg 03  |       38 |                               0.0 % |                     100.0 % |                                    0.0 % |              0.0 % |
| – bg 04  |    2,355 |                               9.4 % |                      41.5 % |                                   14.9 % |             34.2 % |
| – bg 05  |    2,577 |                               0.0 % |                       0.1 % |                                    0.0 % |             99.9 % |
| MIDV-500 |    1,208 |                               8.6 % |                      30.0 % |                                   12.6 % |             48.8 % |

No failure is caused by `minArea` anymore. Most remaining SmartDoc failures are background 5
(clutter, 45 % of the failures, no right candidate) and background 4, where the faint paper
edge often loses to a text column or picture edge on the page, or scores below the threshold.

## Experiment log

All experiments ran on the SmartDoc tuning split at 384 px plus the negatives, with
`minArea 0.02, minScore 0`, so that ranking and calibration are measured separately from the
thresholds. Each line gives the tuning-split JI of the best candidate without a threshold
("any") and the JI at the score thresholds where 10 %, 5 % and 2 % of the negatives yield a
quad ("FP10 / FP5 / FP2"). Held-out data was looked at once, after the defaults were frozen
(commit `e95991f`).

| #   | hypothesis → change                                                                                                                        | tuning result (any / FP10 / FP5 / FP2)                                                      | decision                     |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | ---------------------------- |
| 0   | baseline v1 (default run: JI 0.620, negatives 20.4 %)                                                                                      | 0.791 / 0.626 / 0.397 / 0.017                                                               | –                            |
| 1   | 24 % of failures are documents below `minArea` 0.08 → 0.02 (default thresholds)                                                            | default run 0.620 → 0.687, negatives 21.7 %                                                 | keep (0.04, see #12)         |
| 2   | extended sides and inner text columns lack support near their ends → add the weakest end-quarter support to the score                      | 0.783 / 0.605 / 0.537 / 0.021                                                               | revert (see #4)              |
| 3   | the size preference (full score from 40 % of the frame) lets the table outline beat a 10 % document → full score from 10 %                 | 0.793 / 0.627 / 0.596 / 0.456 (3 %: 0.767 / … / 0.433; 20 %: 0.796 / 0.622 / 0.579 / 0.148) | keep 10 %                    |
| 4   | end support (#2) with #3: weight 0 vs 0.1                                                                                                  | 0.804 / 0.669 / 0.624 / 0.450 vs 0.801 / 0.647 / 0.611 / 0.480                              | drop the end term            |
| 5   | text-column sides have mixed contrast polarity in one bin → count only single-polarity bins                                                | 0.804 / 0.668 / 0.623 / 0.448                                                               | revert (no effect)           |
| 6   | glare flips the contrast along a side on bg 04 → count opposite-polarity support with weight 0.5 / 0.75 / 1                                | 0.839 / 0.706 / 0.649 / 0.440; 0.855 / 0.746 / 0.661 / 0.436; 0.864 / 0.783 / 0.672 / 0.289 | weight 1 (see #7, #8)        |
| 7   | paper is brighter than what it lies on, screens and doors are not → score × (inside / band)^e if the inside is darker                      | e 0.5: 0.865 / 0.780 / 0.732 / 0.585; e 1: 0.861 / 0.769 / 0.716 / 0.556                    | keep e 0.5                   |
| 8   | with weight 1, polarity is irrelevant → drop the polarity bookkeeping                                                                      | 0.866 / 0.781 / 0.729 / 0.581                                                               | keep (simpler, same result)  |
| 9   | a quad bounded by a text column should grow onto the faint paper edge beyond a bright margin → move sides outwards onto parallel lines     | 0.836–0.867 / 0.736–0.785 / 0.696–0.742 / 0.603–0.638, bg 01 0.982 → 0.940–0.971            | revert (bright table strips) |
| 10  | more candidates or lines, other Canny thresholds (refine 10, 40 lines, high = 2 or 3 × median)                                             | 0.865–0.870 / 0.778–0.782 / 0.721–0.728 / 0.553–0.582                                       | revert (flat)                |
| 11  | score weights mean / weakest 0.75 / 0.25 or 0.5 / 0.5 instead of 0.6 / 0.4                                                                 | 0.868 / 0.792 / 0.742 / 0.535; 0.865 / 0.772 / 0.721 / 0.583                                | keep 0.6 / 0.4 (trade-off)   |
| 12  | `minArea` 0.02 / 0.04 / 0.06 with #8                                                                                                       | 0.866 / 0.781 / 0.729 / 0.581 (0.02 = 0.04); 0.864 / 0.778 / 0.724 / 0.581                  | 0.04                         |
| 13  | calibrate `minScore` on the negatives (precision and recall per threshold on the tuning split)                                             | 0.75: negatives 6.6 %, 98 % of returned quads right; 0.85: 1.9 %, 99.9 %                    | 0.75 default, 0.85 confident |
| 14  | a page wins over a block of its own content if it encloses it, scores ≥ 0.85 × the block and the margin is at least as bright as the block | 0.869 / 0.781 / 0.731 / 0.606 (bg 04 0.832 → 0.848); margin ≥ 0.9 × block: bg 01 0.970      | keep (≥ 1 × block)           |
| 15  | full score from 20 % of the frame with #14 (to recover the synthetic low-contrast class)                                                   | 0.864 / 0.743 / 0.635 / 0.348                                                               | revert                       |

After freezing, one diagnostic ablation was run on the held-out MIDV data to explain the MIDV
regression (best candidate without threshold, MIDV-500 / MIDV-2019; v2: 0.786 / 0.755): size
preference back to 40 %: 0.844 / 0.725; without the brightness factor: 0.772 / 0.749;
without the page-over-content rule: 0.784 / 0.751. It was not used to change the detector.

## Baseline results (v1, before tuning)

### SmartDoc 2015 Challenge 1 (all 24,889 frames)

| long side   | JI    |  found |   IoU | IoU ≥ 0.9 | IoU ≥ 0.95 | corner err | p90 err |
| ----------- | ----- | -----: | ----: | --------: | ---------: | ---------: | ------: |
| 256         | 0.585 | 64.5 % | 0.588 |    55.9 % |     55.8 % |     2.45 % |  4.20 % |
| 256 tracked | 0.616 | 72.2 % | 0.619 |    55.0 % |     40.1 % |     3.38 % | 12.99 % |
| 384         | 0.636 | 68.5 % | 0.639 |    61.1 % |     60.4 % |     1.75 % |  2.10 % |
| 384 tracked | 0.670 | 76.7 % | 0.673 |    60.0 % |     43.5 % |     2.64 % |  4.60 % |
| 512         | 0.658 | 70.5 % | 0.661 |    63.5 % |     62.7 % |     1.64 % |  1.49 % |
| 512 tracked | 0.691 | 79.1 % | 0.694 |    62.1 % |     45.0 % |     2.68 % |  4.58 % |

JI per background:

| background                                    | frames |   256 |   384 |   512 | 384 tracked | 384 diagnostic (minArea 0.02, minScore 0) |
| --------------------------------------------- | -----: | ----: | ----: | ----: | ----------: | ----------------------------------------: |
| 01: wooden desk, table top in view            |  6,180 | 0.889 | 0.942 | 0.956 |       0.915 |                                     0.947 |
| 02: light grey round table, small document    |  6,011 | 0.560 | 0.639 | 0.676 |       0.701 |                                     0.949 |
| 03: dark table                                |  5,952 | 0.827 | 0.836 | 0.852 |       0.844 |                                     0.960 |
| 04: white table, glare                        |  4,169 | 0.188 | 0.287 | 0.321 |       0.421 |                                     0.619 |
| 05: cluttered desk, stacked papers, occlusion |  2,577 | 0.001 | 0.001 | 0.001 |       0.008 |                                     0.092 |
| backgrounds 1–4                               | 22,312 | 0.653 | 0.709 | 0.734 |       0.746 |                                     0.890 |
| all                                           | 24,889 | 0.585 | 0.636 | 0.658 |       0.670 |                                     0.807 |

Per background at 384:

| background | JI    |  found |   IoU | IoU ≥ 0.9 | IoU ≥ 0.95 | corner err | p90 err |
| ---------- | ----- | -----: | ----: | --------: | ---------: | ---------: | ------: |
| 01         | 0.942 | 99.9 % | 0.945 |    89.6 % |     88.9 % |     1.39 % |  1.97 % |
| 02         | 0.639 | 64.9 % | 0.639 |    64.8 % |     64.8 % |     0.13 % |  0.20 % |
| 03         | 0.836 | 88.6 % | 0.836 |    84.1 % |     83.9 % |     1.93 % |  0.15 % |
| 04         | 0.287 | 39.9 % | 0.296 |    18.6 % |     15.9 % |     5.85 % | 16.83 % |
| 05         | 0.001 |  1.2 % | 0.002 |     0.0 % |      0.0 % |    29.66 % | 39.30 % |

Per document type at 384:

| document type | frames | JI    |  found | IoU ≥ 0.9 | corner err |
| ------------- | -----: | ----- | -----: | --------: | ---------: |
| datasheet     |  4,391 | 0.759 | 78.8 % |    73.5 % |     0.64 % |
| letter        |  4,230 | 0.660 | 70.6 % |    64.6 % |     1.83 % |
| magazine      |  4,195 | 0.475 | 55.0 % |    41.3 % |     3.28 % |
| paper         |  4,097 | 0.744 | 80.2 % |    71.4 % |     1.80 % |
| patent        |  4,036 | 0.620 | 66.0 % |    61.5 % |     1.32 % |
| tax           |  3,940 | 0.549 | 59.5 % |    53.7 % |     2.23 % |

When the detector finds the right document, it is precise. On frames with JI ≥ 0.9, the
corner error has a median of 0.11 % and a p90 of 0.17 % of the diagonal (about 2–4 px in
the 1920×1080 frame), and IoU ≥ 0.9 and IoU ≥ 0.95 differ by only 0.7 points. The losses come from frames with no detection (31.5 % of all frames) or
a wrong quad (7.4 %). The tracker raises JI by holding the last quad over short misses, but
its smoothing lag lowers the IoU ≥ 0.95 rate on moving clips.

### MIDV-500 (10 of 50 types, 3,000 frames)

| long side   | JI    |  found |   IoU | IoU ≥ 0.9 | IoU ≥ 0.95 | corner err | p90 err |
| ----------- | ----- | -----: | ----: | --------: | ---------: | ---------: | ------: |
| 256         | 0.731 | 84.6 % | 0.739 |    65.2 % |     62.7 % |     3.96 % | 12.12 % |
| 384         | 0.789 | 87.8 % | 0.797 |    72.2 % |     69.5 % |     3.06 % |  6.66 % |
| 384 tracked | 0.788 | 93.0 % | 0.798 |    64.3 % |     42.0 % |     3.97 % |  8.65 % |
| 512         | 0.806 | 89.5 % | 0.814 |    73.9 % |     70.9 % |     3.05 % |  6.08 % |

| condition (600 frames each) |   256 |   384 |   512 | 384 tracked | 384 IoU ≥ 0.9 |
| --------------------------- | ----: | ----: | ----: | ----------: | ------------: |
| table                       | 0.984 | 0.990 | 0.990 |       0.950 |       100.0 % |
| keyboard                    | 0.981 | 0.978 | 0.976 |       0.942 |        96.3 % |
| clutter                     | 0.678 | 0.884 | 0.928 |       0.870 |        80.3 % |
| hand                        | 0.705 | 0.755 | 0.753 |       0.771 |        56.8 % |
| partial (out of frame)      | 0.306 | 0.338 | 0.385 |       0.407 |        27.5 % |
| all                         | 0.731 | 0.789 | 0.806 |       0.788 |        72.2 % |

By the number of GT corners inside the frame, at 384. These are the subsets reported by Tropin
et al.:

| subset                    | frames | JI    | IoU ≥ 0.9 | corner err |
| ------------------------- | -----: | ----- | --------: | ---------: |
| all frames                |  3,000 | 0.789 |    72.2 % |     3.06 % |
| at least 3 corners inside |  2,409 | 0.886 |    81.9 % |     1.50 % |
| all 4 corners inside      |  1,910 | 0.918 |    85.9 % |     1.12 % |

### MIDV-2019 (10 of 50 types, 1,200 frames of 2160×3840)

| long side   | JI    |  found |   IoU | IoU ≥ 0.9 | IoU ≥ 0.95 | corner err | p90 err |
| ----------- | ----- | -----: | ----: | --------: | ---------: | ---------: | ------: |
| 256         | 0.670 | 81.4 % | 0.671 |    55.1 % |     51.7 % |     4.06 % | 14.68 % |
| 384         | 0.690 | 85.8 % | 0.691 |    59.7 % |     58.6 % |     4.82 % | 22.85 % |
| 384 tracked | 0.686 | 92.0 % | 0.689 |    51.9 % |     33.1 % |     5.78 % | 23.27 % |
| 512         | 0.694 | 86.5 % | 0.695 |    61.3 % |     60.9 % |     4.98 % | 22.56 % |

| condition (600 frames each)               |   256 |   384 |   512 | 384 IoU ≥ 0.9 |
| ----------------------------------------- | ----: | ----: | ----: | ------------: |
| low light                                 | 0.956 | 0.981 | 0.987 |        96.2 % |
| distorted (strong perspective, hand-held) | 0.384 | 0.399 | 0.402 |        23.2 % |

Low light alone is not a problem for the detector. The "distorted" clips are: the card held at a
steep angle in one hand, with a thumb over an edge, in front of a table with books and papers.

### Runtime

Single thread, Node 26.12 on an Apple M5 Pro, every 10th SmartDoc frame (2,555 frames). Only
the `detectDocument` call on the already downscaled frame is timed:

| long side (frame size) | mean ms | p95 ms |
| ---------------------- | ------: | -----: |
| 256 (256×144)          |     2.4 |    3.6 |
| 384 (384×216), default |     4.3 |    5.3 |
| 512 (512×288)          |     7.2 |    8.8 |

The iPhone lab measured 5.4 ms mean at 384 in a worker (see the platform notes).

## Comparison with published results

SmartDoc 2015 Challenge 1, Jaccard index (official protocol, mean over frames). "Learned"
marks trained neural networks. Values are copied from the cited tables. Where two sources
disagree, both are given.

| method                                                    | learned? | bg 1   | bg 2   | bg 3   | bg 4   | bg 5   | all                   | source of the numbers |
| --------------------------------------------------------- | -------- | ------ | ------ | ------ | ------ | ------ | --------------------- | --------------------- |
| HU-PageScan (das Neves Jr. et al. 2020)                   | yes      | –      | –      | –      | –      | –      | 0.9923                | [T22]                 |
| LDRNet-1.4 (Teng et al. 2022)                             | yes      | 0.9877 | 0.9838 | 0.9862 | 0.9802 | 0.9858 | 0.9849                | [T22]                 |
| JCD+CSR (Zhu et al. 2019)                                 | yes      | 0.988  | 0.984  | 0.983  | 0.984  | 0.961  | 0.982                 | [Tr20]                |
| Recursive CNN (Javed & Shafait 2017, "SEECS-NUST-2")      | yes      | 0.9832 | 0.9724 | 0.9830 | 0.9695 | 0.9478 | 0.9743                | [T22]; [Tr20]: 0.978  |
| Advanced Hough (Tropin et al. 2021)                       | no       | 0.9886 | 0.9858 | 0.9896 | 0.9806 | –      | 0.9866 (bg 1–4)       | [Tr21]                |
| LRDE (2015 entry)                                         | no       | 0.9869 | 0.9775 | 0.9889 | 0.9837 | 0.8613 | 0.9716                | [T22] (from [B15])    |
| Ngoc et al. 2019 ("LRDE-3")                               | no       | 0.985  | 0.982  | 0.987  | 0.980  | 0.848  | 0.970                 | [Tr20]                |
| Smart Engines (2015 entry; Zhukovsky et al. 2017)         | no       | 0.9885 | 0.9833 | 0.9897 | 0.9785 | 0.6884 | 0.9548                | [T22] (from [B15])    |
| Contours and contrasts (Tropin et al. 2020)               | no       | 0.983  | 0.974  | 0.983  | 0.970  | 0.319  | 0.910                 | [Tr20]                |
| GOP (Leal & Bezerra 2016)                                 | no       | 0.961  | 0.944  | 0.965  | 0.930  | 0.412  | 0.896                 | [Tr20]                |
| NetEase (2015 entry)                                      | n/v      | 0.9624 | 0.9552 | 0.9621 | 0.9511 | 0.2218 | 0.8820                | [T22] (from [B15])    |
| Puybareau & Géraud 2018 (real-time, 40 ms)                | no       | 0.905  | 0.936  | 0.859  | 0.903  | –      | –                     | [P18]                 |
| RPPDI-UPE (2015 entry)                                    | n/v      | 0.8274 | 0.9104 | 0.9697 | 0.3649 | 0.2163 | 0.7408                | [T22] (from [B15])    |
| SEECS-NUST (2015 entry)                                   | n/v      | 0.8875 | 0.8264 | 0.7832 | 0.7811 | 0.0113 | 0.7393                | [T22] (from [B15])    |
| Average of the 2015 entries                               | –        | 0.947  | 0.903  | 0.938  | 0.812  | 0.404  | 0.855                 | [Tr20]                |
| **scanoficus v2, default, 384**                           | no       | 0.967  | 0.880  | 0.984  | 0.500  | 0.000  | 0.772 (bg 1–4: 0.861) | this page             |
| scanoficus v2, default, 512                               | no       | 0.967  | 0.971  | 0.989  | 0.526  | 0.000  | 0.799                 | this page             |
| scanoficus v2, best candidate without threshold, 384      | no       | 0.982  | 0.973  | 0.990  | 0.871  | 0.110  | 0.873 (bg 1–4: 0.961) | this page             |
| scanoficus v1, default, 384                               | no       | 0.942  | 0.639  | 0.836  | 0.287  | 0.001  | 0.636 (bg 1–4: 0.709) | this page             |
| scanoficus v1, default, 512                               | no       | 0.956  | 0.676  | 0.852  | 0.321  | 0.001  | 0.658                 | this page             |
| scanoficus v1, diagnostic (minArea 0.02, minScore 0), 384 | no       | 0.947  | 0.949  | 0.960  | 0.619  | 0.092  | 0.807                 | this page             |

n/v: the method type was not verified (the 2015 competition paper was not accessible).
The A2iA and ISPL-CVML entries of the 2015 competition are missing because no accessible
source reproduces their numbers.

MIDV-500, SmartDoc-style JI ([Tr20], [Tr21]). The published numbers use all 50 types
(15,000 frames), while ours use 10 types (3,000 frames), so the comparison is indicative only:

| method                                                           | learned? | all frames | ≥ 3 corners inside | 4 corners inside |
| ---------------------------------------------------------------- | -------- | ---------: | -----------------: | ---------------: |
| Advanced Hough (Tropin et al. 2021)                              | no       |     0.9056 |             0.9788 |           0.9830 |
| Contours and contrasts (Tropin et al. 2020)                      | no       |      0.870 |              0.961 |            0.972 |
| Recursive CNN (Javed & Shafait), SmartDoc-trained, run by [Tr20] | yes      |      0.626 |              0.705 |            0.739 |
| OctHU-PageScan, pretrained, run by [Tr20]                        | yes      |      0.319 |              0.374 |            0.403 |
| **scanoficus v2, default, 384**                                  | no       |      0.622 |              0.733 |            0.795 |
| scanoficus v1, default, 384                                      | no       |      0.789 |              0.886 |            0.918 |
| scanoficus v1, diagnostic, 384                                   | no       |      0.820 |              0.908 |            0.932 |

No published MIDV-2019 result with the same metric was found.

Sources:

- [B15] J.-C. Burie, J. Chazalon, M. Coustaty et al., "ICDAR2015 Competition on Smartphone
  Document Capture and OCR (SmartDoc)", ICDAR 2015.
- [T22] W. Teng et al., "LDRNet: Enabling Real-time Document Localization on Mobile Devices",
  [arXiv:2206.02136](https://arxiv.org/abs/2206.02136), Table 1. It reproduces the 2015 entries
  from [B15].
- [Tr20] D. V. Tropin, S. A. Ilyuhin, D. P. Nikolaev, V. V. Arlazarov, "Approach for Document
  Detection by Contours and Contrasts", ICPR 2020,
  [arXiv:2008.02615](https://arxiv.org/abs/2008.02615), SmartDoc and MIDV-500 tables.
- [Tr21] D. V. Tropin, A. M. Ershov, D. P. Nikolaev, V. V. Arlazarov, "Advanced Hough-based
  method for on-device document localization", Computer Optics 45(5), 2021,
  [arXiv:2106.09987](https://arxiv.org/abs/2106.09987), Tables 1 and 2 (65 ms per frame on an
  iPhone 6).
- [P18] É. Puybareau, T. Géraud, "Real-time document detection in smartphone videos", ICIP
  2018, Table 1.
- The original papers cited by these tables are K. Javed, F. Shafait, ICDAR 2017; R. B. das
  Neves Jr. et al., IET Image Processing 14(15), 2020; A. Zhu et al. 2019; N. Ngoc et al.
  2019; P. Leal, B. Bezerra 2016; A. Zhukovsky et al. 2017.

## Failure analysis (baseline v1)

This analysis of v1 guided the tuning; see [Failure causes after tuning](#failure-causes-after-tuning-384) for v2.
A failure is a frame with JI < 0.9 (SmartDoc) or IoU < 0.9 (MIDV) at long side 384. There are
two complementary views.

### Automatic: where the right quad was lost

The default run is compared frame by frame with the diagnostic run (`minArea 0.02, minScore
0`). The categories are checked in the order of the columns, so every failure counts once:

| dataset   | failures | document < 8 % of the frame (below default `minArea`) | right quad was a runner-up (ranking) | right quad scored < `minScore` | right quad only among relaxed runner-ups | no right candidate |
| --------- | -------: | ----------------------------------------------------: | -----------------------------------: | -----------------------------: | ---------------------------------------: | -----------------: |
| SmartDoc  |    9,670 |                                                23.9 % |                                7.6 % |                          8.5 % |                                    9.0 % |             51.0 % |
| – bg 01   |      661 |                                                14.1 % |                               49.0 % |                          0.9 % |                                    0.3 % |             35.7 % |
| – bg 02   |    2,113 |                                                68.2 % |                                0.1 % |                         14.0 % |                                   12.1 % |              5.5 % |
| – bg 03   |      948 |                                                82.3 % |                                5.9 % |                          7.0 % |                                    4.0 % |              0.8 % |
| – bg 04   |    3,371 |                                                 0.0 % |                               10.4 % |                         13.5 % |                                   16.8 % |             59.3 % |
| – bg 05   |    2,577 |                                                 0.0 % |                                0.0 % |                          0.0 % |                                    0.2 % |             99.7 % |
| MIDV-500  |      834 |                                                 0.0 % |                               25.5 % |                          3.2 % |                                    4.2 % |             67.0 % |
| MIDV-2019 |      484 |                                                 2.3 % |                               24.6 % |                          2.3 % |                                    2.5 % |             68.4 % |

"Candidates" are the top 6 quads after refinement. "No right candidate" means that none of
them reaches 0.9, even with the relaxed options.

### Visual: what the failing frames look like

`pnpm bench:montage --mode failures --per-clip 1` draws a seeded random sample of failures.
It takes at most one frame per clip and up to 24 per background, or 12 per MIDV condition.
I viewed every sampled frame (112 SmartDoc, 36 MIDV-500 from clutter, hand and partial, and
12 MIDV-2019 distorted) and assigned it to one category. Shares are weighted by the number
of failures per background or condition. They are approximate, because the samples are
small (12–24 per stratum).

SmartDoc (9,670 failures):

| failure mode                                                                                                                               | share | where                      |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ----: | -------------------------- |
| Low contrast: white paper on a light table (often with glare), no detection; in bg 02 the document is also small (median 9 % of the frame) |  51 % | bg 02 (all), bg 04 (20/24) |
| Clutter: stacked papers under the page, pens and a cable across the edges, magazine nearby; no detection                                   |  27 % | bg 05 (all)                |
| Wrong rectangle: the table top outline wins over the document                                                                              |   7 % | bg 01 (8/22), bg 03 (9/18) |
| Small document with good contrast, not detected (below `minArea`)                                                                          |   5 % | bg 03 (9/18)               |
| A side snaps to inner content (text column, margin) because the paper edge is invisible                                                    |   4 % | bg 04 (3/24)               |
| A side snaps to a background line (far table edge) instead of the document edge                                                            |   4 % | bg 01 (8/22), bg 04 (1/24) |
| A side is closed by the frame border instead of the visible document edge                                                                  |   2 % | bg 01 (6/22)               |

Blur showed up in only one sampled SmartDoc failure, and hand occlusion and documents leaving
the frame did not show up at all.

MIDV-500 (834 failures; keyboard failures, 3 % of all failures, were not sampled):

| failure mode                                                                                        | share | where                                       |
| --------------------------------------------------------------------------------------------------- | ----: | ------------------------------------------- |
| Wrong rectangle or a side on a background line (ruler, sticky notes, paper, table edge, guitar)     |  33 % | clutter (7/12), partial (4/12), hand (3/12) |
| Document leaving the frame (partly visible), no detection                                           |  30 % | partial (7/12)                              |
| Fingers cover an edge, so a side snaps to the hand, an inner line or the frame border               |  13 % | hand (5/12)                                 |
| A side on an inner printed line (card band, MRZ, photo box), often because of glare on the laminate |   9 % | hand (2/12), partial (1/12)                 |
| Motion blur or glare, no or a wrong detection                                                       |   8 % | hand (2/12), clutter (2/12)                 |
| Not detected in clutter                                                                             |   4 % | clutter (3/12)                              |

MIDV-2019 (484 failures, 461 of them "distorted"): fingers covering an edge of the hand-held
card (7/12, about 55 %), and a wrong rectangle or a side on background lines from the papers
and books behind it (5/12, about 40 %). The 23 low-light failures (5 %) are almost all ranking
errors (87 %).

### What this meant for an NN detector (v1)

1. **About 40 % of the SmartDoc failures are parameter and scoring problems, not missing
   evidence.** These are documents below `minArea`, correct quads below `minScore` (the
   `area^0.25` size preference caps the score of a perfect 10 % document at about 0.71), and
   larger rectangles winning the ranking. On MIDV, ranking errors are 25 % of the failures.
   The diagnostic run brings backgrounds 1–3 to 0.95–0.96, close to the 0.97–0.99 of the
   best classic methods. A smaller minimum area, a weaker size preference and a ranking that
   penalizes quads with uniform contents or a table-like outline would address this without
   an NN.
2. **Background 4 (white paper on a white table with glare) is a weakness of this edge
   extraction, not of classic methods in general.** Classic methods reach 0.97–0.98 there
   (LRDE, Smart Engines, Tropin et al.), and scanoficus reaches 0.29 (0.62 relaxed). Most of
   these frames have no right candidate, because the faint paper edge does not survive
   Canny with median-relative thresholds at 384 px. Better low-contrast edges (color or
   chroma gradients, adaptive local thresholds, higher resolution near candidate sides) are
   the classic fix to try first.
3. **Clutter, stacked papers and occlusion (background 5, hand-held MIDV) are where learned
   models clearly win.** The best classic results on background 5 are 0.32–0.86, and learned
   models reach 0.95–0.99 (Javed & Shafait, Zhu et al., LDRNet). Here scanoficus finds the
   right quad in 0.3 % of failing frames even with relaxed options. On MIDV, fingers over
   an edge and documents partly outside the frame are the main open cases. Together these
   cases make up about 27 % of the SmartDoc failures, about 45 % of the MIDV-500 failures
   and about 55 % of the MIDV-2019 failures.
4. **Precision is not the issue.** When the right document is found, the corners are accurate
   (p90 corner error 0.15–2 % on backgrounds 1–3), and long side 384 costs 4.3 ms. Going to
   512 gains 0.02 JI at 1.7× the time.

In short, relaxed thresholds already bring backgrounds 1–3 to 0.95–0.96, and background 4
needs better low-contrast edges, which classic methods are known to handle. The cases where
only learned models are known to succeed are clutter, stacked papers and occlusion by hands
or objects. The evidence therefore supports an optional NN detector for those scenes, after
the classic tuning, rather than as a replacement for the classic path. Tuning should be
validated with this benchmark, preferably with a split by clip so that no parameters are
fitted to the frames they are scored on.

### What remains for an NN detector (after v2)

1. **Clutter, stacked papers and occlusion.** Background 5 is still at 0.000 (0.11 without a
   threshold, no right candidate in 99.9 % of its failures), and fingers over an edge (MIDV
   "hand", MIDV-2019 "distorted") remain weak. Learned models reach 0.95–0.99 there.
2. **Documents leaving the frame.** MIDV "partial" is at 0.095. The frame-border penalty and
   the threshold reject most of these quads, and a classic detector cannot know where the
   hidden corners are.
3. **White paper on a white table with glare (background 4)** improved from 0.287 to 0.500
   (held out: 0.316 to 0.557; 0.897 without threshold), but its true quads score low and the
   faint edge sometimes loses to content on the page. Color or chroma gradients (the frame is
   decoded to gray before detection) are the classic fix still untried.
4. **Escalation.** With `CONFIDENT_SCORE`, a hybrid would run the NN on about 38 % of
   SmartDoc held-out frames and about 64 % of MIDV frames, and keep the 5 ms classic result
   on the rest (99.9–100 % right on SmartDoc, 95.5 % on MIDV).
