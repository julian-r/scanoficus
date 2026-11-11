# Detector benchmarks on public datasets

This page measures the classic detector (`detectDocument` with default options) on real,
publicly available video frames. Its purpose is to find out how good the detector is before
deciding whether a neural-network detector is worth adding. Nothing in the detector was changed
for these measurements.

## Summary

| dataset (frames)                    | official / main metric |   256 |   384 |   512 | 384 tracked |
| ----------------------------------- | ---------------------- | ----: | ----: | ----: | ----------: |
| SmartDoc 2015 Ch. 1 (24,889, all)   | SmartDoc JI            | 0.585 | 0.636 | 0.658 |       0.670 |
| SmartDoc 2015, backgrounds 1–4      | SmartDoc JI            | 0.653 | 0.709 | 0.734 |       0.746 |
| MIDV-500 subset (3,000)             | SmartDoc-style JI      | 0.731 | 0.789 | 0.806 |       0.788 |
| MIDV-2019 subset (1,200, 4K frames) | SmartDoc-style JI      | 0.670 | 0.690 | 0.694 |       0.686 |

- On SmartDoc the detector is far below published methods. They reach 0.95–0.99 overall (classic
  and learned alike) on backgrounds 1–4, and only learned models reach above 0.95 on the
  cluttered background 5. scanoficus scores 0.942 on background 1 but 0.001 on background 5.
- About 40 % of the SmartDoc failures are not caused by missing edge evidence. In 24 % the
  document covers less than the default `minArea` of 8 % of the frame. SmartDoc documents
  cover a median of 9–14 %. In 8.5 % the right quad was the best candidate but was rejected by
  the `minScore` threshold, and in 7.6 % the right quad lost to a larger rectangle, usually the
  table top. A diagnostic run with `minArea: 0.02, minScore: 0` raises SmartDoc JI from 0.636
  to 0.807, with backgrounds 1–3 at 0.95–0.96.
- The remaining failures produce no right candidate at all: 51 % on SmartDoc, 67 % on MIDV-500
  and 68 % on MIDV-2019. The causes are clutter, stacked papers and objects on the page
  (SmartDoc background 5), very low contrast with glare (background 4), fingers covering an
  edge, documents leaving the frame, and background lines.
- Detector time per frame: 2.4 / 4.3 / 7.2 ms (mean) at long side 256 / 384 / 512, single
  thread, Node 26 on an Apple M5 Pro.

## Datasets

Datasets live outside the repository, in `$SCANOFICUS_DATASETS` (default
`~/datasets/scanoficus/<dataset>/`). They are never committed. `pnpm bench:fetch` downloads,
verifies and unpacks them. Downloads resume after an interruption, and finished steps are
skipped on re-runs. The script needs `curl`, `tar`, `unzip`, `shasum` and `openssl`.

```sh
pnpm bench:fetch                                  # all three (≈ 1 GB + 2 × ≈ 7.5 GB download)
pnpm bench:fetch --datasets smartdoc              # SmartDoc only
pnpm bench:fetch --datasets midv500 --midv-types 01,02,03 --keep-archives
```

| dataset                                                                                          | content used                                                                                                                                          | license / terms                                                                                                                                                                               | access                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [SmartDoc 2015 Challenge 1](https://github.com/jchazalon/smartdoc15-ch1-dataset), release v2.0.0 | All 150 clips (5 backgrounds × 30 documents of 6 types), 24,889 frames of 1920×1080, GT quad per frame                                                | CC BY 4.0 (`LICENCE` in the archive). Cite Burie et al., ICDAR 2015. "Original images remain the property of their respective authors."                                                       | `frames.tar.gz` (1.0 GB) from the GitHub release, verified with the published SHA-256. No registration.                                                                          |
| [MIDV-500](https://arxiv.org/abs/1807.05786)                                                     | 10 of 50 ID document types (every 5th: 01, 06, 11, …, 46). All 10 clips per type (5 conditions × 2 phones), 30 frames each: 3,000 frames of 1080×1920 | `license.txt` on the FTP server: CC BY-SA 2.5. Source images are public-domain or openly licensed Wikimedia images (see `documents.pdf`). Cite Arlazarov et al., Computer Optics 43(5), 2019. | `ftp://smartengines.com/midv-500/dataset/<type>.zip` (≈ 0.65 GB each), verified with `md5.txt`. Anonymous FTP, no registration. Only `images/` and `ground_truth/` are unpacked. |
| [MIDV-2019](https://arxiv.org/abs/1910.04009)                                                    | The same 10 types. All 4 clips per type (distorted / low light × 2 phones), 30 frames each: 1,200 frames of 2160×3840                                 | `license.txt` on the FTP server: CC BY-SA 2.5. Cite Bulatov et al., ICMV 2019 (Proc. SPIE 11433).                                                                                             | `ftp://smartengines.com/midv-500/extra/midv-2019/dataset/<type>.zip` (≈ 0.75 GB each), verified with `md5.txt`. No registration.                                                 |

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

### Running it

```sh
pnpm bench                       # all datasets, sizes 256,384,512, default options → eval-out/bench/full/
pnpm bench --datasets smartdoc --sizes 384 --min-area 0.02 --min-score 0 --run relaxed
pnpm bench --datasets smartdoc --workers 1 --step 10 --run timing
pnpm bench:report --diagnostic relaxed    # markdown tables → packages/core/eval-out/bench/full/report.md
pnpm bench:montage --dataset smartdoc --mode worst      # or --mode failures; PNG pages + legend
```

Results are stored per clip (`eval-out/bench/<run>/<dataset>/<clip>.json`). An interrupted
run resumes and skips finished clips. Delete a run directory to recompute it. Montages
show the ground truth in green, the detection in red and the best runner-up candidate in thin
blue.

## Results

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

| method                                                 | learned? | bg 1   | bg 2   | bg 3   | bg 4   | bg 5   | all                   | source of the numbers |
| ------------------------------------------------------ | -------- | ------ | ------ | ------ | ------ | ------ | --------------------- | --------------------- |
| HU-PageScan (das Neves Jr. et al. 2020)                | yes      | –      | –      | –      | –      | –      | 0.9923                | [T22]                 |
| LDRNet-1.4 (Teng et al. 2022)                          | yes      | 0.9877 | 0.9838 | 0.9862 | 0.9802 | 0.9858 | 0.9849                | [T22]                 |
| JCD+CSR (Zhu et al. 2019)                              | yes      | 0.988  | 0.984  | 0.983  | 0.984  | 0.961  | 0.982                 | [Tr20]                |
| Recursive CNN (Javed & Shafait 2017, "SEECS-NUST-2")   | yes      | 0.9832 | 0.9724 | 0.9830 | 0.9695 | 0.9478 | 0.9743                | [T22]; [Tr20]: 0.978  |
| Advanced Hough (Tropin et al. 2021)                    | no       | 0.9886 | 0.9858 | 0.9896 | 0.9806 | –      | 0.9866 (bg 1–4)       | [Tr21]                |
| LRDE (2015 entry)                                      | no       | 0.9869 | 0.9775 | 0.9889 | 0.9837 | 0.8613 | 0.9716                | [T22] (from [B15])    |
| Ngoc et al. 2019 ("LRDE-3")                            | no       | 0.985  | 0.982  | 0.987  | 0.980  | 0.848  | 0.970                 | [Tr20]                |
| Smart Engines (2015 entry; Zhukovsky et al. 2017)      | no       | 0.9885 | 0.9833 | 0.9897 | 0.9785 | 0.6884 | 0.9548                | [T22] (from [B15])    |
| Contours and contrasts (Tropin et al. 2020)            | no       | 0.983  | 0.974  | 0.983  | 0.970  | 0.319  | 0.910                 | [Tr20]                |
| GOP (Leal & Bezerra 2016)                              | no       | 0.961  | 0.944  | 0.965  | 0.930  | 0.412  | 0.896                 | [Tr20]                |
| NetEase (2015 entry)                                   | n/v      | 0.9624 | 0.9552 | 0.9621 | 0.9511 | 0.2218 | 0.8820                | [T22] (from [B15])    |
| Puybareau & Géraud 2018 (real-time, 40 ms)             | no       | 0.905  | 0.936  | 0.859  | 0.903  | –      | –                     | [P18]                 |
| RPPDI-UPE (2015 entry)                                 | n/v      | 0.8274 | 0.9104 | 0.9697 | 0.3649 | 0.2163 | 0.7408                | [T22] (from [B15])    |
| SEECS-NUST (2015 entry)                                | n/v      | 0.8875 | 0.8264 | 0.7832 | 0.7811 | 0.0113 | 0.7393                | [T22] (from [B15])    |
| Average of the 2015 entries                            | –        | 0.947  | 0.903  | 0.938  | 0.812  | 0.404  | 0.855                 | [Tr20]                |
| **scanoficus, default, 384**                           | no       | 0.942  | 0.639  | 0.836  | 0.287  | 0.001  | 0.636 (bg 1–4: 0.709) | this page             |
| scanoficus, default, 512                               | no       | 0.956  | 0.676  | 0.852  | 0.321  | 0.001  | 0.658                 | this page             |
| scanoficus, diagnostic (minArea 0.02, minScore 0), 384 | no       | 0.947  | 0.949  | 0.960  | 0.619  | 0.092  | 0.807                 | this page             |

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
| **scanoficus, default, 384**                                     | no       |      0.789 |              0.886 |            0.918 |
| scanoficus, diagnostic, 384                                      | no       |      0.820 |              0.908 |            0.932 |

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

## Failure analysis

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

### What this means for an NN detector

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
