# Auto-capture ("Hold steady")

Auto-capture takes a page when the document is steady and sharp. Before this rework it rarely
fired in real hand-held use. This page describes the offline simulator, the policy that replaced
the old one, and the results on real hand-held video.

## Policy (library defaults)

| step                         | rule                                                                                                                                                                                                                                                                                                            | option (default)                                                                                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Stability (`QuadTracker`)    | Fit a straight-line motion to every corner over the last 500 ms. Drop the worst 25 % of the observations as outliers. Stable if the fastest corner moves at most 0.1 diagonals/s and the rest stays within 1.5 % of the diagonal of its line. Needs at least 4 observations that span at least 250 ms.          | `windowMs` (500), `maxSpeed` (0.1), `outlierFraction` (0.25), `stabilityThreshold` (0.015), `stableFrames` (4) |
| Re-framing                   | A jump of more than 20 % of the diagonal is ignored once (an outlier). A second, consistent jump restarts the tracker and the progress.                                                                                                                                                                         | `resetThreshold` (0.2)                                                                                         |
| Sharpness (`CaptureTrigger`) | The frame's sharpness must be at least 0.6 × the 75th percentile of the frames with a detection in the last 1.5 s (before: 0.6 × a peak that decayed by 0.98 per frame).                                                                                                                                        | `minRelativeSharpness` (0.6), `trigger.sharpnessWindowMs` (1500), `trigger.sharpnessPercentile` (0.75)         |
| Time                         | The progress fills while every criterion holds and decays while one fails (a full progress empties in 1 s). It fires when the document was ready for 600 ms in total (before: 1000 ms without a single bad frame).                                                                                              | `stableMs` (600), `trigger.decayMs` (1000)                                                                     |
| Best frame                   | During the last 300 ms of the progress, the controller also grabs each analyzed video frame at full resolution and keeps only the sharpest ready one (one slot, plus one frame that is being compared). On firing, the kept frame becomes the page. Only with video-frame stills (not `stillCapture: 'photo'`). | `bestFrameMs` (300, 0 = off)                                                                                   |
| Motion sensor (opt-in)       | While scanning, `devicemotion` rotation rates above the limit hold auto-capture for 300 ms. On iOS, call `requestDeviceMotionPermission()` from a tap first. This rule was not measured: the datasets have no gyro data.                                                                                        | `deviceMotion` (false), `maxRotationRate` (10 °/s)                                                             |

A steady hand drifts by about 0.02–0.05 diagonals/s. At 0.1 diagonals/s, a frame exposed for
1/30 s blurs by about 0.3 % of the document. Panning (0.3/s) and walking (about 0.4–0.5/s) fail
the speed rule.

`LiveDetection.autoCapture` reports the first failing criterion (`blocker`: `no-document`,
`holding`, `settling`, `moving`, `jitter`, `blurry`, `device-motion`, `captured`, `disabled`)
together with each measured value and its limit. The demo's debug overlay shows these values.

## Simulator

`pnpm bench:autocapture:features` and then `pnpm bench:autocapture`
(`packages/core/bench/autocapture*.ts`). The simulator replays the stored per-frame detections
of the benchmark runs `nn-own-docquadnet4` ("Neural v4", with its calibrated presence threshold)
and `r2-v3f` (classic, `minScore` 0.74). It uses all hand-held video clips: SmartDoc 2015 Ch.1
(150 clips), MIDV-500 (100) and MIDV-2019 (40). Sharpness is computed per frame as in the live
controller: in the detection's bounding box, at 768 px for the network and at 384 px for the
classic detector.

- **Timing.** SmartDoc frames have no timestamps. The clips run at about 17–20 fps, so the
  simulator uses 50 ms per frame. MIDV frames were extracted at 10 fps (100 ms, 3 s per clip).
  The live loop runs at 17 detections/s, as measured on an iPhone. At each tick it analyzes the
  newest frame, so SmartDoc is subsampled. MIDV is slower than 17/s, so every MIDV frame is
  analyzed. `--rate 100` analyzes every frame; with it, the results change by at most 3 clips.
- **Capture.** The still is the video frame about 60 ms (one detection) after the decision, or
  the frame kept by the best-frame buffer. The page quad is the detection on that frame.
- **Metrics.** For the first capture of each clip: the share of clips with a capture, the median
  time to the capture, the mean JI against the ground truth, bad captures (JI < 0.9), the
  sharpness of the captured frame relative to the sharpest frame of the clip (ground-truth
  region at 768 px), and captures during fast motion (ground-truth corners faster than 0.25
  diagonals/s).
- **Two replays.** The clips were filmed while the camera moved all the time. The median
  ground-truth speed is 0.2–0.3 diagonals/s. Only 9 % of the SmartDoc tuning clips and 24 % of
  the MIDV-500 tuning clips have a 0.5 s period below 0.1/s. The real replay therefore shows
  whether a policy waits for the calm moments. The semi-synthetic replay keeps each frame's
  real detection error, missed detections and sharpness, but replaces the camera motion with a
  synthetic hand: steady (1 % sway at 0.3–0.6 Hz plus tremor), drifting (+0.05/s), panning
  (+0.3/s) or walking (4 % bob at 1.5–2 Hz). This replay is pessimistic for a steady hand,
  because the sharpness still contains the real motion blur of the moving clip.

Design choices were made on the tuning split only (SmartDoc datasheet/magazine/patent and the
MIDV-500 tuning half, `bench/splits.ts`). The held-out numbers were not used for any choice.

### Results (Neural v4, 17 detections/s)

Real replay:

| policy | group               | clips | captured | median time | mean JI | bad | sharpness / best | fast motion |
| ------ | ------------------- | ----: | -------: | ----------: | ------: | --: | ---------------: | ----------: |
| old    | SmartDoc tune       |    75 |      0 % |           – |       – |   – |                – |           – |
| old    | MIDV-500 tune       |    50 |      0 % |           – |       – |   – |                – |           – |
| old    | SmartDoc held-out   |    75 |      1 % |       1.4 s |   0.995 | 0 % |             0.93 |         0 % |
| old    | MIDV-500 held-out\* |    50 |      0 % |           – |       – |   – |                – |           – |
| old    | MIDV-2019 held-out  |    40 |      0 % |           – |       – |   – |                – |           – |
| new    | SmartDoc tune       |    75 |     16 % |       1.0 s |   0.989 | 0 % |             0.93 |         0 % |
| new    | MIDV-500 tune       |    50 |     12 % |       1.0 s |   0.992 | 0 % |             0.82 |         0 % |
| new    | SmartDoc held-out   |    75 |     16 % |       0.8 s |   0.993 | 0 % |             0.82 |         0 % |
| new    | MIDV-500 held-out\* |    50 |      8 % |       1.8 s |   0.992 | 0 % |             0.96 |         0 % |
| new    | MIDV-2019 held-out  |    40 |      8 % |       1.4 s |   0.989 | 0 % |             0.83 |         0 % |

Semi-synthetic hand (captured / median time / sharpness vs best):

| policy | clips          | steady              | drift 0.05/s        | pan 0.3/s | walk |
| ------ | -------------- | ------------------- | ------------------- | --------: | ---: |
| old    | tune (125)     | 33 % / 1.4 s / 0.76 | 28 % / 1.4 s / 0.82 |       0 % |  0 % |
| old    | held-out (165) | 32 % / 1.7 s / 0.81 | 22 % / 1.4 s / 0.83 |       0 % |  0 % |
| new    | tune (125)     | 72 % / 1.1 s / 0.82 | 70 % / 1.1 s / 0.82 |       0 % |  0 % |
| new    | held-out (165) | 72 % / 0.9 s / 0.82 | 70 % / 0.9 s / 0.82 |       0 % |  0 % |

Why the new policy is not ready (analyzed frames before the first capture, tuning clips):
real replay: moving 84 %, ready 7 %, settling 6 %, other criteria 1 % each. Steady hand: ready
40 %, blurry 19 % (real motion blur of the clips), settling 16 %, moving 15 % (the detection
error changes with the real view), jitter 4 %, holding 3 %.

Classic detector (`--detector classic`): the old policy captures in 0–1 % of the real clips. The
new policy captures in 10–15 % of the real clips, and with the steady hand in 70 % (tuning) and
67 % (held-out) of the clips. One weakness is the MIDV-500 held-out half, with a mean JI of
0.853 (14 % bad captures), caused by wrong classic detections.

### Ablation (tuning split, Neural v4)

Each row adds one change to the row above it. Columns: real replay SmartDoc / MIDV-500
(captured, sharpness vs best, fast motion), then steady-hand capture.

| policy                                                             | SmartDoc              | MIDV-500   | steady hand       |
| ------------------------------------------------------------------ | --------------------- | ---------- | ----------------- |
| old (8 consecutive frames ≤ 1.5 %, peak sharpness, 1000 ms, reset) | 0 %                   | 0 %        | 33 %              |
| + decaying progress, percentile sharpness                          | 0 %                   | 0 %        | 50 %              |
| + 600 ms                                                           | 1 %                   | 2 %        | 59 %              |
| + window spread ≤ 2 % around the median (500 ms, 25 % outliers)    | 19 %, 0.82, 21 % fast | 14 %, 0.71 | 66 %              |
| instead: motion fit (speed ≤ 0.1/s, jitter ≤ 1.5 %)                | 16 %, 0.88, 8 % fast  | 10 %, 0.77 | 66 %              |
| + best frame (ideal buffer of all frames of the last 300 ms)       | 16 %, 0.95, 0 % fast  | 10 %, 0.83 | 66 % (0.83 sharp) |
| library (one-slot buffer, ready frames only)                       | 16 %, 0.93, 0 % fast  | 12 %, 0.82 | 72 %              |

The spread rule fires more often, but a fifth of its SmartDoc captures happen during fast motion.
The motion-fit rule separates slow drift, which is allowed, from panning. Other settings had
effects within 1–3 clips: decay 500–2000 ms, window 400–700 ms, jitter 1–2 %, 3 or 4 frames.
A 400 ms ready time caught more clips (24 % / 16 %), but at a lower sharpness (0.83) and with
13 % bad MIDV captures, so the default is 600 ms.

## Limits

- No iPhone recordings are in the evaluation. The per-frame predictions come from benchmark runs
  on the dataset frames, not from the 768-px live pipeline.
- The motion-sensor rule and the cost of the best-frame buffer on a phone (up to two
  full-resolution bitmaps for about 300 ms) were not measured.
- MIDV clips are 3 s at 10 fps, so a time to capture longer than 3 s is not possible there.
