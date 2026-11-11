# @scanoficus/detector-nn

An optional neural document detector for scanoficus. It runs **DocQuadNet** (our own model,
349k parameters, 192×192 RGB input, 81 M multiply-adds) on a small WebGL2 runtime written for
this graph, with a plain TypeScript CPU fallback. It plugs into the scanner as a core
`DocumentDetector`.

- The network predicts 4 corners (also outside the frame) and a document-presence probability.
  It handles cluttered backgrounds, white-on-white, hand-held and partly visible pages, where the
  classic edge detector fails (see "Quality" below).
- Two model bundles ship in the package: **round 1** (`DOCQUADNET`, the default) and
  **round 2 with a corner refiner** (`DOCQUADNET2` from `@scanoficus/detector-nn/docquadnet2`):
  a second, 124k-parameter network that moves each corner onto the true corner from 64×64
  crops of the full-resolution still (see "Round 2 and the corner refiner").
- The JavaScript is 18.1 KB min+gzip without the weights. The weights (fp16; round 1 688 kB,
  round 2 688 kB + refiner 244 kB) ship in the package (`models/<bundle>/`) and are fetched at
  runtime; bundlers emit them as content-hashed assets. Each graph names its weights file,
  length and SHA-256, so a graph and weights from different versions never run together.
- It runs on the main thread and in a Web Worker (`OffscreenCanvas`). A self-test after
  initialization catches a broken GPU path (the `'auto'` backend then uses the CPU). It survives
  WebGL context loss and frees all GPU resources on `dispose()`.
- Optional hybrids: run the network first, then classic edge refinement near the predicted
  quad; or run the classic detector first and use the network only when the classic detector
  is unsure.

## Usage

```ts
import { ScannerController } from '@scanoficus/core/controller';
import { createNnDetectorWorker } from '@scanoficus/detector-nn';

const scanner = new ScannerController({
  // Owned by the controller: if the worker fails, the controller falls back to the classic
  // main-thread detector.
  createDetector: () =>
    // The model files shipped in the package are used by default.
    createNnDetectorWorker({ refine: true }), // network first, then classic edge refinement
  // Refinement was tuned on frames downscaled to a long side of 1024 px; the controller gives
  // detectors 384 px by default. A larger size helps refinement (and costs a larger frame copy
  // per detection; not measured on phones).
  detection: { maxSide: 768 },
});
```

`createNnDetectorWorker` returns a detector with `ready()`. It resolves with `{ backend,
precision, ms }` once the worker has loaded the model and initialized its backend. It rejects
if that failed, for example with `backend: 'webgl2'` when WebGL2 does not work on an
`OffscreenCanvas` in the worker. The demo uses this to choose between the worker, the main
thread and the classic fallback (`apps/demo/src/detector.ts`).

With the custom elements, set the same option on the element:
`document.querySelector('scan-document').options = { createDetector: … }`.

On the calling thread (the main thread, or your own worker):

```ts
import { createNnDetector } from '@scanoficus/detector-nn';

const detector = createNnDetector(); // or { graphUrl } for self-hosted files
await detector.ready(); // optional: load and compile ahead of the first frame (the cold start)
const detection = await detector.detect(imageDataOrBitmap); // { quad, score } or null
const raw = await detector.infer(frame); // { quad, presence, heads }, no threshold, no refinement
detector.backend; // 'webgl2' | 'cpu' (the backend that answered the last frame)
detector.precision; // 'float32' | 'float16' | null
detector.glError; // why WebGL2 is not used, with backend 'auto'
detector.dispose();
```

`detect()` returns the quad (TL, TR, BR, BL) in the frame's pixels. `score` is the presence
probability. The result is `null` when the presence is not above the threshold. The
`DetectOptions` of the classic detector are ignored. Corners outside the frame are clamped onto
its border, and a non-convex or degenerate quad gives `null` (`clipToFrame: false` keeps the raw
quad; `infer()` always does). Like every `DocumentDetector`, the detector consumes (closes)
`ImageBitmap` frames and leaves RGBA frames unchanged. Frames of any size work: a side shorter
than 192 px (a 384×173 frame from a 20:9 stream) is upscaled as `cv2.resize(INTER_AREA)` does.

The package needs a `@scanoficus/core` that has the classic detector v3 (`CONFIDENT_SCORE`
0.80 and `DEFAULT_MIN_SCORE` 0.74, colour edges from RGBA input) and the optional
`DocumentDetector.detectStill` hook (used by the controller for stills), i.e. the core of this
repository from the same commit on. Core is 0.0.x and unpublished, so no semver range can
express that yet: the peer dependency is `>=0.0.0`. Publish both together; the first published
core version with these features should become the lower bound (e.g. `^0.1.0`).

## Round 2 and the corner refiner

```ts
import { createNnDetectorWorker } from '@scanoficus/detector-nn';
import { DOCQUADNET2 } from '@scanoficus/detector-nn/docquadnet2';

const scanner = new ScannerController({
  createDetector: () => createNnDetectorWorker({ bundle: DOCQUADNET2 }),
});
```

Round 2 is opt-in. On a private iPhone set (a thin white receipt on a white table) round 2 +
refiner reached an IoU of 0.72 against 0.81 for round 1 + edge refinement: the refiner pulled a
corner out to the stronger edge of the table. Round 1 (with refinement, or in the classic-first
hybrid) stays the default until round 2 does at least as well on real phone photos.

- **Pipeline** (a port of `scanoficus_ml/refiner.py` and the `net:` step of
  `scanoficus_ml/pipeline.py`): main network → raw quad → the frame area-downscaled to a long
  side of 1600 (one `cv2.resize(INTER_AREA)`; smaller frames are used as they are) → a pyramid
  of 2× box halvings → per pass and corner a 64×64 crop in the canonical corner frame (rotated
  so the bisector of the corner points to the crop's bottom right; side 0.4 × √area, then 0.2 ×
  √area), sampled bilinearly with clamp-to-edge from the pyramid level whose pixel is at most
  1.5 crop pixels → the refiner network (the 4 crops as one batch) → offset per corner, rotated
  back. 2 passes, then the presence decision (round 2: presence > 0.98).
- **Stills.** `detectStill(frame)` runs the refiner (by default). The `ScannerController`
  calls it with the full-resolution still for the re-detection after capture and for
  `addImage()` (core hook `DocumentDetector.detectStill`, additive and optional). Live
  `detect()` returns the raw network quad unless `refiner: { live: true }` (it costs two
  refiner passes per frame, see the speed table). `refiner: false` never runs it.
- **WebGL2.** One context runs both networks. The frame texture of the main network's run is
  reused: one area draw for the work image (a halving when the factor is exactly 2), halvings
  for the pyramid (built once per frame), one crop draw per corner into the refiner's input,
  then the refiner network with the 4 crops side by side in each texture (batch layout
  `N·W × H·⌈C/4⌉`), and one read-back of 8 values per pass.
- **Self-tests.** The refiner's self-test runs the 4 crops of a batch with different constants
  (0, 0.3, 0.7, 1), so mixed-up batch items are caught.
- **Crops match OpenCV.** OpenCV 5's `warpAffine(INTER_LINEAR, BORDER_REPLICATE)` works in
  float32 with fused multiply-adds and rounds half to even. The CPU path reproduces it
  bit-exactly (SHA-256 of all crops of the 12 test frames, both passes). The GPU shader has no
  `fma` (GLSL ES 3.00) and can differ by one level on rare pixels: of the 1,179,648 crop
  values of the test frames, none on Apple M5 (ANGLE Metal) and 22 under SwiftShader. The work image is bit-exact on the CPU; on the
  GPU its area filter is float32 as for the main network.
- **Presence decision.** `presence: (signals) => score` replaces the rule
  `presence > threshold`. It gets the presence logit and probability, the raw quad, the refined
  quad and the refiner's offsets per pass, the frame (still open; do not close it), whether it
  is a still, and the threshold. The frame reports a document if the returned score exceeds the
  threshold; the score becomes `Detection.score`. With the default rule, frames below the
  threshold skip the refiner; with a custom rule the refiner runs first. In a worker, pass
  `presenceModule: new URL('./my-rule.js', import.meta.url)` (an ES module whose default export
  is the rule), because functions cannot be posted. **The worker imports and executes that
  module.** It must be self-contained JavaScript (no bare imports, no TypeScript): Vite copies a
  `new URL('./rule.js', import.meta.url)` file as a raw asset without bundling it. By default
  only http(s) URLs of the worker's own origin are accepted; `allowCrossOriginPresenceModule:
  true` allows others (and `blob:` / `data:` URLs).
- **Concurrency.** A detector processes one frame at a time, in call order (the GPU holds one
  uploaded frame for the refiner); `GlRunner.runFrame()` returns a frame token and
  `refine(quad, settings, token)` refuses another frame (`FrameChangedError`).
- **Large stills.** Pixels are read once per frame. A bitmap above the GPU's texture limit (on
  the CPU path: above 2 × `workSide` = 3200 px) is drawn by the browser at a long side of
  3200 px first, so a 50 MP still is never read at full size; this path is not bit-exact with
  the Python pipeline (the browser's filter instead of one area resize). RGBA stills are used
  as they are.
- **Bundles.** `loadBundle(source)` loads a main network and an optional refiner and checks
  every weights file. `source` is a `BundleSource` (`DOCQUADNET`, `DOCQUADNET2`, or your own
  URLs and refiner settings) or the URL of a self-hosted `bundle.json` (in each
  `models/<bundle>/` folder). Pass the source or the loaded bundle as `bundle`.

## Round 3: the verified presence decision

```ts
import { createNnDetectorWorker } from '@scanoficus/detector-nn';
import { DOCQUADNET3 } from '@scanoficus/detector-nn/docquadnet3';

createDetector: () => createNnDetectorWorker({ bundle: DOCQUADNET3 }),
```

- **Bundle.** The round-2 main network (its files in `models/docquadnet2/` are shared, not
  copied), a refiner with a third output per crop, the corner-confidence logit
  (`docquadnet3-refiner`; offsets as round 2), and the decision of scanoficus-ml's
  `decision.json`. The bundle module carries the decision's runtime fields inline (no file is
  fetched for it); `models/docquadnet3/decision.json` is the copy for self-hosting.
- **Pinned.** The packaged decision records the SHA-256 of the main network's and the refiner's
  weights it was fitted with (`mainSha256`, `refinerSha256`, added by `package_model.py`).
  `loadBundle` rejects a decision whose hashes do not match the loaded weights, or that has
  none.
- **Decision** (`scanoficus_ml/detectors.py` `OwnPipeline3`, features of
  `scanoficus_ml/verify.py`): `score = bias + Σ weights[k] · features[k]`, a document if
  `score > threshold`, with the features `z` (main presence logit, clipped to ±12), `cl_min` /
  `cl_mean` (min / mean of the 4 pass-2 corner logits) and `cl1_min` (min of the pass-1
  logits). It is the bundle's built-in presence rule, built from the file at load time
  (`createDecision`): new coefficients or a new threshold are a file swap; a feature this
  runtime cannot compute is refused with an error naming it. `Detection.score` is
  `sigmoid(score)` and the bundle's threshold `sigmoid(threshold)` (0.222), as in Python. A
  `presence` option still replaces the rule. An explicit `threshold` is compared with
  `sigmoid(score)` (a warning says so).
- **The refiner runs on live frames too** (the decision needs its corner scores), so round 3
  costs about 7 ms more per live frame than round 1 on an M5 Pro, more on phones (see the
  speed table). Turning the refiner off (`refiner: false`, `live: false` or `still: false`)
  makes `ready()` throw for a decision bundle, unless a `presence` rule is given: without the
  corner scores the decision would reject every frame. Round 3
  starts the refiner from the network's normalized corners times the work-image size and maps
  the result back per axis, as `verify.run_pipeline` does (round 2: frame pixels times one
  scale, as `pipeline.py`); `RefinerSettings.coordinates` selects this.
- **Precision: prefer float32.** float32 render targets (the default where available) reproduce
  the NumPy score within 0.05 on clear pages (0.1 on faint near-threshold pages). With float16
  activations the corner logits move by up to 0.05 and the score shifts by an amount that
  depends on the GPU: float16 − float32 averaged −0.14 (sd 0.22) on Apple M5 / Metal and about
  −0.02 (sd 0.13) on SwiftShader (21 test scenes each), so no constant correction fits and
  float16 decisions near the threshold vary by GPU (on Metal 2 of the 8 near-threshold test
  scenes flipped). On scanoficus-ml's validation sets 0.3 % of the frames lie within 0.05 of
  the threshold, 1.0 % within 0.15, 2.2 % within 0.3 and 4.3 % within 0.66 (0–11 % by set);
  those are the frames whose float16 decision may differ. `PresenceSignals.precision` tells a
  custom rule which precision a frame ran with. `pnpm check:nn` prints the float16 shift of the
  GPU it runs on.

### Options (`createNnDetector`, `createNnDetectorWorker`)

| option                                               | default      | meaning                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bundle`                                             | `DOCQUADNET` | model bundle: a `BundleSource`, a `bundle.json` URL, or a loaded `NnBundle` (`createNnDetector` only). Its threshold is the default `threshold`.                                                                                                                                                                                          |
| `refiner`                                            | stills       | `{ live, still }`, `true` for both, `false` for neither; only with a bundle that has a refiner. Defaults: `{ live: false, still: true }`; for a bundle with a decision (round 3) `{ live: true, still: true }`, which costs about 7 ms per live frame on an M5 Pro (more on phones), and turning it off without a `presence` rule throws. |
| `presence`                                           | threshold    | presence rule `(signals) => score` (`createNnDetector`); `presenceModule` (URL of a module exporting it) for `createNnDetectorWorker`.                                                                                                                                                                                                    |
| `graphUrl`, `weightsUrl`                             | package      | the graph JSON and the weights. Default: the files in the package's `models/` (`DOCQUADNET_GRAPH_URL`, `DOCQUADNET_WEIGHTS_URL`). With only `graphUrl`, the weights file named in the graph is loaded from next to it. `createNnDetector` also accepts `model` (from `loadModel`).                                                        |
| `threshold`                                          | bundle       | presence threshold, chosen on validation: round 1 0.85 (on 785 real photos without a document, 8.8 % pass at 0.85, 7.1 % at 0.9 and 4.6 % at 0.95); round 2 0.98 (at most 5 % of the COCO calibration half; 4.6 % on the report half).                                                                                                    |
| `backend`                                            | `'auto'`     | `'auto'`: WebGL2, with the CPU when WebGL2 or float render targets are missing, when a run fails, and for frames during a context loss. `'webgl2'`: no fallback (errors reject). `'cpu'`: CPU only.                                                                                                                                       |
| `precision`                                          | `'auto'`     | storage of WebGL2 activations: `'float32'` if renderable (`EXT_color_buffer_float`), else `'float16'`. Arithmetic is always float32.                                                                                                                                                                                                      |
| `refine`                                             | `false`      | `true` or `RefineOptions`: network first, then classic edge refinement near the quad.                                                                                                                                                                                                                                                     |
| `canvas`                                             | new canvas   | canvas for the WebGL2 context (`createNnDetector` only). Default: an `OffscreenCanvas`, or a `<canvas>` where `OffscreenCanvas` has no WebGL2 (Safari 16.4–16.x).                                                                                                                                                                         |
| `clipToFrame`                                        | `true`       | clamp the quad to the frame; `null` for non-convex or degenerate quads.                                                                                                                                                                                                                                                                   |
| `workerUrl`, `worker`, `timeoutMs`, `transferPixels` | see the core | as in `createDetectorWorker` from `@scanoficus/core/worker`. The default worker URL is `new URL('./worker-entry.js', import.meta.url)`. `timeoutMs` defaults to 30 s, because the first request includes the model download.                                                                                                              |

### Hybrids

- **Network first, then edge refinement** (`refine: true`). This is a port of
  `scanoficus_ml/refine.py`. For each side of the predicted quad, the gradient across the side
  is profiled at 48 positions within ±0.8 % of the image diagonal. The window was tuned for
  DocQuadNet. A robust (Huber) line through the edge points replaces the side only if it stays
  close to and parallel with the side. Offline, on full-resolution frames downscaled to a long
  side of 1024 px, this raised the held-out SmartDoc JI from 0.950 to 0.973 and MIDV-2019 from
  0.793 to 0.813. The `ScannerController` gives detectors 384-px frames by default (also for the
  re-detection on a captured still), where the window is coarser and the gain was not
  measured: set `detection.maxSide` to 768–1024 when you use refinement. Gradients are computed
  only where the profiles sample. Cost per frame: +1.5 ms for 384-px frames, +11 ms for
  1280×720 RGBA frames (most of it is the full-frame gray conversion and downscale).
- **Classic first, network on doubt** (`createHybridDetector`; the demo's default). Offline
  (scanoficus-datasets `results/hybrid.md`), classic v3 with `CONFIDENT_SCORE` 0.80, else round 1
  - edge refinement (presence > 0.85), had the best JI on every set with documents (SmartDoc
    held-out 0.976, MIDV-500 0.872, MIDV-2019 0.844, Dizaj test 0.705, RWMD 0.654), at 10.7 %
    false positives on photos without a document (network alone 7.4 %). The thresholds are
    options for tuning: `confidentScore` (default `HYBRID_DEFAULTS.confidentScore`), the network
    detector's `threshold` (`HYBRID_DEFAULTS.networkThreshold`), `fallbackMinScore` (classic
    answer when the network fails) and `classicOptions` (e.g. `{ maxSide: 384 }` to keep the
    classic stage at its calibrated size when the frames are larger for refinement):

  ```ts
  import { createDetectorWorker } from '@scanoficus/core/worker';
  import {
    createHybridDetector,
    createNnDetectorWorker,
    HYBRID_DEFAULTS,
  } from '@scanoficus/detector-nn';

  // The benchmarked setup (as in the demo): round 1 with refinement on 768-px frames, the
  // classic stage at its calibrated 384 px.
  const scanner = new ScannerController({
    detection: { maxSide: 768 },
    createDetector: () =>
      createHybridDetector({
        classic: createDetectorWorker(),
        classicOptions: { maxSide: 384 },
        confidentScore: HYBRID_DEFAULTS.confidentScore,
        nn: createNnDetectorWorker({ refine: true, threshold: HYBRID_DEFAULTS.networkThreshold }),
        onError: (error) => console.warn(error), // network failures; the classic answer is used
      }),
  });
  ```

  Each frame first goes to the classic detector (`minScore: 0`), RGBA frames unchanged so that
  it uses its colour edges. A quad with a score of at least `CONFIDENT_SCORE` (0.80 in the core
  v3: all such quads on the SmartDoc tuning split and 95 % on the MIDV-500 tuning half are
  right, and about 3 % of photos without a document reach it) is returned as is. Frames
  with a lower score go to the network, which then decides alone. The `score` of the result is
  the classic edge score or the network's presence probability, depending on which detector
  answered. If the network fails on a frame, the classic quad answers if it reaches
  `DEFAULT_MIN_SCORE`, else `null`, and `onError` is called.

## Hosting the weights

By default the models come from the package, one folder per bundle:

| folder                | files                                                                                                                                                    | raw / gzip       |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `models/docquadnet/`  | `docquadnet.json`, `docquadnet-9bc8c22c6a4c.bin`, `bundle.json`                                                                                          | 702 kB / 645 kB  |
| `models/docquadnet2/` | `docquadnet2.json`, `docquadnet2-6ea1c21d6f87.bin`, `docquadnet2-refiner.json`, `docquadnet2-refiner-ed145db94b95.bin`, `bundle.json`                    | 955 kB / 874 kB  |
| `models/docquadnet3/` | `docquadnet3-refiner.json`, `docquadnet3-refiner-216d024894ca.bin`, `decision.json`, `bundle.json`, `coco-credits.csv` (main network: `../docquadnet2/`) | 1245 kB / 548 kB |

They are referenced as `new URL('../models/…', import.meta.url)`. Vite, webpack 5 and other
bundlers that understand this pattern emit the files as content-hashed assets, so caches
(including a cache-first service worker) can keep them forever. Round 2 lives in its own entry
(`@scanoficus/detector-nn/docquadnet2`), so apps that do not import it do not emit its files.
Keep `models/LICENSE.md` with them (for example, link it from your app's credits).

To host the files yourself, copy a bundle folder and load its `bundle.json`
(`bundle: 'https://…/docquadnet2/bundle.json'`; `docquadnet3/bundle.json` refers to the main
network in `../docquadnet2/`, so copy both folders side by side), or pass `graphUrl` for a main network only (the
weights are loaded from the file name in the graph, relative to it). The manifest and the
graphs are fetched with `cache: 'no-cache'`; the weights' length and SHA-256 are checked against
their graph (with `crypto.subtle`, which exists only in secure contexts; elsewhere only the
length is checked), and a mismatched pair is rejected. Precompress the files (the weights shrink
by about 7 % with gzip and 12 % with brotli; the graphs to 1.5 kB). They are fetched once per
detector: share one bundle from `loadBundle()` between `createNnDetector` instances. After
retraining, regenerate `models/` with `scripts/package_model.py`.

**Package size.** Round 2 grows the tarball by 0.87 MB (gzip; 0.96 MB unpacked) compared with
round 1 alone, round 3 by another 0.55 MB (gzip; 1.25 MB unpacked, of which the COCO
image list is 0.32 / 0.99 MB). When round 2 becomes the default, drop `models/docquadnet/` from
the package (`package_model.py` builds each bundle independently) and keep round 1 only where
it is still needed, for example in the demo's `public/` folder, loaded with
`bundle: '…/bundle.json'`.

## How it works

- **Graph.** `docquadnet.json` describes 44 nodes, with weights as offsets into the fp16 blob.
  It contains 34 convolutions (1 dense 3×3 stride 2, 22 pointwise, 11 depthwise), ReLU6, 7
  residual adds, a flatten and 2 linear layers. `parseModel` lowers it into 36 steps. Each
  linear layer becomes a convolution whose kernel covers the whole input, because PyTorch's
  C-major flatten matches the weight layout. Each residual add is fused into the preceding
  projection.
- **WebGL2.** A `C×H×W` tensor is one texture of `W × H·⌈C/4⌉` texels, with 4 channels per
  texel. Each step is one full-screen draw with a fragment shader. There are 5 programs:
  dense convolution, depthwise convolution, elementwise, 2× box halving and resize. The
  weights are RGBA16F textures that hold the exact stored fp16 values. Activations are float32
  (or float16) textures, reused by a static liveness plan. Only the 9 outputs are read back.
- **Self-test.** After compiling and uploading, the runtime checks `getError()`, the
  completeness of every framebuffer and one inference on a constant input against the outputs
  stored in the graph (`selfTest`, from the NumPy reference). Each new frame size is checked for
  GL errors once. Any failure throws (the `'auto'` backend then uses the CPU and reports
  `glError`); failed shader links carry both compile logs. A failed constructor releases
  everything it created, including the context.
- **Preprocessing.** The whole frame is stretched to 192×192 (no letterbox). The CPU path
  reproduces `scanoficus_ml.data.preprocess` **bit-exactly** on the 10 test frames: 2×2 box
  halvings while both sides are even and at least 384 px, then `cv2.resize(INTER_AREA)`: area
  weights in float32 with OpenCV's rounding when both axes shrink (half up for 2×2, half to
  even otherwise, e.g. 4:1), and OpenCV's 11-bit fixed-point bilinear emulation when an axis
  grows. The WebGL2 path does the halvings and the upscaling exactly (integer arithmetic in the
  shader, the same coefficient tables) and the area filter in float32 on 8-bit levels: 263 of
  1,105,920 input values differ by one level from Python. For refinement, `ImageBitmap` frames
  are drawn to the refinement size by the browser (`imageSmoothingQuality: 'high'`), not by an
  area filter.
- **Context loss.** While the context is lost, `'auto'` answers from the CPU. After the
  browser restores the context, the next frame rebuilds the programs, weights and textures
  from the model kept in memory. `dispose()` deletes every GPU object and releases the
  context.

## Parity and speed

**Round 3** (`test/round3.test.ts`, `pnpm check:nn`; fixtures from
`scripts/make_round3_fixtures.py`: `verify.run_pipeline` + `features` + the decision with the
NumPy interpreter as main network and refiner, and `OwnPipeline3` itself with the fp32 ONNX
exports, on the 12 scenes of the round-2 fixtures, 8 faint near-threshold blends and a white page on a light table):

| path (clear pages unless noted)                                 | Apple M5 / Metal    | SwiftShader (CI)    |
| --------------------------------------------------------------- | ------------------- | ------------------- |
| CPU: crops from Python's pass inputs                            | bit-exact (SHA-256) | bit-exact           |
| WebGL2 crops: values differing by one level from the CPU        | 0 of 2,064,384      | 24                  |
| WebGL2 float32 network on the CPU crops: offsets / corner logit | 1.3·10⁻⁷ / 8.1·10⁻⁶ | 1.0·10⁻⁷ / 6.2·10⁻⁶ |
| WebGL2 float32 end to end (own crops): offsets / corner logit   | 1.9·10⁻⁷ / 6.2·10⁻⁶ | 7.0·10⁻⁶ / 1.1·10⁻³ |
| WebGL2 float32 score vs NumPy: clear / faint pages              | 0.011 / 0.042       | 0.005 / 0.061       |
| WebGL2 float16 score vs NumPy: clear / faint pages              | 0.26 / 0.66         | 0.04 / 0.30         |
| float16 − float32 score: mean (sd)                              | −0.14 (0.22)        | −0.02 (0.13)        |
| CPU end to end: score vs NumPy, clear / faint pages             | 0.020 / 0.040       | (same)              |
| accept / reject where the margin exceeds the tolerance          | all equal to NumPy  | all equal to NumPy  |

Tolerances in `check:nn` (float32): the network on identical crops 10⁻⁴; end to end offsets
10⁻³ and corner logit 4·10⁻³ (GLSL ES 3.00 has no fma, so a GPU may round a few crop pixels
differently from OpenCV, and the corner logit reacts to single pixels; 3.5× SwiftShader's
1.1·10⁻³); score 0.05 on clear and 0.1 on faint pages.

End to end, the main network's 10⁻⁶ differences move the crops by about 10⁻³ px, which flips
a few 8-bit crop levels and moves the corner logits by up to a few hundredths; from identical
crops they agree to 10⁻⁵. All test scenes score far from the threshold (−1.25; documents +2.1
to +8.1, no-document frames −8.3 to −8.8), so the decisions near the threshold are not
covered by these fixtures.

Live cost of round 3 (main network + 2 refiner passes + decision on every frame; same machine
and run as the table below, WebGL2 float32, warm p50 / p95 ms):

| frame      | main thread | worker      |
| ---------- | ----------- | ----------- |
| 384 × 216  | 9.5 / 12.4  | 10.7 / 13.3 |
| 768 × 432  | 10.2 / 13.1 | 11.7 / 14.8 |
| 1280 × 720 | 11.1 / 15.5 | 12.9 / 18.0 |

For comparison in the same run: round 1 alone 1.9 (384) to 5.5 ms, round 1 + edge refinement
4.9 (384) to 13.9 ms (1280), round 2 + live refiner 9.1–12.0 ms. The hybrid default sends only
frames with a classic score below 0.80 to the network (35–69 % of the frames offline).

**Round 2 + refiner**
(`test/refiner.test.ts`, `test/pipeline.test.ts`, `pnpm check:nn`;
fixtures from `scripts/make_refiner_fixtures.py`, which runs `pipeline.py` with the NumPy
refiner and with the exported ONNX refiner on the 10 test scenes plus 2000×1500 and
3200×1800 frames):

| path                                                 | result                                                |
| ---------------------------------------------------- | ----------------------------------------------------- |
| CPU: work image and all crops vs Python              | bit-exact (SHA-256, 12 frames × 2 passes)             |
| CPU: refiner offsets (Python's pass inputs)          | 5.6·10⁻⁶ half-crop units                              |
| CPU: refined quad from Python's raw quad (documents) | 3.9·10⁻⁴ px (vs the ONNX pipeline: 0.035 px)          |
| WebGL2 float32: round-2 main network, end to end     | corners 1.1·10⁻⁵, logit 7.4·10⁻⁴                      |
| WebGL2 float32 / float16: crops vs the CPU crops     | 0 of 1,179,648 values differ                          |
| WebGL2 float32 / float16: refiner offsets            | 1.1·10⁻⁷ / 3.4·10⁻⁴ half-crop units                   |
| WebGL2 float32 / float16: refined quad (documents)   | < 10⁻⁴ / 0.072 px (vs ONNX pipeline 0.035 / 0.073 px) |
| detectStill / detect on both threads, 24 runs        | stills within 0.018 px of pipeline.py, live = raw     |

Without a document (empty, noise frames) the refiner chases texture: tiny offset differences
of pass 1 change the pass-2 crops, so those frames get a looser tolerance (0.25 px on the CPU);
the presence threshold rejects them anyway.

**Round 1:**

Checked with `pnpm check:nn` (headless Chrome; `pnpm check:nn:swiftshader` forces the
SwiftShader software renderer that CI uses). The
reference is the NumPy interpreter `scanoficus_ml.rawgraph.run_graph`. The 10 synthetic test
frames (160×120 to 1280×720, including 384×173 and 173×384 from 20:9 streams, 768×192, an
empty and a noise frame) are generated bit-identically in Python and TypeScript, so the
fixtures (`test/fixtures/cases.json`) hold only hashes and expected outputs. Corners are
normalized coordinates; the presence logit is checked separately (its probability changes far
less). Two tiny random-weight graphs (`test/fixtures/synthetic/`, from
`scripts/make_synthetic.py`) cover what DocQuadNet does not: non-square and odd sizes with
stride 2, channel counts that are not a multiple of 4, a 2-channel input, an add that cannot
be fused, hswish, a convolution without padding and a sigmoid node.

| path                                                   |                     corners (max abs) |   logit (max abs) | tolerance (corners / logit) |
| ------------------------------------------------------ | ------------------------------------: | ----------------: | --------------------------- |
| CPU (Vitest), all 9 outputs                            |                              4.8·10⁻⁶ |          4.8·10⁻⁶ | 1e-3                        |
| WebGL2 float32, network only                           |                              1.8·10⁻⁷ |          5.3·10⁻⁶ | 1e-5 / 1e-4                 |
| WebGL2 float32, end to end with GPU preprocessing      |                              1.7·10⁻⁵ |          5.1·10⁻⁴ | 1e-4 / 5e-3                 |
| WebGL2 float16 activations, end to end                 |                              6.8·10⁻⁴ |          4.2·10⁻² | 2e-3 / 1e-1                 |
| synthetic graphs (relative): WebGL2 float32 / float16  |                   4.2·10⁻⁷ / 1.9·10⁻³ |                   | 1e-4 / 1e-2 (CPU: 1e-5)     |
| detect() on main thread and in a worker, 20 frame runs | within 0.023 px of the reference quad | presence 2.4·10⁻⁷ |                             |
| `refineQuad` vs `refine.py`, where refine.py converged |                        within 0.28 px |                   | 0.5 px                      |

Speed in headless Chrome on an Apple M5 Pro (ANGLE Metal; `pnpm bench:nn`). A frame is
1280×720 unless noted. "384" is a 384×216 frame, which is what the `ScannerController` gives
detectors by default; "12 MP" is a 4000×3000 still (`ImageBitmap`) through `detectStill`. Each
warm figure is the median and 95th percentile over 100 frames (20 stills, 3 for the CPU
still). Cold start = model fetch from localhost, parse and SHA-256 check / shader compilation,
weight upload and self-tests / first frame:

| thread | backend | model                    | input       | cold start ms      | warm p50 / p95 ms |
| ------ | ------- | ------------------------ | ----------- | ------------------ | ----------------- |
| main   | WebGL2  | round 1                  | RGBA        | 61 (13 / 35 / 13)  | 5.5 / 7.6         |
| main   | WebGL2  | round 1                  | ImageBitmap | 53 (9 / 32 / 10)   | 6.3 / 8.8         |
| main   | WebGL2  | round 1, float16         | RGBA        | 58 (12 / 33 / 12)  | 5.3 / 7.4         |
| main   | WebGL2  | round 1 + edge refine    | RGBA        | 75                 | 16.6 / 21.4       |
| main   | WebGL2  | round 1                  | RGBA 384    | 54                 | 4.4 / 7.1         |
| main   | WebGL2  | round 1 + edge refine    | RGBA 384    | 52                 | 5.5 / 8.7         |
| main   | CPU     | round 1                  | RGBA        | 216                | 126 / 147         |
| worker | WebGL2  | round 1                  | ImageBitmap | 62                 | 7.0 / 9.4         |
| worker | WebGL2  | round 1                  | RGBA 384    | 71                 | 3.9 / 6.3         |
| worker | CPU     | round 1                  | ImageBitmap | 222                | 137 / 210         |
| main   | WebGL2  | round 2 (live, raw)      | RGBA        | 63 (7 / 45 / 11)   | 3.5 / 5.7         |
| main   | WebGL2  | round 2 + refiner (live) | RGBA        | 56 (9 / 34 / 13)   | 7.8 / 12.4        |
| main   | WebGL2  | round 2 + refiner (live) | RGBA 384    | 55                 | 8.0 / 10.6        |
| worker | WebGL2  | round 2 + refiner (live) | RGBA        | 97                 | 13.5 / 18.7       |
| main   | WebGL2  | round 2 + refiner        | 12 MP still | 109 (16 / 47 / 45) | 14.6 / 16.9       |
| worker | WebGL2  | round 2 + refiner        | 12 MP still | 132                | 13.2 / 17.1       |
| main   | CPU     | round 2 + refiner        | 12 MP still | 924                | 719 / 730         |

The machine was busy during this run (the round-1 CPU path took 126 ms per frame here and
84–106 ms in other runs), so compare rows within one table only. The refiner adds 2 passes of
4 crop draws, 25 network draws and one read-back to a frame: about 4 ms here. A 12 MP still
adds the texture upload of the bitmap and the 1600-px work image. A synchronous read-back round
trip is part of every WebGL2 frame; a worker adds the message round trip (and, for RGBA frames,
a copy). Phones are not measured yet.

## Quality

These are offline results of the Python pipeline from scanoficus-ml, held-out data, official
SmartDoc Jaccard index. This package reproduces that pipeline's outputs (see above).

| detector                | SmartDoc held-out | bg 5 (clutter) | MIDV-500 | MIDV-2019 | photos without a document with a quad |
| ----------------------- | ----------------: | -------------: | -------: | --------: | ------------------------------------: |
| classic (core v3)       |             0.849 |          0.000 |   0.710¹ |     0.655 |                                 6.9 % |
| DocQuadNet              |             0.950 |          0.884 |    0.792 |     0.793 |                                 8.8 % |
| DocQuadNet + refinement |             0.973 |          0.887 |    0.808 |     0.813 |                                 8.8 % |

¹ The classic v3 MIDV-500 figure is on the held-out half (5 of the 10 document types); the
DocQuadNet figures are on all 10 types, none of which it was trained on. The DocQuadNet rows
were measured against classic v2, before v3 existed; they were not re-run.

On clean, high-contrast scenes the classic detector is still more precise. The classic-first
hybrid uses it there.

## License of the weights

The code is BSD-3-Clause (`LICENSE`). The weights (`models/*/*.json`, `models/*/*.bin`) were
trained by us
from scratch (no pretrained weights) and are also released under BSD-3-Clause. The training
data includes frames of the SmartDoc 2015 dataset, whose license requires attribution. Keep
this notice with the weights:

> SmartDoc 2015 Ch. 1 **tuning split only** (datasheet, magazine, patent documents 001–004 on
> all 5 backgrounds; every 3rd frame = 3,392 frames, downscaled to 640×360) — CC BY 4.0 —
> © Chazalon et al., ICDAR 2015 SmartDoc Challenge 1
> (github.com/jchazalon/smartdoc15-ch1-dataset); attribution must be kept with the weights.
> License: https://creativecommons.org/licenses/by/4.0/

Rounds 2 and 3 (`models/docquadnet2/`, `models/docquadnet3/`) were also trained on the training
part of the Dizaj photos, which are CC BY 4.0 as well; keep this notice with those weights:

> "A New Image Dataset for Document Corner Localization" by Shima Dizaj, Mendeley Data v3,
> https://doi.org/10.17632/x3nm4cxr83.3, licensed under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/). Changes: photos resized, cropped, rotated and
> augmented for training; labels converted. Used: the training part (524 photos).
>
> Note (Mendeley Data): the dataset's license does not cover content shown in the photos, which
> can belong to third parties.

Round 3 (`models/docquadnet3/`: its refiner and decision; the main network is round 2's) also
used COCO photos; keep this notice and `models/docquadnet3/coco-credits.csv` (the image list)
with those files:

> Images from COCO train2017 (Lin et al., "Microsoft COCO: Common Objects in Context", ECCV
> 2014; https://cocodataset.org; annotations CC BY 4.0, used to select the images), originally
> from Flickr under CC BY 2.0 (https://creativecommons.org/licenses/by/2.0/) or "no known
> copyright restrictions" (https://flickr.com/commons/usage/): 6,329 photos without documents;
> 3,921 trained the round-3 refiner (corner crops), 1,637 fitted the presence decision and 771
> were used for validation. Changes: cropped (corner crops), resized, rotated and augmented.
> `docquadnet3/coco-credits.csv` lists each image's COCO id, role, license, Flickr photo page
> (https://flic.kr/p/…) and, for the 6,223 CC BY 2.0 photos, the author and title as Flickr
> shows them today (oEmbed, fetched November 2026): 5,150 found, 1,073 no longer on Flickr
> (author and title unknown; the COCO source URL is listed). 106 photos with "no known copyright
> restrictions" need no author. Some owners have changed the license on Flickr since COCO
> collected the photos (column `flickr_license_now`); the license under which COCO obtained
> them (column `license`) applies to the copies used here.

The other training inputs need no attribution: procedural pages, CC0 textures and HDRIs from
Poly Haven, and OFL fonts that were only rendered into images. The data manifests are in the
scanoficus-ml repository. `models/LICENSE.md` carries this notice next to the files.

## Development

```sh
pnpm --filter @scanoficus/detector-nn test            # CPU parity, preprocessing, refinement, hybrid
pnpm build && pnpm check:nn                           # WebGL2 parity and robustness in headless Chrome
pnpm check:nn:swiftshader                             # the same on SwiftShader (as on CI)
pnpm bench:nn                                         # also cold start and warm timings
# regenerate models/ and the fixtures (needs the scanoficus-ml checkout and its virtualenv):
PY="env PYTHONDONTWRITEBYTECODE=1 $HOME/src/scanoficus-ml/.venv/bin/python"
$PY packages/detector-nn/scripts/package_model.py
$PY packages/detector-nn/scripts/make_fixtures.py
$PY packages/detector-nn/scripts/make_synthetic.py
$PY packages/detector-nn/scripts/make_refiner_fixtures.py
```
