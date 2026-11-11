# @scanoficus/detector-nn

An optional neural document detector for scanoficus. It runs **DocQuadNet** (our own model,
349k parameters, 192×192 RGB input, 81 M multiply-adds) on a small WebGL2 runtime written for
this graph, with a plain TypeScript CPU fallback. It plugs into the scanner as a core
`DocumentDetector`.

- The network predicts 4 corners (also outside the frame) and a document-presence probability.
  It handles cluttered backgrounds, white-on-white, hand-held and partly visible pages, where the
  classic edge detector fails (see "Quality" below).
- The JavaScript is 12.7 KB min+gzip without the weights. The weights (fp16, 688 kB raw,
  608 kB brotli) ship in the package (`models/`) and are fetched at runtime; bundlers emit
  them as content-hashed assets. The graph names its weights file, length and SHA-256, so a
  graph and weights from different versions never run together.
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

The package needs `@scanoficus/core` with the classic detector v3 (colour edges from RGBA input,
`CONFIDENT_SCORE` 0.80); the peer dependency range is open (`>=0.0.0`) while all packages are
0.0.x.

### Options (`createNnDetector`, `createNnDetectorWorker`)

| option                                               | default      | meaning                                                                                                                                                                                                                                                                            |
| ---------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `graphUrl`, `weightsUrl`                             | package      | the graph JSON and the weights. Default: the files in the package's `models/` (`DOCQUADNET_GRAPH_URL`, `DOCQUADNET_WEIGHTS_URL`). With only `graphUrl`, the weights file named in the graph is loaded from next to it. `createNnDetector` also accepts `model` (from `loadModel`). |
| `threshold`                                          | `0.85`       | presence threshold, chosen on validation. On 785 real photos without a document, 8.8 % pass at 0.85, 7.1 % at 0.9 and 4.6 % at 0.95.                                                                                                                                               |
| `backend`                                            | `'auto'`     | `'auto'`: WebGL2, with the CPU when WebGL2 or float render targets are missing, when a run fails, and for frames during a context loss. `'webgl2'`: no fallback (errors reject). `'cpu'`: CPU only.                                                                                |
| `precision`                                          | `'auto'`     | storage of WebGL2 activations: `'float32'` if renderable (`EXT_color_buffer_float`), else `'float16'`. Arithmetic is always float32.                                                                                                                                               |
| `refine`                                             | `false`      | `true` or `RefineOptions`: network first, then classic edge refinement near the quad.                                                                                                                                                                                              |
| `canvas`                                             | new canvas   | canvas for the WebGL2 context (`createNnDetector` only). Default: an `OffscreenCanvas`, or a `<canvas>` where `OffscreenCanvas` has no WebGL2 (Safari 16.4–16.x).                                                                                                                  |
| `clipToFrame`                                        | `true`       | clamp the quad to the frame; `null` for non-convex or degenerate quads.                                                                                                                                                                                                            |
| `workerUrl`, `worker`, `timeoutMs`, `transferPixels` | see the core | as in `createDetectorWorker` from `@scanoficus/core/worker`. The default worker URL is `new URL('./worker-entry.js', import.meta.url)`. `timeoutMs` defaults to 30 s, because the first request includes the model download.                                                       |

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
- **Classic first, network on doubt** (`createHybridDetector`):

  ```ts
  import { createDetectorWorker } from '@scanoficus/core/worker';
  import { createHybridDetector, createNnDetectorWorker } from '@scanoficus/detector-nn';

  createDetector: () =>
    createHybridDetector({
      classic: createDetectorWorker(),
      nn: createNnDetectorWorker(),
      onError: (error) => console.warn(error), // network failures; the classic answer is used
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

By default the model comes from the package: `models/docquadnet.json` and
`models/docquadnet-<sha256 prefix>.bin`, referenced as `new URL('../models/…', import.meta.url)`.
Vite, webpack 5 and other bundlers that understand this pattern emit both as content-hashed
assets, so caches (including a cache-first service worker) can keep them forever. Keep
`models/LICENSE.md` with them (for example, link it from your app's credits).

To host the files yourself, copy the three files from `models/` and pass `graphUrl` (the
weights are loaded from the file name in the graph, relative to it) or both URLs. `loadModel`
fetches the graph with `cache: 'no-cache'` and checks the weights' length and SHA-256 against
the graph (with `crypto.subtle`, which exists only in secure contexts; elsewhere only the length
is checked) and rejects a mismatched pair. Precompress the files (the weights shrink to 608 kB
with brotli or 642 kB with gzip; the graph to 1.4 kB). They are fetched once per detector:
share one `model` from `loadModel()` between `createNnDetector` instances. After retraining,
regenerate `models/` with `scripts/package_model.py`.

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

Checked with `pnpm check:nn` (headless Chrome; it also passes with SwiftShader, as on CI). The
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
detectors by default. Each warm figure is the median and 95th percentile over 100 frames
(20 for the CPU). Cold start = model fetch from localhost, parse and SHA-256 check / shader
compilation, weight upload and self-test / first frame:

| thread | backend | input             | cold start ms            | warm p50 / p95 ms |
| ------ | ------- | ----------------- | ------------------------ | ----------------- |
| main   | WebGL2  | RGBA              | 44 (13 / 23 / 8)         | 1.9 / 5.3         |
| main   | WebGL2  | ImageBitmap       | 40 (8 / 23 / 7)          | 4.1 / 5.9         |
| main   | WebGL2  | RGBA, float16     | 37 (7 / 22 / 7)          | 2.6 / 3.9         |
| main   | WebGL2  | RGBA 384          | 34 (7 / 22 / 5)          | 1.9 / 2.2         |
| main   | WebGL2  | RGBA 384 + refine | 43                       | 3.4 / 4.3         |
| main   | WebGL2  | RGBA + refine     | 53                       | 13.4 / 15.4       |
| main   | CPU     | RGBA              | 167 (7 / 0 / 160)        | 106 / 127         |
| worker | WebGL2  | ImageBitmap       | 43 (to the first result) | 5.0 / 6.0         |
| worker | WebGL2  | RGBA (copied)     | 43                       | 5.2 / 7.5         |
| worker | WebGL2  | RGBA 384          | 42                       | 3.6 / 4.0         |
| worker | CPU     | ImageBitmap       | 180                      | 109 / 147         |

The machine was less loaded than for the first measurement of this package (the CPU path then
took 160 ms per frame instead of 106 ms, WebGL2 5.6 instead of 1.9 ms), so compare rows within
one table only. A synchronous read-back round trip is part of every WebGL2 frame; a worker adds
the message round trip. Phones are not measured yet.

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

The code is BSD-3-Clause (`LICENSE`). The weights (`models/docquadnet.json`,
`models/docquadnet-*.bin`) were trained by us
from scratch (no pretrained weights) and are also released under BSD-3-Clause. The training
data includes frames of the SmartDoc 2015 dataset, whose license requires attribution. Keep
this notice with the weights:

> SmartDoc 2015 Ch. 1 **tuning split only** (datasheet, magazine, patent documents 001–004 on
> all 5 backgrounds; every 3rd frame = 3,392 frames, downscaled to 640×360) — CC BY 4.0 —
> © Chazalon et al., ICDAR 2015 SmartDoc Challenge 1
> (github.com/jchazalon/smartdoc15-ch1-dataset); attribution must be kept with the weights.
> License: https://creativecommons.org/licenses/by/4.0/

The other training inputs need no attribution: procedural pages, CC0 textures and HDRIs from
Poly Haven, and OFL fonts that were only rendered into images. The data manifests are in the
scanoficus-ml repository. `models/LICENSE.md` carries this notice next to the files.

## Development

```sh
pnpm --filter @scanoficus/detector-nn test            # CPU parity, preprocessing, refinement, hybrid
pnpm build && pnpm check:nn                           # WebGL2 parity and robustness in headless Chrome
pnpm bench:nn                                         # also cold start and warm timings
# regenerate models/ and the fixtures (needs the scanoficus-ml checkout and its virtualenv):
PY="env PYTHONDONTWRITEBYTECODE=1 $HOME/src/scanoficus-ml/.venv/bin/python"
$PY packages/detector-nn/scripts/package_model.py
$PY packages/detector-nn/scripts/make_fixtures.py
$PY packages/detector-nn/scripts/make_synthetic.py
```
