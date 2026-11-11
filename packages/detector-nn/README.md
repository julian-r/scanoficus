# @scanoficus/detector-nn

An optional neural document detector for scanoficus. It runs **DocQuadNet** (our own model,
349k parameters, 192×192 RGB input, 81 M multiply-adds) on a small WebGL2 runtime written for
this graph, with a plain TypeScript CPU fallback. It plugs into the scanner as a core
`DocumentDetector`.

- The network predicts 4 corners (also outside the frame) and a document-presence probability.
  It handles cluttered backgrounds, white-on-white, hand-held and partly visible pages, where the
  classic edge detector fails (see "Quality" below).
- The JavaScript is 10.5 KB min+gzip without the weights. The weights (fp16, 688 kB raw,
  608 kB brotli) are fetched at runtime from URLs that you host.
- It runs on the main thread and in a Web Worker (`OffscreenCanvas`). It survives WebGL
  context loss and frees all GPU resources on `dispose()`.
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
    createNnDetectorWorker({
      graphUrl: '/models/docquadnet/docquadnet.json',
      weightsUrl: '/models/docquadnet/docquadnet.bin',
      refine: true, // network first, then classic edge refinement
    }),
});
```

With the custom elements, set the same option on the element:
`document.querySelector('scan-document').options = { createDetector: … }`.

On the calling thread (the main thread, or your own worker):

```ts
import { createNnDetector } from '@scanoficus/detector-nn';

const detector = createNnDetector({ graphUrl, weightsUrl });
await detector.ready(); // optional: load and compile ahead of the first frame (the cold start)
const detection = await detector.detect(imageDataOrBitmap); // { quad, score } or null
const raw = await detector.infer(frame); // { quad, presence, heads }, no threshold, no refinement
detector.dispose();
```

`detect()` returns the quad (TL, TR, BR, BL) in the frame's pixels. `score` is the presence
probability. The result is `null` when the presence is not above the threshold. The
`DetectOptions` of the classic detector are ignored. Like every `DocumentDetector`, the
detector consumes (closes) `ImageBitmap` frames and leaves RGBA frames unchanged. Frames must
be at least 192×192 pixels.

### Options (`createNnDetector`, `createNnDetectorWorker`)

| option                                               | default      | meaning                                                                                                                                                                                                                      |
| ---------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `graphUrl`, `weightsUrl`                             | –            | where to fetch `docquadnet.json` and `docquadnet.bin` (relative URLs resolve against the page). `createNnDetector` also accepts `model` (from `loadModel`).                                                                  |
| `threshold`                                          | `0.85`       | presence threshold, chosen on validation. On 785 real photos without a document, 8.8 % pass at 0.85, 7.1 % at 0.9 and 4.6 % at 0.95.                                                                                         |
| `backend`                                            | `'auto'`     | `'auto'`: WebGL2, with the CPU when WebGL2 or float render targets are missing, when a run fails, and for frames during a context loss. `'webgl2'`: no fallback (errors reject). `'cpu'`: CPU only.                          |
| `precision`                                          | `'auto'`     | storage of WebGL2 activations: `'float32'` if renderable (`EXT_color_buffer_float`), else `'float16'`. Arithmetic is always float32.                                                                                         |
| `refine`                                             | `false`      | `true` or `RefineOptions`: network first, then classic edge refinement near the quad.                                                                                                                                        |
| `canvas`                                             | new canvas   | canvas for the WebGL2 context (`createNnDetector` only).                                                                                                                                                                     |
| `workerUrl`, `worker`, `timeoutMs`, `transferPixels` | see the core | as in `createDetectorWorker` from `@scanoficus/core/worker`. The default worker URL is `new URL('./worker-entry.js', import.meta.url)`. `timeoutMs` defaults to 30 s, because the first request includes the model download. |

### Hybrids

- **Network first, then edge refinement** (`refine: true`). This is a port of
  `scanoficus_ml/refine.py`. For each side of the predicted quad, the gradient across the side
  is profiled at 48 positions within ±0.8 % of the image diagonal. The window was tuned for
  DocQuadNet. A robust (Huber) line through the edge points replaces the side only if it stays
  close to and parallel with the side. Offline, this raised the held-out SmartDoc JI from 0.950
  to 0.973 and MIDV-2019 from 0.793 to 0.813. Gradients are computed only where the profiles
  sample, at most at a long side of 1024 px. Cost per frame: +1.5 ms for the controller's
  384-px frames, +14 ms for 1280×720 RGBA frames (most of it is the full-frame gray conversion
  and downscale).
- **Classic first, network on doubt** (`createHybridDetector`):

  ```ts
  import { createDetectorWorker } from '@scanoficus/core/worker';
  import { createHybridDetector, createNnDetectorWorker } from '@scanoficus/detector-nn';

  createDetector: () =>
    createHybridDetector({
      classic: createDetectorWorker(),
      nn: createNnDetectorWorker({ graphUrl, weightsUrl }),
    });
  ```

  Each frame first goes to the classic detector (`minScore: 0`). A quad with a score of at
  least `CONFIDENT_SCORE` (0.85) is returned as is: 99.9 % of such quads are right on the
  SmartDoc tuning split, and about 2 % of photos without a document reach this score. Frames
  with a lower score go to the network, which then decides alone. The `score` of the result is
  the classic edge score or the network's presence probability, depending on which detector
  answered.

## Hosting the weights

Copy `docquadnet.json` and `docquadnet.bin` (the files are in `apps/demo/public/models/docquadnet/`
in this repository, or in `export/docquadnet/` of the scanoficus-ml repository). Keep the
`LICENSE.md` file next to them. Serve both files from your origin, or with CORS. Precompress
them (the weights shrink to 608 kB with brotli or 642 kB with gzip; the graph to 1.3 kB) and
cache them for a long time, for example with a versioned path, or by letting the PWA's service
worker cache them. They are fetched once per detector: share one `model` from `loadModel()`
between `createNnDetector` instances.

## How it works

- **Graph.** `docquadnet.json` describes 44 nodes, with weights as offsets into the fp16 blob.
  It contains 34 convolutions (1 dense 3×3 stride 2, 22 pointwise, 11 depthwise), ReLU6, 7
  residual adds, a flatten and 2 linear layers. `parseModel` lowers it into 36 steps. Each
  linear layer becomes a convolution whose kernel covers the whole input, because PyTorch's
  C-major flatten matches the weight layout. Each residual add is fused into the preceding
  projection.
- **WebGL2.** A `C×H×W` tensor is one texture of `W × H·⌈C/4⌉` texels, with 4 channels per
  texel. Each step is one full-screen draw with a fragment shader. There are 5 programs:
  dense convolution, depthwise convolution, elementwise, 2× box halving and area resize. The
  weights are RGBA16F textures that hold the exact stored fp16 values. Activations are float32
  (or float16) textures, reused by a static liveness plan. Only the 9 outputs are read back.
- **Preprocessing.** The whole frame is stretched to 192×192 (no letterbox). The CPU path
  reproduces `scanoficus_ml.data.preprocess` **bit-exactly** on the test frames: 2×2 box
  halvings while both sides are even and at least 384 px, then OpenCV's `INTER_AREA` with its
  float32 weights and rounding. The WebGL2 path does the same halvings exactly, then a
  single-pass area filter in float32. In that pass, 192 of 663,552 input values differ by one
  8-bit level from Python. For refinement, `ImageBitmap` frames are drawn to the refinement
  size by the browser (`imageSmoothingQuality: 'high'`), not by an area filter.
- **Context loss.** While the context is lost, `'auto'` answers from the CPU. After the
  browser restores the context, the next frame rebuilds the programs, weights and textures
  from the model kept in memory. `dispose()` deletes every GPU object and releases the
  context.

## Parity and speed

Checked with `pnpm --filter @scanoficus/detector-nn check:browser`. The reference is the NumPy
interpreter `scanoficus_ml.rawgraph.run_graph`. The 6 synthetic test frames (640×480 to
1280×720, including empty and noise frames) are generated bit-identically in Python and
TypeScript, so the fixtures (`test/fixtures/cases.json`, 7 kB) hold only hashes and expected
outputs. Errors are the maximum absolute difference over all 9 outputs; the presence logit is
the largest of them.

| path                                                   | max abs error vs NumPy                                      |
| ------------------------------------------------------ | ----------------------------------------------------------- |
| CPU (Vitest)                                           | 1.9·10⁻⁶                                                    |
| WebGL2 float32, network only                           | 4.3·10⁻⁶ (corners 1.8·10⁻⁷)                                 |
| WebGL2 float32, end to end with GPU preprocessing      | 1.8·10⁻⁴ (corners 2.7·10⁻⁵, presence 8.7·10⁻⁸)              |
| WebGL2 float16 activations                             | 1.6·10⁻² on the logit (corners 6.8·10⁻⁴, presence 2.7·10⁻⁵) |
| detect() on main thread and in a worker, 12 frame runs | corners within 0.033 px of the reference quad               |
| `refineQuad` vs `refine.py`                            | within 0.28 px                                              |

Speed in headless Chrome on an Apple M5 Pro (ANGLE Metal; `pnpm bench:nn`). A frame is
1280×720 unless noted. "384" is a 384×216 frame, which is what the `ScannerController` gives
detectors by default. Each warm figure is the median and 95th percentile over 100 frames
(20 for the CPU). Cold start = model fetch from localhost and parse / shader compilation and
weight upload / first frame:

| thread | backend | input             | cold start ms            | warm p50 / p95 ms |
| ------ | ------- | ----------------- | ------------------------ | ----------------- |
| main   | WebGL2  | RGBA              | 44 (10 / 20 / 13)        | 5.6 / 11.0        |
| main   | WebGL2  | ImageBitmap       | 110 (29 / 29 / 45)       | 6.8 / 10.6        |
| main   | WebGL2  | RGBA, float16     | 39 (8 / 20 / 12)         | 5.9 / 10.7        |
| main   | WebGL2  | RGBA 384          | 38 (9 / 20 / 9)          | 4.6 / 9.6         |
| main   | WebGL2  | RGBA 384 + refine | 52                       | 6.1 / 9.7         |
| main   | WebGL2  | RGBA + refine     | 73                       | 20.1 / 29.1       |
| main   | CPU     | RGBA              | 272 (9 / 0 / 263)        | 160 / 196         |
| worker | WebGL2  | ImageBitmap       | 63 (to the first result) | 6.7 / 9.2         |
| worker | WebGL2  | RGBA (copied)     | 63                       | 7.3 / 13.4        |
| worker | WebGL2  | RGBA 384          | 91                       | 4.4 / 8.2         |
| worker | CPU     | ImageBitmap       | 266                      | 159 / 178         |

Timings vary between runs by about ±1 ms (p50) and more at p95. About 1.5 ms of each WebGL2
frame is the synchronous read-back round trip. The remaining cost is about 0.1 ms per draw
(36 network draws), so float16 storage hardly helps on this GPU. Phones are not measured yet.

## Quality

These are offline results of the Python pipeline from scanoficus-ml, held-out data, official
SmartDoc Jaccard index. This package reproduces that pipeline's outputs (see above).

| detector                | SmartDoc held-out | bg 5 (clutter) | MIDV-500 | MIDV-2019 | photos without a document with a quad |
| ----------------------- | ----------------: | -------------: | -------: | --------: | ------------------------------------: |
| classic (core, current) |             0.786 |          0.000 |    0.622 |     0.587 |                                 6.5 % |
| DocQuadNet              |             0.950 |          0.884 |    0.792 |     0.793 |                                 8.8 % |
| DocQuadNet + refinement |             0.973 |          0.887 |    0.808 |     0.813 |                                 8.8 % |

On clean, high-contrast scenes the classic detector is still more precise. The classic-first
hybrid uses it there.

## License of the weights

The code is BSD-3-Clause. The weights (`docquadnet.json`, `docquadnet.bin`) were trained by us
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
scanoficus-ml repository.

## Development

```sh
pnpm --filter @scanoficus/detector-nn test            # CPU parity, preprocessing, refinement, hybrid
pnpm build && pnpm check:nn                           # WebGL2 parity and robustness in headless Chrome
pnpm bench:nn                                         # also cold start and warm timings
# regenerate the fixtures (needs the scanoficus-ml checkout and its virtualenv):
PYTHONDONTWRITEBYTECODE=1 ~/src/scanoficus-ml/.venv/bin/python packages/detector-nn/scripts/make_fixtures.py
```
