# scanoficus

A document scanner for progressive web apps on iOS Safari and Android Chrome, modeled on Apple
VisionKit's `VNDocumentCameraViewController`. Runtime dependencies are limited to the
workspace's own packages (React is an optional peer), and there is no OpenCV. The whole
scanner, UI included, loads about 25 KB min+gzip (plus 9 KB loaded on demand, see [Size](#size)).

- **Live edge detection** at camera frame rate, run in a Web Worker on a downscaled frame.
- **Quad tracking**: an animated outline that is smoothed over time and reports when it is
  stable.
- **Auto-capture** once the document is stable and sharp. A manual shutter is also available.
- **Sharp stills** taken from the video track at full stream resolution, with an optional
  high-resolution preset. The document is re-detected on the still.
- **Perspective correction** on the GPU (WebGL2) with a CPU fallback. The output aspect ratio
  is recovered from the perspective.
- **Filters** mirroring VisionKit: Color (shadow removal, white balance, auto levels),
  Grayscale, Black & White (Sauvola) and Photo.
- **Multi-page**: review, reorder, rotate, crop with a magnifier loupe, retake, delete.
- **PDF export** with no extra dependency. JPEG pages are embedded as-is, sized to the image,
  A4 or Letter.
- **Three API levels**: drop-in custom elements, a headless controller, and React bindings.

**Status:** pre-release (0.0.0). The algorithms are tested in Node and the UI flows in headless
Chrome with a fake camera. Real-device testing is still to come (see
[Browser support](#browser-support)). Once the repository is on GitHub, every push to `main`
deploys the demo to GitHub Pages (`.github/workflows/pages.yml`).

## Packages

| Package                                           | What it is                                                                                        |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [`@scanoficus/core`](packages/core)               | Headless engine: camera, detector, tracker, renderer, filters, PDF, `ScannerController`           |
| [`@scanoficus/elements`](packages/elements)       | `<scan-document>`, `<scan-camera>`, `<scan-crop-editor>`, `<scan-review>`                         |
| [`@scanoficus/react`](packages/react)             | `useDocumentScanner()` and `<DocumentScanner>` / `<ScanCamera>` / `<ScanReview>` / `<CropEditor>` |
| [`@scanoficus/detector-nn`](packages/detector-nn) | Optional neural detector (DocQuadNet on a small WebGL2 runtime, CPU fallback, hybrids)            |
| [`apps/demo`](apps/demo)                          | Vite PWA demo (live scan, scan from photo, React, debug overlay)                                  |

## Install

```sh
npm install @scanoficus/elements   # custom elements (includes @scanoficus/core)
npm install @scanoficus/react      # React bindings (react ≥ 18)
npm install @scanoficus/core       # headless only
```

The packages are not published yet. Until they are, build them from this repository.

The camera needs a secure context (HTTPS or `localhost`).

## Quick start

### Custom element

```html
<scan-document page-size="a4" style="position: fixed; inset: 0"></scan-document>
<script type="module">
  import '@scanoficus/elements/define';

  const scanner = document.querySelector('scan-document');
  scanner.addEventListener('scan-complete', (event) => {
    const { pages, pdf } = event.detail; // pdf: Blob, pages[i].image: JPEG Blob
    window.open(URL.createObjectURL(pdf));
  });
  scanner.addEventListener('scan-cancel', () => scanner.remove());
</script>
```

You can theme it with CSS custom properties: `--scan-accent`, `--scan-background`,
`--scan-foreground`, `--scan-toolbar-background`, `--scan-quad-fill` and `--scan-quad-stroke`.
The inner elements also expose `::part()`s.

### Headless controller

```ts
import { ScannerController } from '@scanoficus/core/controller';

const scanner = new ScannerController({ autoCapture: true, defaultFilter: 'color' });
scanner.on('detection', ({ quad, stable, progress }) => drawOverlay(quad, progress));
scanner.on('capture', ({ page }) => addThumbnail(page.url));
await scanner.start(document.querySelector('video')!);

// later
scanner.finish(); // review: camera released
await scanner.updatePage(scanner.pages[0].id, { filter: 'bw', rotation: 90 });
const pdf = await scanner.exportPdf({ pageSize: 'a4' });
scanner.dispose();
```

`ScannerController` moves through the states idle → starting → scanning ⇄ capturing →
reviewing, plus `error`. It has typed events (`state`, `detection`, `capture`, `pages`,
`error`) and a `subscribe`/`getSnapshot` pair for `useSyncExternalStore`. Scanning from a
photo works without a camera: `await scanner.addImage(file)`.

### React

```tsx
import { DocumentScanner, useDocumentScanner } from '@scanoficus/react';

// The ready-made flow
<DocumentScanner pageSize="a4" onComplete={({ pdf }) => save(pdf)} onCancel={close} />;

// Or your own UI
function MyScanner() {
  const scanner = useDocumentScanner();
  return (
    <>
      <video
        ref={(v) => {
          if (v && scanner.state === 'idle') void scanner.start(v);
        }}
        playsInline
        muted
      />
      <button onClick={() => scanner.capture()}>Capture ({scanner.pages.length})</button>
    </>
  );
}
```

The components are SSR-safe: they render an empty `<div>` on the server and mount the custom
element on the client. They work with both React 18 and 19.

## Browser support

The strategy and its evidence are documented in
[docs/platform-notes.md](docs/platform-notes.md). In short:

- **Camera:** `getUserMedia` with `facingMode: 'environment'`, 1080p preview. This works in
  Safari, installed iOS PWAs (iOS 13.4+) and Android Chrome. The camera is released when the
  page is hidden and restarted when it becomes visible again. A frame-stall watchdog drops into
  the `error` state, where the user can retry or import a photo.
- **Frames:** `requestVideoFrameCallback` (Safari 15.4+, Chrome 83+) with a
  `requestAnimationFrame` fallback. Detection runs in a module Web Worker, falling back to the
  main thread. The default worker URL is checked unbundled and with Vite 8 (dev and build,
  both as an installed package and from source). webpack 5 / Next.js are expected to work but
  are not checked. For other bundlers, pass `workerUrl` or `worker` (see the
  [core README](packages/core/README.md#the-detection-worker)).
- **Stills:** by default, the current video frame at stream resolution. On iPhone,
  `takePhoto()` returned only preview-sized images and failed with maximum photo settings.
  Choose `camera: { resolution: 'max' }` (3024×4032 at 30 fps was measured on iPhone) for
  sharper pages. `takePhoto()` is opt-in (`stillCapture: 'photo'`) and is validated against
  the stream size. Photo import (`capture="environment"`) is always available.
- **Startup:** staged (acquiring → playing → first frame) with 4 s deadlines. On iOS,
  `play()` can stall after permission was granted; the controller retries once, then offers
  `retry()`.
- **Rendering:** WebGL2. Sources larger than `MAX_TEXTURE_SIZE` are downscaled, and output is
  capped at 4000 px (about 12 MP or less). A CPU fallback is used if WebGL2 is missing or the
  context is lost.
- **Torch:** shown only when `getCapabilities().torch` is advertised.
- **Device status:** camera behavior was measured on iPhones with a lab app (summarized in
  the platform notes), and those results shaped the defaults above. The library itself has
  not yet been verified end-to-end on physical iPhones or Android phones; run the release gate
  from the platform notes before publishing support claims.

## Size

min+gzip, as measured by `pnpm size` (budget: 40 KB for core):

The controller loads two parts with `import()` only when they are needed: the main-thread
detector (the whole classic detector, about 7 KB; only without a detection worker or after the
worker failed) and the PDF writer (`exportPdf()`, about 1.7 KB). Bundlers emit them as separate
chunks. The "loaded on import" rows count only the code loaded statically (size-limit's
`entry` option); the "including" row also counts the lazy chunks.

| Entry                                                                                    |                     Size |
| ---------------------------------------------------------------------------------------- | -----------------------: |
| `@scanoficus/core` (all exports)                                                         |                  26.6 KB |
| `@scanoficus/core/controller` (camera, tracking, auto-capture, render; loaded on import) |                  17.0 KB |
| `@scanoficus/core/controller` including its lazy chunks                                  |                  26.0 KB |
| `@scanoficus/core/detect`                                                                |                   8.9 KB |
| `@scanoficus/core/render` (WebGL2 + CPU filters)                                         |                   6.5 KB |
| `@scanoficus/core/worker-entry` (detector worker)                                        |                   7.9 KB |
| `@scanoficus/core/warp` · `/filters` · `/camera` · `/pdf`                                | 3.0 · 2.2 · 2.7 · 1.7 KB |
| `@scanoficus/elements/define` (including core; loaded on import)                         |                  25.0 KB |
| `@scanoficus/react` (own code, without elements, core and React)                         |                   1.4 KB |
| `@scanoficus/detector-nn` (optional; own code, without weights)                          |                  18.4 KB |

## Detector quality (synthetic benchmark)

`pnpm eval` renders 300 seeded synthetic scenes per class at 384×288. Each scene is a paper
sheet with text-like strokes under a random camera pose, plus noise, blur and shadows. The eval
reports the share of scenes with IoU ≥ 0.9 and the mean corner error as a percentage of the
image diagonal (default options, RGBA input as in the live pipeline; `--gray` passes the luma;
the real-data timings below are the reference):

| class                                                   | IoU ≥ 0.9 | mean IoU | corner error |
| ------------------------------------------------------- | --------: | -------: | -----------: |
| easy                                                    |     100 % |    1.000 |      0.004 % |
| perspective (tilt ≤ 40°, roll ≤ 45°)                    |     100 % |    1.000 |      0.004 % |
| noisy + blurred                                         |     100 % |    0.999 |      0.014 % |
| shadow (gradient + cast shadow)                         |     100 % |    1.000 |      0.005 % |
| low contrast (paper on light table)                     |    91.3 % |    0.980 |       0.46 % |
| touching / crossing the frame border                    |    96.7 % |    0.985 |       0.37 % |
| textured / cluttered background                         |    98.0 % |    0.996 |       0.09 % |
| no document, with rectangular clutter (false positives) |     6.3 % |        – |            – |

The synthetic sheets have perfectly straight edges, so these corner errors are far lower than
real photos will give. Every false positive in the document-free class is a uniform
rectangular object (a book, a box): an edge-based detector cannot tell such objects from
paper. The default score threshold is calibrated on real photos (below). With gray input, the
low-contrast class reaches 71.7 % and the false-positive rate 4.7 %.

## Detector quality (public datasets)

`pnpm bench` runs the detector with default options on real video frames and single photos.
The datasets are fetched with `pnpm bench:fetch` (photo sets: a separate intake) and stored
outside the repository. The detector was tuned on half of the SmartDoc documents and half of
the MIDV-500 document types only. The other halves, MIDV-2019 and two single-photo sets (Dizaj,
RWMD) are held out, and 785 COCO photos of desks, tables and screens without documents measure
false positives (one half sets the thresholds, the other reports the rate). The table shows the
SmartDoc Jaccard index (JI) at the default long side of 384 px with RGBA input, for v1 (before
any tuning), v2 (round 1) and v3 (round 2), each at its default threshold:

| dataset                                      |     v1 |    v2 |    v3 | best published, classic | best published, learned |
| -------------------------------------------- | -----: | ----: | ----: | ----------------------: | ----------------------: |
| SmartDoc 2015, held-out documents            |  0.652 | 0.786 | 0.849 |                       – |                       – |
| SmartDoc 2015, held-out documents, bg 1–4    |  0.728 | 0.877 | 0.947 |   0.987 (Tropin et al.) |                       – |
| SmartDoc 2015 Ch. 1, background 5 (clutter)  |  0.001 | 0.000 | 0.000 |            0.861 (LRDE) |          0.986 (LDRNet) |
| MIDV-500, held-out half (5 of 50 types)      |  0.785 | 0.620 | 0.710 |                       – |                       – |
| MIDV-2019, 10 of 50 types                    |  0.690 | 0.588 | 0.655 |                       – |                       – |
| Dizaj, 1,055 single photos                   |  0.700 | 0.515 | 0.659 |                       – |                       – |
| RWMD, 1,975 single photos                    |  0.483 | 0.266 | 0.369 |                       – |                       – |
| photos without a document: share with a quad | 18.8 % | 6.4 % | 6.9 % |                       – |                       – |

At an equal false-positive rate (5 % of the negatives), v3 beats v1 and v2 on every held-out
set (SmartDoc held-out 0.836, MIDV-500 0.700, MIDV-2019 0.638, Dizaj 0.654, RWMD 0.356). v1
looks better on MIDV and the photos at its own default only because it returns a quad on 19 %
of the photos without a document. The largest gain came from colour edges: `detectDocument`
accepts RGBA input (e.g. `ImageData`) and adds a yellow–blue chroma gradient, which finds white
paper on a light table (SmartDoc background 4: 0.948 on held-out documents). Gray input still
works. Detection takes 6.0 ms per frame at 384 px with RGBA input (3.6 ms with gray input; v2
4.7 ms). `DEFAULT_MIN_SCORE` (0.74) and `CONFIDENT_SCORE` (0.80) document what a score means
on video frames, so a hybrid can hand unsure frames to a learned detector; on single photos
with several documents (RWMD) the score is much less reliable. See
[docs/benchmarks.md](docs/benchmarks.md) for the protocol, the experiment log, licenses, the
photo-set flags, the comparison with published methods and the failure analysis.

## Optional neural detector

[`@scanoficus/detector-nn`](packages/detector-nn) runs DocQuadNet, our own 349k-parameter
document detector, in the browser. It uses a small WebGL2 runtime written for this graph, with
a CPU fallback. It works on the main thread and in a worker, and it plugs into the scanner as
a `DocumentDetector`:

```ts
import { createNnDetectorWorker } from '@scanoficus/detector-nn';

const scanner = new ScannerController({
  createDetector: () =>
    // The model files ship in the package; bundlers emit them as content-hashed assets.
    createNnDetectorWorker({ refine: true }), // network first, then edge refinement near its quad
  detection: { maxSide: 768 }, // refinement profits from larger frames than the default 384
});
```

- **Quality.** Offline on held-out data it reaches a SmartDoc JI of 0.950 (0.973 with
  refinement), against 0.849 for the classic detector v3. It also finds pages on cluttered
  backgrounds and pages that leave the frame. On clean scenes the classic detector is still
  more precise; `createHybridDetector` uses the network only when the classic score is below
  `CONFIDENT_SCORE`. On single photos that hybrid is worse than the network alone, because the
  classic detector is confidently wrong on many of them (RWMD-Extended: see round 4).
- **Round 4** (`@scanoficus/detector-nn/docquadnet4`, optional; the demo's default): a newer
  main network, a refiner that moves each corner onto the true corner from 64×64 crops of the
  frame (2 passes) and scores it, and the verified presence decision of scanoficus-ml
  (`decision.json`, loaded as data, pinned to the weights). The controller gives detectors the
  full still through the optional `DocumentDetector.detectStill`. Offline, held-out, network
  alone: SmartDoc 0.983, MIDV-500 0.877, MIDV-2019 0.860, Dizaj test 0.882, RWMD 0.685, 6.6 %
  false positives on photos without a document; on the fresh RWMD-Extended set (503 photos)
  0.707, against 0.560 for the classic-first hybrid with round 1 (the old default) and 0.450
  for the classic detector. The refiner runs on every live frame (about 4–8 ms per frame on an
  M5 Pro). Rounds 2 and 3 are no longer in the package; the demo keeps them for comparison
  (`apps/demo/models/`).
- **Size.** The package is 18.4 KB min+gzip. The fp16 weights (round 1 688 kB; round 4
  688 kB + refiner 244 kB) ship in its `models/<bundle>/` folders with their license and
  attribution and are fetched at runtime; each graph names its weights' SHA-256, so mismatched
  files are rejected.
- **Speed.** A 1280×720 frame takes 2–6 ms (median, depending on machine load) on an M5 Pro in
  headless Chrome, and the cold start is about 35–60 ms. The CPU fallback takes 100–160 ms
  per frame. Phones are not measured yet.
- **Parity.** On the same input, the outputs match the NumPy reference within 5·10⁻⁶ (CPU and
  WebGL2 with float32 activations). End to end with GPU preprocessing, the corners match within
  2·10⁻⁵ (float32) and 7·10⁻⁴ (float16 activations).
- **License.** The weights are BSD-3. They require the SmartDoc CC BY 4.0 attribution (round 4
  also Dizaj, CC BY 4.0, and COCO / Flickr photos, CC BY 2.0, with an image list);
  see the package README.

## Roadmap

- Real-device testing and tuning on iOS and Android, following the release gate in the
  platform notes.
- An optional learned detector for clutter, stacked papers, books, occlusion and documents
  leaving the frame; see [docs/benchmarks.md](docs/benchmarks.md).
- OCR adapters, lazy-loaded: `@scanoficus/ocr-tesseract`, then PaddleOCR via
  `@scanoficus/ocr-onnx`, plus searchable PDFs with an invisible text layer.
- The neural detector (`@scanoficus/detector-nn`, available): phone measurements, and fusing
  depthwise and pointwise passes to cut the per-draw cost.
- A camera picker (switching between rear lenses) and page persistence (IndexedDB).

## Development

Requires Node ≥ 24 and pnpm.

```sh
pnpm install
pnpm lint        # Biome
pnpm typecheck
pnpm test        # Vitest (core in Node, elements/react in happy-dom)
pnpm build       # libraries (tsdown) + demo (Vite)
pnpm size        # size-limit
pnpm eval        # synthetic detector benchmark; failure images go to packages/core/eval-out/
pnpm bench:fetch # download SmartDoc 2015 / MIDV-500 / MIDV-2019 to ~/datasets/scanoficus
pnpm bench       # real-data detector benchmark; then pnpm bench:report, pnpm bench:montage
pnpm dev:demo    # demo dev server (use https or localhost for camera access)
pnpm check:nn    # detector-nn WebGL2 parity and robustness in headless Chrome (after pnpm build)
pnpm check:nn:swiftshader  # the same with the SwiftShader software renderer, as on CI
pnpm bench:nn    # detector-nn cold start and per-frame timings
```

## License

[BSD-3-Clause](LICENSE)
