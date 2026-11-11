# scanoficus

A document scanner for progressive web apps on iOS Safari and Android Chrome, modeled on Apple
VisionKit's `VNDocumentCameraViewController`. Runtime dependencies are limited to the
workspace's own packages (React is an optional peer), and there is no OpenCV. The whole
scanner, UI included, is about 29 KB min+gzip.

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

| Package                                     | What it is                                                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [`@scanoficus/core`](packages/core)         | Headless engine: camera, detector, tracker, renderer, filters, PDF, `ScannerController`           |
| [`@scanoficus/elements`](packages/elements) | `<scan-document>`, `<scan-camera>`, `<scan-crop-editor>`, `<scan-review>`                         |
| [`@scanoficus/react`](packages/react)       | `useDocumentScanner()` and `<DocumentScanner>` / `<ScanCamera>` / `<ScanReview>` / `<CropEditor>` |
| [`apps/demo`](apps/demo)                    | Vite PWA demo (live scan, scan from photo, React, debug overlay)                                  |

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

| Entry                                                            |                     Size |
| ---------------------------------------------------------------- | -----------------------: |
| `@scanoficus/core` (all exports)                                 |                  23.0 KB |
| `@scanoficus/core/controller` (camera, detection, render, PDF)   |                  21.7 KB |
| `@scanoficus/core/detect`                                        |                   7.0 KB |
| `@scanoficus/core/render` (WebGL2 + CPU filters)                 |                   6.5 KB |
| `@scanoficus/core/worker-entry` (detector worker)                |                   6.9 KB |
| `@scanoficus/core/warp` · `/filters` · `/camera` · `/pdf`        | 3.0 · 2.2 · 2.6 · 1.7 KB |
| `@scanoficus/elements/define` (including core)                   |                  29.5 KB |
| `@scanoficus/react` (own code, without elements, core and React) |                   1.4 KB |

## Detector quality (synthetic benchmark)

`pnpm eval` renders 300 seeded synthetic scenes per class at 384×288. Each scene is a paper
sheet with text-like strokes under a random camera pose, plus noise, blur and shadows. The eval
reports the share of scenes with IoU ≥ 0.9 and the mean corner error as a percentage of the
image diagonal (default options; the real-data timings below are the reference):

| class                                                   | IoU ≥ 0.9 | mean IoU | corner error |
| ------------------------------------------------------- | --------: | -------: | -----------: |
| easy                                                    |     100 % |    1.000 |      0.004 % |
| perspective (tilt ≤ 40°, roll ≤ 45°)                    |     100 % |    1.000 |      0.004 % |
| noisy + blurred                                         |     100 % |    0.999 |      0.014 % |
| shadow (gradient + cast shadow)                         |     100 % |    1.000 |      0.005 % |
| low contrast (paper on light table)                     |      69 % |    0.979 |       0.45 % |
| touching / crossing the frame border                    |      96 % |    0.991 |       0.25 % |
| textured / cluttered background                         |    98.3 % |    0.999 |       0.02 % |
| no document, with rectangular clutter (false positives) |     9.3 % |        – |            – |

The synthetic sheets have perfectly straight edges, so these corner errors are far lower than
real photos will give. Every false positive in the document-free class is a uniform
rectangular object (a book, a box): an edge-based detector cannot tell such objects from
paper. The default score threshold is calibrated on real photos (below). It drops the
faintest synthetic low-contrast sheets, which scored 85 % before that calibration.

## Detector quality (public datasets)

`pnpm bench` runs the detector with default options on real video frames. The datasets are
fetched with `pnpm bench:fetch` and stored outside the repository. The detector was tuned on
half of the SmartDoc documents only. The other half and MIDV are held out, and 785 COCO photos
of desks, tables and screens without documents measure false positives. The table shows the
official SmartDoc Jaccard index (JI) at the default long side of 384 px, before and after
that tuning:

| dataset                                      | before | after |  best published, classic | best published, learned |
| -------------------------------------------- | -----: | ----: | -----------------------: | ----------------------: |
| SmartDoc 2015, held-out documents            |  0.652 | 0.786 |                        – |                       – |
| SmartDoc 2015, held-out documents, bg 1–4    |  0.728 | 0.877 |    0.987 (Tropin et al.) |                       – |
| SmartDoc 2015 Ch. 1, all 24,889 frames       |  0.636 | 0.772 |             0.972 (LRDE) |     0.992 (HU-PageScan) |
| SmartDoc 2015, background 5 (clutter)        |  0.001 | 0.000 |             0.861 (LRDE) |          0.986 (LDRNet) |
| MIDV-500, 10 of 50 types (held out)          |  0.789 | 0.622 | 0.906 (Tropin, 50 types) |                       – |
| MIDV-2019, 10 of 50 types (held out)         |  0.690 | 0.587 |                        – |                       – |
| photos without a document: share with a quad | 20.4 % | 6.5 % |                        – |                       – |

The tuned detector takes 5.0 ms per frame at 384 px (4.6 ms before), and corners are accurate
when it finds the document. It returns far fewer false quads, and on held-out SmartDoc
backgrounds 1–4 the best candidate is right in 0.968 JI when no threshold is applied. Its
calibrated threshold is too strict for ID cards that are held in the hand or leave the frame
(MIDV), where it now misses more cards than before. `DEFAULT_MIN_SCORE` and `CONFIDENT_SCORE`
document what a detection score means, so a hybrid can hand unsure frames to a learned
detector. See [docs/benchmarks.md](docs/benchmarks.md) for the protocol, the experiment log,
licenses, per-background tables, the comparison with published methods and the failure
analysis.

## Roadmap

- Real-device testing and tuning on iOS and Android, following the release gate in the
  platform notes.
- Detector tuning on a more diverse tuning split (hand-held and partly visible documents),
  and an optional learned detector for clutter, occlusion and documents leaving the frame;
  see [docs/benchmarks.md](docs/benchmarks.md).
- OCR adapters, lazy-loaded: `@scanoficus/ocr-tesseract`, then PaddleOCR via
  `@scanoficus/ocr-onnx`, plus searchable PDFs with an invisible text layer.
- An optional ML detector plugin (`@scanoficus/detector-ml`) behind the `DocumentDetector`
  interface.
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
```

## License

[BSD-3-Clause](LICENSE)
