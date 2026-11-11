# scanoficus

A document scanner for progressive web apps on iOS Safari and Android Chrome, modeled on Apple
VisionKit's `VNDocumentCameraViewController`. Runtime dependencies are limited to the
workspace's own packages (React is an optional peer), and there is no OpenCV. The whole
scanner, UI included, is about 27 KB min+gzip.

- **Live edge detection** at camera frame rate, run in a Web Worker on a downscaled frame.
- **Quad tracking**: an animated outline that is smoothed over time and reports when it is
  stable.
- **Auto-capture** once the document is stable and sharp. A manual shutter is also available.
- **Sharp stills**: Image Capture `takePhoto()` where available, with a fallback to the video
  frame. The document is re-detected on the still.
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
  main thread.
- **Stills:** Image Capture `takePhoto()` (Safari 18.4+, Chrome Android) is tried first,
  serialized and with a timeout. The fallback is the current video frame. Resolution depends
  on the device; nothing is promised.
- **Rendering:** WebGL2. Sources larger than `MAX_TEXTURE_SIZE` are downscaled, and output is
  capped at 4000 px (about 12 MP or less). A CPU fallback is used if WebGL2 is missing or the
  context is lost.
- **Torch:** shown only when `getCapabilities().torch` is advertised.
- **Not yet verified on physical iPhones or Android phones.** Before publishing support
  claims, the release gate in the platform notes (camera selection, capture resolution, focus,
  permission persistence, memory) must be run on real devices.

## Size

min+gzip, as measured by `pnpm size` (budget: 40 KB for core):

| Entry                                                            |                     Size |
| ---------------------------------------------------------------- | -----------------------: |
| `@scanoficus/core` (all exports)                                 |                  20.8 KB |
| `@scanoficus/core/controller` (camera, detection, render, PDF)   |                  19.6 KB |
| `@scanoficus/core/detect`                                        |                   6.6 KB |
| `@scanoficus/core/render` (WebGL2 + CPU filters)                 |                   6.5 KB |
| `@scanoficus/core/worker-entry` (detector worker)                |                   6.5 KB |
| `@scanoficus/core/warp` · `/filters` · `/camera` · `/pdf`        | 3.0 · 2.2 · 2.1 · 1.7 KB |
| `@scanoficus/elements/define` (including core)                   |                  27.3 KB |
| `@scanoficus/react` (own code, without elements, core and React) |                   1.3 KB |

## Detector quality (synthetic benchmark)

`pnpm eval` renders 300 seeded synthetic scenes per class at 384×288. Each scene is a paper
sheet with text-like strokes under a random camera pose, plus noise, blur and shadows. The eval
reports the share of scenes with IoU ≥ 0.9 and the mean corner error as a percentage of the
image diagonal. Times are measured with Node 26 on an Apple-silicon Mac:

| class                                | IoU ≥ 0.9 | mean IoU | corner error | ms/frame |
| ------------------------------------ | --------: | -------: | -----------: | -------: |
| easy                                 |     100 % |    1.000 |      0.004 % |      5.0 |
| perspective (tilt ≤ 40°, roll ≤ 45°) |     100 % |    1.000 |      0.004 % |      5.2 |
| noisy + blurred                      |     100 % |    0.999 |      0.014 % |      5.1 |
| shadow (gradient + cast shadow)      |     100 % |    1.000 |      0.005 % |      5.4 |
| low contrast (paper on light table)  |      85 % |    0.971 |       0.68 % |      5.2 |
| touching / crossing the frame border |      99 % |    0.991 |       0.25 % |      5.1 |
| textured / cluttered background      |    97.7 % |    0.994 |       0.17 % |      5.4 |
| no document (false positives)        |       0 % |        – |            – |      3.7 |

The synthetic sheets have perfectly straight edges, so these corner errors are far lower than
real photos will give. Evaluation on SmartDoc 2015 is on the roadmap.

## Roadmap

- Real-device testing and tuning on iOS and Android, following the release gate in the
  platform notes.
- Evaluation on public datasets (SmartDoc 2015 Ch.1, then MIDV after a license review).
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
pnpm dev:demo    # demo dev server (use https or localhost for camera access)
```

## License

[BSD-3-Clause](LICENSE)
