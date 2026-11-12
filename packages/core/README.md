# @scanoficus/core

Framework-agnostic engine of the scanoficus document scanner. It has no runtime dependencies.
Detection, filters, the CPU warp and the PDF writer also run in Node.

| Subpath                         | Contents                                                                                                                            |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `@scanoficus/core/controller`   | `ScannerController`: camera, live detection, auto-capture, pages, export (state machine + typed events)                             |
| `@scanoficus/core/camera`       | `openCamera`, `listCameras`, `startFrameLoop`, `FrameGrabber`, `watchPageVisibility`                                                |
| `@scanoficus/core/detect`       | `detectDocument`, `DetectorContext`, `QuadTracker`, `sharpness`                                                                     |
| `@scanoficus/core/render`       | `createPageRenderer` (WebGL2 rectify + filter + rotate, CPU fallback), `renderPageCpu`                                              |
| `@scanoficus/core/filters`      | `applyFilter` with `ScanFilter` = `'color' \| 'grayscale' \| 'bw' \| 'photo'`, `removeShadows`, `enhanceColor`, `adaptiveThreshold` |
| `@scanoficus/core/warp`         | `rectifyCpu`, `createGlRectifier`                                                                                                   |
| `@scanoficus/core/pdf`          | `createPdf` (JPEG pages via `DCTDecode`, image/A4/Letter page sizes), `readJpegInfo`                                                |
| `@scanoficus/core/geometry`     | `Quad`, `orderQuad`, homographies, `convexIoU`, `estimateOutputSize`                                                                |
| `@scanoficus/core/worker`       | `DocumentDetector` interface, `createDetectorWorker`, `createMainThreadDetector`                                                    |
| `@scanoficus/core/worker-entry` | the module that runs inside the detection Web Worker                                                                                |
| `@scanoficus/core/image`        | `GrayImage` / `RgbaImage` (layout compatible with `ImageData`), `rgbaToGray`, `downscaleGray`                                       |

```ts
import { ScannerController } from '@scanoficus/core/controller';

const scanner = new ScannerController();
scanner.on('detection', ({ quad, progress }) => draw(quad, progress));
await scanner.start(videoElement);
// … auto-capture adds pages; then:
scanner.finish();
const pdf = await scanner.exportPdf({ pageSize: 'a4' });
```

`ScannerController` loads two parts on demand with `import()`, as separate chunks of `dist/`:
the main-thread detector (only without a working detection worker) and the PDF writer (on the
first `exportPdf()`). Serve the whole `dist/` folder (bundlers do this automatically).

### Auto-capture

Auto-capture fires when the document has been steady and sharp for `stableMs` (600 ms) in
total. Steady means that, over the last 500 ms, the corners move at most 0.1 diagonals/s
(`tracker.maxSpeed`) and scatter at most 1.5 % around that motion (`tracker.stabilityThreshold`),
with the worst quarter of the detections ignored. Sharp means at least `minRelativeSharpness`
(0.6) of the recent 75th percentile. The progress decays instead of resetting. A re-framing
restarts it. The sharpest ready full-resolution frame of the last `bestFrameMs` (300 ms)
becomes the page. `deviceMotion: true` also uses the motion sensor (call
`requestDeviceMotionPermission()` from a tap on iOS). `detection.autoCapture` tells which
criterion fails, with values and limits. See [docs/autocapture.md](../../docs/autocapture.md)
for the evaluation on hand-held video.

### The detection worker

By default, `ScannerController` creates and owns a module worker started with
`new Worker(new URL('./worker-entry.js', import.meta.url), { type: 'module' })`, using the
entry shipped next to the package files. Coverage by setup:

| Setup                                                                                         | Works by default?                                                                |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Serving `dist/` unbundled (import map or plain ESM)                                           | yes, checked by `pnpm check:browser`                                             |
| Vite 8 dev (with dependency pre-bundling) and build, consuming the installed package          | yes, checked by `pnpm check:pack` (packed tarball fixture)                       |
| Vite 8 dev and build from source through the `@scanoficus/source` export condition (the demo) | yes, checked by `pnpm check:browser`                                             |
| Other Vite versions                                                                           | expected, not checked                                                            |
| webpack 5 / Next.js                                                                           | expected (both recognize `new Worker(new URL(…, import.meta.url))`), not checked |
| esbuild, and bundlers or pre-bundling that do not handle the pattern                          | no: use `workerUrl` or `worker`                                                  |

To control the worker yourself, use one of these:

```ts
import { createDetectorWorker } from '@scanoficus/core/worker';
// 1. Your own URL to a served copy of dist/worker-entry.js and its chunks:
const detector = createDetectorWorker({ workerUrl: '/assets/scanoficus/worker-entry.js' });
// 2. A bundler-created worker, e.g. Vite:
import DetectorWorker from '@scanoficus/core/worker-entry?worker';
const detector2 = createDetectorWorker({ worker: new DetectorWorker() });

// Let the controller own (and dispose) it, with automatic main-thread fallback:
new ScannerController({
  createDetector: () => createDetectorWorker({ worker: new DetectorWorker() }),
});
```

Worker failures are terminal and descriptive. A script that cannot be loaded, a crash, a
message deserialization error, or no answer within `timeoutMs` (10 s) makes every pending and
later `detect()` reject. The message names the cause and the worker URL, for example
`Detector worker failed to load: the browser gave no details (worker: …/worker-entry.js)`.
Browsers do not report why a worker script failed to load: check the network panel for a 404.
`ScannerController` then switches to main-thread detection and emits a non-fatal `error`
event that starts with `Falling back to main-thread detection:`. A detector passed as
`detector` is owned by the caller: it is neither disposed nor replaced.

If you reference library source files directly (for example through an alias), import the
worker entry by its package export (`@scanoficus/core/worker-entry`) rather than by its file
path, which may change.
