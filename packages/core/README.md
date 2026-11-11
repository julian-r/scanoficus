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

By default the detector runs in a module worker created from `./worker-entry.js` next to the
package files. If your bundler pre-bundles dependencies and breaks that URL, create the worker
yourself, for example with Vite:

```ts
import DetectorWorker from '@scanoficus/core/worker-entry?worker';
import { createDetectorWorker } from '@scanoficus/core/worker';

new ScannerController({ detector: createDetectorWorker({ worker: new DetectorWorker() }) });
```

If the worker cannot start, the controller falls back to detecting on the main thread.
