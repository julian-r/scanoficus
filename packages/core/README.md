# @scanoficus/core

Framework-agnostic building blocks for a document scanner. It has no runtime dependencies,
works without a DOM (detection, filters and the CPU warp also run in Node), and is about 12 KB
min+gzip in total.

| Subpath                         | Contents                                                                                                                               |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `@scanoficus/core/image`        | `GrayImage` / `RgbaImage` (layout compatible with `ImageData`), `rgbaToGray`, `downscaleGray`                                          |
| `@scanoficus/core/geometry`     | `Point`, `Quad`, `orderQuad`, `isConvexQuad`, homographies, `convexIoU`, `estimateOutputSize` (with perspective aspect-ratio recovery) |
| `@scanoficus/core/detect`       | `detectDocument`, `DetectorContext`, `QuadTracker`, `sharpness`                                                                        |
| `@scanoficus/core/warp`         | `rectifyCpu`, `createGlRectifier` (WebGL2)                                                                                             |
| `@scanoficus/core/filters`      | `applyFilter` with `ScanFilter` = `'color' \| 'grayscale' \| 'bw' \| 'photo'`, `removeShadows`, `enhanceColor`, `adaptiveThreshold`    |
| `@scanoficus/core/worker`       | `DocumentDetector` interface, `createDetectorWorker`, `createMainThreadDetector`                                                       |
| `@scanoficus/core/worker-entry` | the module that runs inside the detection Web Worker                                                                                   |

```ts
import { createDetectorWorker } from '@scanoficus/core/worker';
import { estimateOutputSize } from '@scanoficus/core/geometry';
import { rectifyCpu } from '@scanoficus/core/warp';
import { applyFilter } from '@scanoficus/core/filters';

const detector = createDetectorWorker();
const detection = await detector.detect(await createImageBitmap(video)); // the bitmap is consumed
if (detection) {
  const size = estimateOutputSize(detection.quad, {
    imageSize: { width: video.videoWidth, height: video.videoHeight },
  });
  const page = applyFilter(
    rectifyCpu(frameImageData, detection.quad, size.width, size.height),
    'color',
  );
}
```

To build without Web Workers, use `createMainThreadDetector()`. It has the same interface.
