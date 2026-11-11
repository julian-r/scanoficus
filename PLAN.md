# scanoficus — plan

Document scanner for PWAs (iOS Safari + Android Chrome), equivalent to Apple VisionKit's
`VNDocumentCameraViewController`: live edge detection, auto-capture, perspective correction,
filters, multi-page, PDF export. OCR is a separate, optional, lazy-loaded add-on.

License: BSD-3-Clause. Repo: github.com/julian-r/scanoficus (public).

## Decisions

- **No OpenCV.** Own TypeScript implementation. Detection runs on a downscaled frame
  (long side ≈ 320–512 px) in a Web Worker in pure TS. Full-resolution warping and filters
  run in WebGL2, with a CPU fallback. OpenCV is used only in tests as a reference, if at all.
- **Size budget:** `@scanoficus/core` ≤ 40 KB min+gz without OCR or ML. Enforce it in CI with size-limit.
- **Framework-agnostic layering:**
  - `@scanoficus/core`: pure, tree-shakeable modules with subpath exports:
    - `geometry`: points, quads, homography (DLT, 4-point), quad ordering and validation
    - `detect`: classic detector (gray → blur → Canny → dilate → contours/lines → quad scoring)
    - `warp`: perspective rectification (WebGL2 + CPU fallback), output size estimated from the quad
    - `filters`: original, color-enhance, grayscale, B&W (Sauvola/Bradley via integral image),
      shadow removal (background estimate + divide)
    - `camera`: stream acquisition and constraints, frame loop, quad tracking/smoothing,
      stability + sharpness (Laplacian variance), auto-capture state machine, still capture
      with fallbacks
    - `controller`: headless `ScannerController` (state machine + events), used by all UIs
    - `pdf`: minimal PDF writer (JPEG pages, later an invisible text layer)
    - `worker`: worker entry for detection, with a main-thread fallback
  - `@scanoficus/elements`: vanilla custom elements (`<scan-camera>`, `<scan-crop-editor>`,
    `<scan-review>`, `<scan-document>`) on the controller. Shadow DOM, CSS custom properties.
  - `@scanoficus/react`: `useDocumentScanner()` hook (controller) + thin React components.
    `react` is a peer dependency.
  - Later: `@scanoficus/ocr-tesseract`, `@scanoficus/ocr-onnx`, `@scanoficus/detector-ml`.
- **Demo:** `apps/demo`, a Vite PWA (manifest + service worker) deployed to GitHub Pages
  (HTTPS for the camera), for testing on real iOS and Android devices.
- **Tooling:** pnpm workspaces, TypeScript strict, tsdown/tsup for library builds, Vitest,
  Biome, size-limit, GitHub Actions (CI + Pages).

## Phases

1. Scaffold, plus core geometry/detect/warp/filters with unit tests and synthetic eval harness
   (rendered quads with noise, blur, shadows, low contrast; reports IoU and corner error).
2. Camera, controller, elements, React wrapper, and demo PWA (still-image path + live path).
3. Multi-page + PDF export, then real-device testing and tuning on iOS and Android.
4. Evaluation on public datasets (SmartDoc 2015 Ch.1, MIDV-500), with detector tuning.
5. OCR adapters (tesseract.js first, PaddleOCR via onnxruntime-web later) + searchable PDF.
6. Optional ML detector plugin behind the `DocumentDetector` interface.

## Platform risks to verify

Still-capture resolution (no `ImageCapture` on Safari), getUserMedia in installed iOS
PWAs, torch/focus/zoom constraints, canvas memory limits on iOS, OffscreenCanvas/WebGL
in workers, Shape Detection API (`TextDetector`) status. See `docs/platform-notes.md`.
