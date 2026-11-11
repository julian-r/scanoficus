# scanoficus demo

A Vite PWA (manifest, icons, offline shell service worker) that shows:

- **Live scan**: `<scan-document>` in a full-screen overlay, with auto-capture, review, crop
  and PDF download. Tick **Debug** before starting to see the detector's edge map, detection
  rate and latency, stability and sharpness.
- **From photo**: pick or take a picture (`capture="environment"`), then detection, the crop
  editor, review and PDF.
- **React**: `useDocumentScanner()` with a custom UI, and `<DocumentScanner>`.

The **Detector** menu in the top bar chooses the detector for the live scan (live frames and
the re-detection on the captured still) and for the photo import. The choice is stored in
localStorage:

- **Classic**: the core's edge detector in a worker.
- **Neural network**: DocQuadNet from `@scanoficus/detector-nn`, loaded lazily. The weights
  (688 kB) come from `public/models/docquadnet/`.
- **Neural + refine** (the default when WebGL2 is available, else Classic): the network, then
  classic edge refinement near its quad.
- **Hybrid**: the classic detector first, and the network only when the classic score is
  below `CONFIDENT_SCORE`.

The network runs in a worker when WebGL2 works on an `OffscreenCanvas` there, else on the main
thread. If the model or WebGL2 is unavailable, the demo falls back to Classic. With **Debug**
on, the overlay shows the detector and its backend (worker or main thread, WebGL2 float32 or
float16), the reason for any fallback, the cold start (module, model download, shader
compilation, weight upload) and the median detect time over the last 60 frames. The edge map
in the overlay is always the classic detector's, with its quad in blue. The active detector's
quad is drawn in orange.

```sh
pnpm dev:demo                                        # http://localhost:5173 (camera works on localhost)
DEMO_BASE=/scanoficus/ pnpm --filter @scanoficus/demo build && pnpm preview:demo
```

To try it on a phone, use the GitHub Pages deployment (HTTPS) or a tunnel. Plain-HTTP LAN
addresses cannot access the camera.
