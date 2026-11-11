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
- **Neural network**: DocQuadNet from `@scanoficus/detector-nn`, loaded lazily. The model
  files (688 kB) come from the package and are emitted with content hashes, so the service
  worker never serves an old graph or old weights. The footer links their license and the
  SmartDoc and Dizaj attributions.
- **Neural + refine**: the network, then classic edge refinement near its quad, on 768-px
  frames instead of 384.
- **Hybrid** (the default when float WebGL2 rendering works, else Classic; a stored choice is
  kept): the classic detector first (at 384 px); when its score is below `CONFIDENT_SCORE`
  (0.80), "Neural + refine" (round 1, presence > 0.85, 768-px frames). This is the combination
  with the best offline JI on every set with documents.
- **Neural v2 + refiner**: round 2 of the network (threshold 0.98) on live frames (raw quad),
  and the corner refiner on the full-resolution still after capture and on imported photos
  (`detectStill`). The Debug overlay shows the still's size and time. Its model files (930 kB)
  are only fetched when this option is used.

The network runs in a worker when WebGL2 works on an `OffscreenCanvas` there, else on the main
thread. If the model or WebGL2 is unavailable, or the network fails, the demo falls back to
Classic. A WebGL context loss only skips frames: the network recovers after the browser
restores the context. With **Debug**
on, the overlay shows the detector and its backend (worker or main thread, WebGL2 float32 or
float16), the reason for any fallback, context-loss notices, the cold start (module, model download, shader
compilation, weight upload) and the median detect time over the last 60 frames. The edge map
in the overlay is always the classic detector's, with its quad in blue. The active detector's
quad is drawn in orange.

```sh
pnpm dev:demo                                        # http://localhost:5173 (camera works on localhost)
DEMO_BASE=/scanoficus/ pnpm --filter @scanoficus/demo build && pnpm preview:demo
```

To try it on a phone, use the GitHub Pages deployment (HTTPS) or a tunnel. Plain-HTTP LAN
addresses cannot access the camera.
