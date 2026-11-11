# scanoficus demo

A Vite PWA (manifest, icons, offline shell service worker) that shows:

- **Live scan**: `<scan-document>` in a full-screen overlay, with auto-capture, review, crop
  and PDF download. Tick **Debug** before starting to see the detector's edge map, detection
  rate and latency, stability and sharpness.
- **From photo**: pick or take a picture (`capture="environment"`), then detection, the crop
  editor, review and PDF.
- **React**: `useDocumentScanner()` with a custom UI, and `<DocumentScanner>`.

```sh
pnpm dev:demo                                        # http://localhost:5173 (camera works on localhost)
DEMO_BASE=/scanoficus/ pnpm --filter @scanoficus/demo build && pnpm preview:demo
```

To try it on a phone, use the GitHub Pages deployment (HTTPS) or a tunnel. Plain-HTTP LAN
addresses cannot access the camera.
