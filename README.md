# scanoficus

A document scanner for progressive web apps on iOS Safari and Android Chrome, modeled on Apple
VisionKit's `VNDocumentCameraViewController`. It covers live edge detection, quad tracking,
auto-capture, perspective correction, scan filters, multi-page capture and PDF export. There is
no OpenCV and core has no runtime dependencies. OCR will be a separate, optional add-on.

## Status

Early development. The architecture is described in [PLAN.md](PLAN.md).

| Package                                               | Contents                                                            | Status                             |
| ----------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------- |
| [`@scanoficus/core`](packages/core)                   | geometry, detection, tracking, warp (CPU + WebGL2), filters, worker | algorithms done and tested in Node |
| `@scanoficus/core` camera / controller / pdf          | camera loop, auto-capture state machine, PDF writer                 | planned (phase 2)                  |
| `@scanoficus/elements`, `@scanoficus/react`, demo PWA | UI layers                                                           | planned (phase 2)                  |

Nothing is published to npm yet.

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
real photos will give. Evaluation on SmartDoc 2015 and real devices is planned (see PLAN.md).

## Development

Requires Node ≥ 24 and pnpm.

```sh
pnpm install
pnpm lint        # Biome
pnpm typecheck
pnpm test        # Vitest (Node, no DOM)
pnpm build       # tsdown → packages/*/dist
pnpm size        # size-limit (core budget: 40 KB min+gzip)
pnpm eval        # synthetic detector benchmark; failure images go to packages/core/eval-out/
```

## License

[BSD-3-Clause](LICENSE)
