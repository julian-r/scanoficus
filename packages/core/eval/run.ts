/**
 * Synthetic evaluation of the document detector.
 *
 * Usage: `pnpm eval [--n 300] [--classes easy,shadow] [--failures 8] [--gray]`
 *
 * For every difficulty class, renders N seeded scenes at 384×288, runs `detectDocument` on the
 * RGBA image (as the live pipeline does; `--gray`: on its luma) and reports: found rate, success rate (IoU ≥ 0.9), mean IoU and mean/p90 corner
 * error (% of the image diagonal) over found documents, and mean/p95 detection time. For the
 * document-free class 'none', "found" is the false-positive rate. Failure cases are written to
 * `eval-out/` as PNG (green: ground truth, red: detection).
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { DetectorContext, detectDocument } from '../src/detect/index.ts';
import { rgbaToGray } from '../src/image.ts';
import { drawQuads, evaluateDetection } from '../test/support/metrics.ts';
import { encodePng } from '../test/support/png.ts';
import { generateScene, SCENE_CLASSES, type SceneClass } from '../test/support/synthetic.ts';

const { values } = parseArgs({
  options: {
    n: { type: 'string', default: '300' },
    classes: { type: 'string', default: SCENE_CLASSES.join(',') },
    failures: { type: 'string', default: '8' },
    gray: { type: 'boolean', default: false },
  },
});
const n = Number(values.n);
const classes = values.classes.split(',') as SceneClass[];
const maxFailureImages = Number(values.failures);
const outDir = new URL('../eval-out/', import.meta.url);
mkdirSync(outDir, { recursive: true });

const context = new DetectorContext();
const pct = (v: number, digits = 1) => `${(100 * v).toFixed(digits)}%`;
const quantile = (sorted: number[], q: number) =>
  sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : Number.NaN;

// Warm up the JIT so that timings reflect steady-state per-frame cost.
for (let s = 0; s < 20; s++) {
  const image = generateScene('easy', 10_000 + s).image;
  detectDocument(values.gray ? rgbaToGray(image) : image, {}, context);
}

const rows: string[][] = [];
for (const kind of classes) {
  let found = 0;
  let success = 0;
  const ious: number[] = [];
  const errors: number[] = [];
  const times: number[] = [];
  let written = 0;
  for (let seed = 0; seed < n; seed++) {
    const scene = generateScene(kind, seed);
    const gray = rgbaToGray(scene.image);
    const input = values.gray ? gray : scene.image;
    const t0 = performance.now();
    const detection = detectDocument(input, {}, context);
    times.push(performance.now() - t0);
    const outcome = evaluateDetection(scene.quad, detection?.quad ?? null, gray.width, gray.height);
    if (outcome.found) found++;
    if (scene.quad && outcome.found) {
      ious.push(outcome.iou);
      errors.push(outcome.cornerError);
      if (outcome.iou >= 0.9) success++;
    }
    const failed = scene.quad ? outcome.iou < 0.9 : outcome.found;
    if (failed && written < maxFailureImages) {
      written++;
      const quads = [];
      if (scene.quad)
        quads.push({ quad: scene.quad, color: [0, 255, 0] as [number, number, number] });
      if (detection)
        quads.push({ quad: detection.quad, color: [255, 0, 0] as [number, number, number] });
      const file = new URL(`${kind}-${seed}.png`, outDir);
      writeFileSync(file, encodePng(drawQuads(scene.image, quads)));
    }
  }
  ious.sort((a, b) => a - b);
  errors.sort((a, b) => a - b);
  times.sort((a, b) => a - b);
  const mean = (v: number[]) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : Number.NaN);
  rows.push([
    kind,
    String(n),
    pct(found / n),
    kind === 'none' ? '-' : pct(success / n),
    kind === 'none' ? '-' : mean(ious).toFixed(3),
    kind === 'none' ? '-' : pct(mean(errors), 3),
    kind === 'none' ? '-' : pct(quantile(errors, 0.9), 3),
    mean(times).toFixed(2),
    quantile(times, 0.95).toFixed(2),
  ]);
}

const header = [
  'class',
  'n',
  'found',
  'IoU≥0.9',
  'mean IoU',
  'corner err',
  'p90 err',
  'ms',
  'p95 ms',
];
const widths = header.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i].length)));
const format = (r: string[]) => r.map((c, i) => c.padStart(widths[i])).join('  ');
console.log(format(header));
for (const r of rows) console.log(format(r));
console.log(`\nfailure images: ${new URL('.', outDir).pathname}`);
