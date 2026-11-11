/**
 * Failure montages for the real-data benchmark: frames with the ground truth (green), the
 * detection (red) and the best runner-up candidate (blue, thin), numbered, with a legend file.
 *
 * Usage: `pnpm bench:montage [--run full] [--dataset smartdoc] [--size 384] [--split tune]
 *   [--mode worst|failures] [--k 24] [--per-clip 2] [--threshold 0.9] [--seed 1]`
 *
 * - `worst`: the K lowest-scoring frames per background, at most `per-clip` per clip.
 * - `failures`: a seeded random sample of K frames per background among those scoring below
 *   `threshold`, at most `per-clip` per clip (for estimating failure-mode shares).
 *
 * The score is the SmartDoc Jaccard index for SmartDoc, the visible-region IoU for MIDV and
 * 1 − detector confidence of the false detection for the negatives. `--split tune` restricts
 * the frames to the tuning split (look only at those while developing the detector).
 * Pages of 12 tiles go to `eval-out/bench/montage/`.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { createRgba, type RgbaImage } from '../src/image.ts';
import { encodePng } from '../test/support/png.ts';
import { Random } from '../test/support/random.ts';
import { type DatasetName, datasetKind, loadClips } from './datasets.ts';
import { decodeFrames } from './frames.ts';
import { frameScore } from './report-data.ts';
import { type ClipResult, loadRun, type PackedQuad } from './results.ts';
import { splitOf } from './splits.ts';

const { values } = parseArgs({
  options: {
    run: { type: 'string', default: 'full' },
    dataset: { type: 'string', default: 'smartdoc' },
    size: { type: 'string', default: '384' },
    mode: { type: 'string', default: 'worst' },
    k: { type: 'string', default: '24' },
    'per-clip': { type: 'string', default: '2' },
    threshold: { type: 'string', default: '0.9' },
    seed: { type: 'string', default: '1' },
    split: { type: 'string' },
  },
});
const dataset = values.dataset as DatasetName;
if (datasetKind(dataset) === 'photo') {
  throw new Error(`${dataset} is a frozen photo test set; montages are for video datasets only`);
}
const size = Number(values.size);
const k = Number(values.k);
const perClip = Number(values['per-clip']);
const threshold = Number(values.threshold);
const TILE_WIDTH = 480;
const COLUMNS = 4;
const TILES_PER_PAGE = 12;
/** 3×5 bitmap digits, one row per string, '#' = set. */
const DIGITS = [
  '###,#.#,#.#,#.#,###',
  '.#.,##.,.#.,.#.,###',
  '###,..#,###,#..,###',
  '###,..#,.##,..#,###',
  '#.#,#.#,###,..#,..#',
  '###,#..,###,..#,###',
  '###,#..,###,#.#,###',
  '###,..#,.#.,.#.,.#.',
  '###,#.#,###,#.#,###',
  '###,#.#,###,..#,###',
].map((d) => d.split(','));

const outDir = new URL('../eval-out/bench/montage/', import.meta.url);
mkdirSync(outDir, { recursive: true });

interface Pick {
  clip: ClipResult;
  frame: ClipResult['frames'][number];
  score: number;
}

const paths = new Map<string, string>();
for (const clip of loadClips(dataset)) {
  for (const f of clip.frames) paths.set(`${clip.id}#${f.index}`, f.path);
}
const clips = loadRun(values.run, dataset).filter(
  (clip) => !values.split || splitOf(clip) === values.split,
);
const backgrounds = [...new Set(clips.map((c) => c.background))].sort();
const random = new Random(Number(values.seed));

for (const background of backgrounds) {
  const pool: Pick[] = [];
  for (const clip of clips.filter((c) => c.background === background)) {
    for (const frame of clip.frames)
      pool.push({ clip, frame, score: frameScore(clip, frame, size) });
  }
  // Shuffle first so that ties (e.g. the many frames without a detection) are sampled evenly.
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(random.next() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const ordered =
    values.mode === 'worst'
      ? pool.sort((a, b) => a.score - b.score)
      : pool.filter((p) => p.score < threshold);
  const perClipCount = new Map<string, number>();
  const picks: Pick[] = [];
  for (const p of ordered) {
    const n = perClipCount.get(p.clip.id) ?? 0;
    if (n >= perClip) continue;
    perClipCount.set(p.clip.id, n + 1);
    picks.push(p);
    if (picks.length === k) break;
  }
  const legend: string[] = [];
  for (let page = 0; page * TILES_PER_PAGE < picks.length; page++) {
    const tiles = picks.slice(page * TILES_PER_PAGE, (page + 1) * TILES_PER_PAGE);
    const tileHeight = Math.max(
      ...tiles.map(({ clip }) =>
        Math.round((TILE_WIDTH * clip.frameSize.height) / clip.frameSize.width),
      ),
    );
    const rows = Math.ceil(tiles.length / COLUMNS);
    const sheet = createRgba(COLUMNS * TILE_WIDTH, rows * tileHeight);
    for (let t = 0; t < tiles.length; t++) {
      const { clip, frame, score } = tiles[t];
      const number = page * TILES_PER_PAGE + t + 1;
      const path = paths.get(`${clip.id}#${frame.index}`);
      if (!path) throw new Error(`frame ${clip.id}#${frame.index} not found in the dataset`);
      const scale = TILE_WIDTH / clip.frameSize.width;
      const tileSize = { width: TILE_WIDTH, height: Math.round(clip.frameSize.height * scale) };
      let image: RgbaImage | null = null;
      await decodeFrames(
        [path],
        'rgba',
        clip.frameSize,
        (rgba) => {
          image = createRgba(rgba.width, rgba.height);
          image.data.set(rgba.data);
        },
        tileSize,
      );
      if (!image) throw new Error(`cannot decode ${path}`);
      const tile = image as RgbaImage;
      const run = frame.runs[size];
      if (frame.truth) drawQuad(tile, frame.truth, scale, [0, 255, 0], 2);
      if (run.candidates[0]) drawQuad(tile, run.candidates[0], scale, [40, 120, 255], 1);
      if (run.quad) drawQuad(tile, run.quad, scale, [255, 0, 0], 2);
      drawNumber(tile, number, 6, 6);
      blit(sheet, tile, (t % COLUMNS) * TILE_WIDTH, Math.floor(t / COLUMNS) * tileHeight);
      legend.push(
        `${number}\t${clip.id}\tframe ${frame.index}\tscore ${score.toFixed(3)}\tdetector score ${run.score}\t${run.quad ? 'found' : 'not found'}\t${path}`,
      );
    }
    const name = `${values.run}-${dataset}-${size}-${values.mode}-${background}-${page + 1}.png`;
    writeFileSync(new URL(name, outDir), encodePng(sheet));
    console.log(new URL(name, outDir).pathname);
  }
  const legendName = `${values.run}-${dataset}-${size}-${values.mode}-${background}.txt`;
  writeFileSync(new URL(legendName, outDir), `${legend.join('\n')}\n`);
}

function drawQuad(
  image: RgbaImage,
  quad: PackedQuad,
  scale: number,
  color: [number, number, number],
  thickness: number,
): void {
  for (let s = 0; s < 4; s++) {
    const ax = quad[2 * s] * scale;
    const ay = quad[2 * s + 1] * scale;
    const bx = quad[(2 * s + 2) % 8] * scale;
    const by = quad[(2 * s + 3) % 8] * scale;
    const steps = Math.ceil(Math.hypot(bx - ax, by - ay) * 2);
    for (let i = 0; i <= steps; i++) {
      const x = ax + ((bx - ax) * i) / steps;
      const y = ay + ((by - ay) * i) / steps;
      fillRect(
        image,
        Math.floor(x - thickness / 2),
        Math.floor(y - thickness / 2),
        thickness,
        thickness,
        color,
      );
    }
  }
}

function fillRect(
  image: RgbaImage,
  x0: number,
  y0: number,
  w: number,
  h: number,
  color: [number, number, number],
): void {
  for (let y = Math.max(0, y0); y < Math.min(image.height, y0 + h); y++) {
    for (let x = Math.max(0, x0); x < Math.min(image.width, x0 + w); x++) {
      const p = (y * image.width + x) * 4;
      image.data[p] = color[0];
      image.data[p + 1] = color[1];
      image.data[p + 2] = color[2];
      image.data[p + 3] = 255;
    }
  }
}

function drawNumber(image: RgbaImage, value: number, x: number, y: number): void {
  const text = String(value);
  const cell = 4;
  fillRect(image, x - 3, y - 3, text.length * 4 * cell + 3, 5 * cell + 6, [0, 0, 0]);
  for (let c = 0; c < text.length; c++) {
    const glyph = DIGITS[Number(text[c])];
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 3; col++) {
        if (glyph[row][col] === '#') {
          fillRect(image, x + (c * 4 + col) * cell, y + row * cell, cell, cell, [255, 255, 0]);
        }
      }
    }
  }
}

function blit(dst: RgbaImage, src: RgbaImage, x0: number, y0: number): void {
  for (let y = 0; y < src.height && y0 + y < dst.height; y++) {
    const from = y * src.width * 4;
    dst.data.set(src.data.subarray(from, from + src.width * 4), ((y0 + y) * dst.width + x0) * 4);
  }
}
