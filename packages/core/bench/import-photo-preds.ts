/**
 * Imports single-photo predictions made outside this repository (e.g. by a learned detector or
 * the Python photo bench in scanoficus-datasets) into a run directory, so that `bench:report`
 * scores them with the same metrics as the classic detector.
 *
 * Usage: `node bench/import-photo-preds.ts --dataset rwmd --pred <file.jsonl> --run <name>
 *   [--size 384]`
 *
 * Predictions: JSONL, one line per image `{ id, quad: [[x, y] × 4] | null, score?, ms? }`, quads
 * in original pixels after EXIF orientation (the `gt.jsonl` frame). Images missing from the file
 * count as "nothing found". `--size` only labels the run (the processing size of the detector).
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { type DatasetName, datasetKind, loadClips } from './datasets.ts';
import { type ClipResult, clipResultPath, packQuad, runDir } from './results.ts';

const { values } = parseArgs({
  options: {
    dataset: { type: 'string' },
    pred: { type: 'string' },
    run: { type: 'string' },
    size: { type: 'string', default: '384' },
  },
});
if (!values.dataset || !values.pred || !values.run) {
  throw new Error('--dataset, --pred and --run are required');
}
const dataset = values.dataset as DatasetName;
if (datasetKind(dataset) !== 'photo') throw new Error(`${dataset} is not a photo set`);
const size = Number(values.size);
const file = resolve(process.env.INIT_CWD ?? process.cwd(), values.pred);
const preds = new Map<string, { quad: [number, number][] | null; score?: number; ms?: number }>();
for (const line of readFileSync(file, 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const p = JSON.parse(line);
  preds.set(p.id, p);
}
mkdirSync(new URL(`${dataset}/`, runDir(values.run)), { recursive: true });
let missing = 0;
for (const clip of loadClips(dataset)) {
  const result: ClipResult = {
    dataset,
    id: clip.id,
    background: clip.background,
    docType: clip.docType,
    reference: clip.reference,
    frameSize: clip.reference,
    step: 1,
    sizes: [size],
    minScore: null,
    minArea: null,
    detector: { name: 'imported', source: file },
    frames: clip.frames.map((frame) => {
      const p = preds.get(frame.id ?? '');
      if (!p) missing++;
      const quad = p?.quad ? packQuad(p.quad.map(([x, y]) => ({ x, y }))) : null;
      return {
        index: frame.index,
        truth: frame.quad && packQuad(frame.quad),
        id: frame.id,
        flags: frame.flags,
        ...(frame.category ? { category: frame.category } : {}),
        runs: {
          [size]: { quad, score: quad ? (p?.score ?? 1) : 0, ms: p?.ms ?? 0, candidates: [] },
        },
      };
    }),
  };
  writeFileSync(clipResultPath(values.run, dataset, clip.id), JSON.stringify(result));
}
console.log(`imported ${preds.size} predictions into ${values.run}/${dataset}, ${missing} missing`);
