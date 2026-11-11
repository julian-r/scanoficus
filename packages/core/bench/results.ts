/** Raw per-frame benchmark results, as written by `run.ts` and read by the report scripts. */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import type { Point, Quad } from '../src/geometry/index.ts';
import type { DatasetName } from './datasets.ts';

/** Quad as `[x0, y0, x1, y1, x2, y2, x3, y3]` (TL, TR, BR, BL), rounded to 0.01 px. */
export type PackedQuad = number[];

export interface DetectorRun {
  /** Detected quad in frame pixels, or `null` if nothing was found. */
  quad: PackedQuad | null;
  score: number;
  /** Time of the `detectDocument` call on the downscaled frame. */
  ms: number;
  /** Runner-up candidates (best first) for the oracle analysis. */
  candidates: PackedQuad[];
}

export interface FrameResult {
  index: number;
  /** Ground truth; `null` for the no-document negatives. */
  truth: PackedQuad | null;
  /** Photo sets: image id, `gt.jsonl` flags and category (see `BenchFrame`). */
  id?: string;
  flags?: string[];
  category?: string;
  /** Keyed by processing long side. */
  runs: Record<number, DetectorRun>;
}

export interface ClipResult {
  dataset: DatasetName;
  id: string;
  background: string;
  docType: string;
  reference: { width: number; height: number };
  frameSize: { width: number; height: number };
  /** Every `step`-th frame was evaluated. */
  step: number;
  sizes: number[];
  /** `minScore` passed to the detector, `null` for the default. */
  minScore: number | null;
  /** `minArea` passed to the detector, `null` for the default. */
  minArea: number | null;
  /**
   * Detector that produced the run (absent in runs of the classic detector made before this
   * field existed). `options` are the detector options that differ from its defaults. Runs made
   * since round 2 also record the `decode` mode, the effective `minScore` and `minArea` (defaults
   * resolved), the tunable `params` and the git `commit` of the benchmark checkout (with
   * `+dirty` for uncommitted changes; a `source` option names the detector's source instead).
   */
  detector?: { name: string; options?: Record<string, unknown>; [key: string]: unknown };
  frames: FrameResult[];
}

export function runDir(run: string): URL {
  return new URL(`../eval-out/bench/${run}/`, import.meta.url);
}

export function clipResultPath(run: string, dataset: DatasetName, clipId: string): URL {
  return new URL(`${dataset}/${clipId}.json`, runDir(run));
}

export function loadRun(run: string, dataset: DatasetName): ClipResult[] {
  const dir = new URL(`${dataset}/`, runDir(run));
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith('.json'))
    .sort()
    .map((f) => JSON.parse(readFileSync(new URL(f, dir), 'utf8')) as ClipResult);
}

export function packQuad(quad: readonly Point[]): PackedQuad {
  return quad.flatMap((p) => [Math.round(p.x * 100) / 100, Math.round(p.y * 100) / 100]);
}

export function unpackQuad(q: PackedQuad): Quad {
  return [
    { x: q[0], y: q[1] },
    { x: q[2], y: q[3] },
    { x: q[4], y: q[5] },
    { x: q[6], y: q[7] },
  ];
}
