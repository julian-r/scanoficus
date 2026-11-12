/**
 * Per-frame sharpness of the hand-held video clips, for the auto-capture simulator
 * (`autocapture.ts`). Detections come from existing benchmark runs; this script adds what the
 * runs do not store: the Laplacian-variance sharpness that the live controller computes.
 *
 * Usage: `node bench/autocapture-features.ts [--bench-dir <eval-out/bench>]
 *   [--datasets smartdoc,midv500,midv2019] [--workers N]`
 *
 * For every clip, ffmpeg decodes each frame to gray, area-scaled to long side 768 (the live
 * frame size of the neural detectors), and the library's `downscaleGray` reduces it to 384 (the
 * classic detector's live size). Per frame, three values are stored:
 *
 * - `nn`: `sharpness()` at 768 in the bounding box of the neural detection (run
 *   {@link AUTOCAPTURE_RUNS}`.nn`), or of the whole frame without a detection: what the live
 *   controller sees with "Neural v4".
 * - `classic`: the same at 384 with the classic detection (run `classic`).
 * - `truth`: at 768 in the bounding box of the ground-truth quad (clamped to the frame). Only
 *   used to score the captured frame against the clip's best frame, never by a policy.
 *
 * Results go to `eval-out/autocapture/features/<dataset>/<clip>.json`; finished clips are
 * skipped.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { parseArgs } from 'node:util';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';
import { sharpness } from '../src/detect/sharpness.ts';
import { downscaleGray, fitLongSide, type GrayImage } from '../src/image.ts';
import { type BenchClip, type DatasetName, loadClips } from './datasets.ts';
import { decodeFrames, probeSize } from './frames.ts';
import type { ClipResult, PackedQuad } from './results.ts';

/** Benchmark runs replayed by the simulator: run name, size key and presence threshold. */
export const AUTOCAPTURE_RUNS = {
  /** DocQuadNet round 4 ("Neural v4"), with its calibrated presence threshold. */
  nn: { run: 'nn-own-docquadnet4', size: 192, minScore: 0.19880950805048475, liveSide: 768 },
  /** Classic detector v3 (colour), with its default `minScore`. */
  classic: { run: 'r2-v3f', size: 384, minScore: 0.74, liveSide: 384 },
} as const;
export type AutocaptureDetector = keyof typeof AUTOCAPTURE_RUNS;

export const VIDEO_DATASETS: readonly DatasetName[] = ['smartdoc', 'midv500', 'midv2019'];

export interface ClipFeatures {
  dataset: DatasetName;
  id: string;
  frameSize: { width: number; height: number };
  /** Sharpness per frame (see the module docs), rounded to 0.1. */
  nn: number[];
  classic: number[];
  truth: number[];
}

export function featuresDir(): URL {
  return new URL('../eval-out/autocapture/features/', import.meta.url);
}

export function featuresPath(dataset: DatasetName, id: string): URL {
  return new URL(`${dataset}/${id}.json`, featuresDir());
}

export function defaultBenchDir(): string {
  return (
    process.env.SCANOFICUS_BENCH_DIR ?? new URL('../eval-out/bench/', import.meta.url).pathname
  );
}

export function loadClipRun(benchDir: string, run: string, dataset: DatasetName, id: string) {
  const path = `${benchDir.replace(/\/$/, '')}/${run}/${dataset}/${id}.json`;
  return JSON.parse(readFileSync(path, 'utf8')) as ClipResult;
}

interface Job {
  clip: BenchClip;
  benchDir: string;
  file: string;
}

if (isMainThread && process.argv[1] && import.meta.filename === process.argv[1]) {
  const { values } = parseArgs({
    options: {
      'bench-dir': { type: 'string', default: defaultBenchDir() },
      datasets: { type: 'string', default: VIDEO_DATASETS.join(',') },
      workers: { type: 'string', default: String(Math.max(1, availableParallelism() - 2)) },
      clips: { type: 'string' },
    },
  });
  const filter = values.clips ? new RegExp(values.clips) : null;
  const jobs: Job[] = [];
  for (const dataset of values.datasets.split(',') as DatasetName[]) {
    mkdirSync(new URL(`${dataset}/`, featuresDir()), { recursive: true });
    for (const clip of loadClips(dataset)) {
      const file = featuresPath(dataset, clip.id);
      if (filter && !filter.test(clip.id)) continue;
      if (!existsSync(file)) jobs.push({ clip, benchDir: values['bench-dir'], file: file.href });
    }
  }
  const total = jobs.length;
  console.log(`${total} clips`);
  let done = 0;
  const started = performance.now();
  const next = async (): Promise<void> => {
    for (let job = jobs.shift(); job; job = jobs.shift()) {
      const current = job;
      await new Promise<void>((resolve) => {
        const worker = new Worker(new URL(import.meta.url), { workerData: current });
        worker.once('message', (m: { error?: string }) => {
          done++;
          if (m.error) {
            process.exitCode = 1;
            console.error(`${current.clip.id}: ${m.error}`);
          } else if (done % 10 === 0 || done === total) {
            const s = ((performance.now() - started) / 1000).toFixed(0);
            console.log(`[${done}/${total}] ${s} s`);
          }
        });
        worker.once('exit', () => resolve());
      });
    }
  };
  await Promise.all(Array.from({ length: Number(values.workers) }, next));
} else if (!isMainThread) {
  const job = workerData as Job;
  try {
    writeFileSync(new URL(job.file), JSON.stringify(await clipFeatures(job.clip, job.benchDir)));
    parentPort?.postMessage({});
  } catch (error) {
    parentPort?.postMessage({ error: (error as Error).stack ?? String(error) });
  }
}

async function clipFeatures(clip: BenchClip, benchDir: string): Promise<ClipFeatures> {
  const size = probeSize(clip.frames[0].path);
  const runs = Object.values(AUTOCAPTURE_RUNS).map((r) => {
    const result = loadClipRun(benchDir, r.run, clip.dataset, clip.id);
    if (result.frames.length !== clip.frames.length) {
      throw new Error(`${r.run}: ${result.frames.length} frames, expected ${clip.frames.length}`);
    }
    return { ...r, result };
  });
  const big = fitLongSide(size.width, size.height, 768);
  const out: ClipFeatures = {
    dataset: clip.dataset,
    id: clip.id,
    frameSize: size,
    nn: [],
    classic: [],
    truth: [],
  };
  let small: GrayImage | undefined;
  await decodeFrames(
    clip.frames.map((f) => f.path),
    'gray',
    size,
    (gray768, i) => {
      const gray384 = downscaleGray(gray768, 384, small);
      if (gray384 !== gray768) small = gray384;
      const values = runs.map(({ result, size: key, minScore, liveSide }) => {
        const run = result.frames[i].runs[key];
        const quad = run?.quad && run.score >= minScore ? run.quad : null;
        const gray = liveSide === 768 ? gray768 : gray384;
        return sharpness(gray, quad ? box(quad, gray.width / size.width) : undefined);
      });
      const truth = clip.frames[i].quad;
      const truthBox = truth
        ? box(
            truth.flatMap((p) => [p.x, p.y]),
            big.width / size.width,
          )
        : undefined;
      out.nn.push(round(values[0]));
      out.classic.push(round(values[1]));
      out.truth.push(round(sharpness(gray768, truthBox)));
    },
    big,
  );
  return out;
}

function box(q: PackedQuad, scale: number) {
  const xs = [q[0], q[2], q[4], q[6]].map((v) => v * scale);
  const ys = [q[1], q[3], q[5], q[7]].map((v) => v * scale);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y };
}

function round(v: number): number {
  return Math.round(v * 10) / 10;
}
