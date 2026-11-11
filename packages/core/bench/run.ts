/**
 * Runs the classic detector over real dataset frames and stores the raw per-frame results.
 *
 * Usage: `pnpm bench [--datasets smartdoc,midv500,midv2019] [--sizes 256,384,512]
 *   [--workers N] [--step 1] [--run full] [--clips <regex>]
 *   [--min-score <s>] [--min-area <a>]`
 *
 * Each clip is processed by one worker thread: ffmpeg decodes every `step`-th frame at full
 * resolution to gray, the library's `downscaleGray` reduces it to each long side in `sizes`
 * (as the live pipeline hands the detector an already downscaled frame), and `detectDocument`
 * runs with default options (plus `candidates: true` to record the ranked alternatives). Only
 * the `detectDocument` call is timed. `--min-score` and `--min-area` override the detector
 * defaults for diagnostic runs only (with `--min-score 0`, the best candidate of every frame is
 * recorded); benchmark numbers use the defaults. Results go to
 * `eval-out/bench/<run>/<dataset>/<clip>.json`; finished clips are skipped, so an interrupted run
 * resumes where it stopped. `pnpm bench:report` turns them into tables.
 */
import { existsSync, mkdirSync, renameSync, writeFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { parseArgs } from 'node:util';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';
import { DetectorContext, detectDocument } from '../src/detect/index.ts';
import type { Quad } from '../src/geometry/index.ts';
import { downscaleGray, type GrayImage } from '../src/image.ts';
import { type BenchClip, DATASETS, type DatasetName, loadClips } from './datasets.ts';
import { decodeFrames, probeSize } from './frames.ts';
import { type ClipResult, clipResultPath, type FrameResult, packQuad, runDir } from './results.ts';

interface Job {
  clip: BenchClip;
  sizes: number[];
  step: number;
  minScore: number | undefined;
  minArea: number | undefined;
  /** `file:` URL of the result file (URLs cannot be posted to workers). */
  file: string;
}

if (isMainThread) {
  const { values } = parseArgs({
    options: {
      datasets: { type: 'string', default: DATASETS.join(',') },
      sizes: { type: 'string', default: '256,384,512' },
      workers: { type: 'string', default: String(Math.max(1, availableParallelism() - 2)) },
      step: { type: 'string', default: '1' },
      run: { type: 'string', default: 'full' },
      clips: { type: 'string' },
      'min-score': { type: 'string' },
      'min-area': { type: 'string' },
    },
  });
  const sizes = values.sizes.split(',').map(Number);
  const step = Number(values.step);
  const optional = (v: string | undefined) => (v === undefined ? undefined : Number(v));
  const minScore = optional(values['min-score']);
  const minArea = optional(values['min-area']);
  const workers = Number(values.workers);
  const filter = values.clips ? new RegExp(values.clips) : null;
  const jobs: Job[] = [];
  let skipped = 0;
  for (const dataset of values.datasets.split(',') as DatasetName[]) {
    let clips: BenchClip[];
    try {
      clips = loadClips(dataset);
    } catch (error) {
      console.warn(`skipping ${dataset}: ${(error as Error).message}`);
      continue;
    }
    mkdirSync(new URL(`${dataset}/`, runDir(values.run)), { recursive: true });
    for (const clip of clips) {
      if (filter && !filter.test(clip.id)) continue;
      const file = clipResultPath(values.run, dataset, clip.id);
      if (existsSync(file)) skipped++;
      else jobs.push({ clip, sizes, step, minScore, minArea, file: file.href });
    }
  }
  console.log(`${jobs.length} clips to run (${skipped} already done), ${workers} workers`);
  const started = performance.now();
  const total = jobs.length;
  let done = 0;
  let failed = 0;
  const next = async (): Promise<void> => {
    for (let job = jobs.shift(); job; job = jobs.shift()) {
      const current = job;
      await new Promise<void>((resolve) => {
        const worker = new Worker(new URL(import.meta.url), { workerData: current });
        const name = `${current.clip.dataset}/${current.clip.id}`;
        let reported = false;
        const report = (error: string | undefined, frames?: number) => {
          if (reported) return;
          reported = true;
          done++;
          if (error) {
            failed++;
            console.error(`${name}: ${error}`);
          } else {
            const elapsed = ((performance.now() - started) / 1000).toFixed(0);
            console.log(`[${done}/${total}] ${name}: ${frames} frames (${elapsed} s)`);
          }
        };
        worker.once('message', (message: { error?: string; frames?: number }) =>
          report(message.error, message.frames),
        );
        worker.once('error', (error: Error) => report(error.stack ?? String(error)));
        worker.once('exit', (code) => {
          report(`worker exited with code ${code} without a result`);
          resolve();
        });
      });
    }
  };
  await Promise.all(Array.from({ length: workers }, next));
  console.log(
    `finished in ${((performance.now() - started) / 1000).toFixed(0)} s, ${failed} failed`,
  );
  if (failed) process.exitCode = 1;
} else {
  const job = workerData as Job;
  try {
    const frames = await runClip(job);
    parentPort?.postMessage({ frames });
  } catch (error) {
    parentPort?.postMessage({ error: (error as Error).stack ?? String(error) });
  }
}

async function runClip({ clip, sizes, step, minScore, minArea, file }: Job): Promise<number> {
  const selected = clip.frames.filter((_, i) => i % step === 0);
  const size = probeSize(selected[0].path);
  const last = probeSize(selected[selected.length - 1].path);
  if (last.width !== size.width || last.height !== size.height) {
    throw new Error('frames of one clip differ in size');
  }
  const contexts = sizes.map(() => new DetectorContext());
  const small: (GrayImage | undefined)[] = sizes.map(() => undefined);
  const results: FrameResult[] = [];
  // Warm up the JIT on the first frame so that timings reflect the steady state.
  let warm = false;
  await decodeFrames(
    selected.map((f) => f.path),
    'gray',
    size,
    (gray, i) => {
      const frame = selected[i];
      const runs: FrameResult['runs'] = {};
      for (let s = 0; s < sizes.length; s++) {
        const work = downscaleGray(gray, sizes[s], small[s]);
        if (work !== gray) small[s] = work;
        if (!warm) {
          for (let k = 0; k < 5; k++) detectDocument(work, { maxSide: sizes[s] }, contexts[s]);
        }
        const t0 = performance.now();
        const detection = detectDocument(
          work,
          { maxSide: sizes[s], minScore, minArea, candidates: true },
          contexts[s],
        );
        const ms = performance.now() - t0;
        const sx = gray.width / work.width;
        const sy = gray.height / work.height;
        const toFrame = (q: Quad) => packQuad(q.map((p) => ({ x: p.x * sx, y: p.y * sy })));
        runs[sizes[s]] = detection
          ? {
              quad: toFrame(detection.quad),
              score: Math.round(detection.score * 1e4) / 1e4,
              ms: Math.round(ms * 1e3) / 1e3,
              candidates: (detection.candidates ?? []).slice(1).map((c) => toFrame(c.quad)),
            }
          : { quad: null, score: 0, ms: Math.round(ms * 1e3) / 1e3, candidates: [] };
      }
      warm = true;
      results.push({ index: frame.index, truth: packQuad(frame.quad), runs });
    },
  );
  const result: ClipResult = {
    dataset: clip.dataset,
    id: clip.id,
    background: clip.background,
    docType: clip.docType,
    reference: clip.reference,
    frameSize: size,
    step,
    sizes,
    minScore: minScore ?? null,
    minArea: minArea ?? null,
    frames: results,
  };
  const tmp = new URL(`${file}.tmp`);
  writeFileSync(tmp, JSON.stringify(result));
  renameSync(tmp, new URL(file));
  return results.length;
}
