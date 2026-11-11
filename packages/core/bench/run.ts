/**
 * Runs a detector over real dataset frames and stores the raw per-frame results.
 *
 * Usage: `pnpm bench [--datasets smartdoc,midv500,midv2019,negatives] [--sizes 256,384,512]
 *   [--workers N] [--step 1] [--run full] [--clips <regex>] [--split tune|heldout]
 *   [--detector classic] [--options '<json>'] [--min-score <s>] [--min-area <a>]`
 *
 * Each clip is processed by one worker thread: ffmpeg decodes every `step`-th frame at full
 * resolution to gray, the library's `downscaleGray` reduces it to each long side in `sizes`
 * (as the live pipeline hands the detector an already downscaled frame), and the detector named
 * by `--detector` (see `detectors.ts`; default `classic`, the library's `detectDocument`) runs
 * with the JSON `--options` (default: none, i.e. the detector defaults). Only the detector call is
 * timed. `--min-score` and `--min-area` are shortcuts for those options, meant for diagnostic runs
 * (with `--min-score 0`, the best candidate of every frame is recorded). `--split` restricts the
 * run to the tuning or held-out clips (see `splits.ts`; the negatives belong to neither and are
 * always included). Results go to `eval-out/bench/<run>/<dataset>/<clip>.json`; finished clips
 * are skipped, so an interrupted run resumes where it stopped. `pnpm bench:report` turns them
 * into tables.
 */
import { existsSync, mkdirSync, renameSync, writeFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { parseArgs } from 'node:util';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';
import type { Quad } from '../src/geometry/index.ts';
import { downscaleGray, type GrayImage } from '../src/image.ts';
import { type BenchClip, DATASETS, type DatasetName, loadClips } from './datasets.ts';
import { BENCH_DETECTORS } from './detectors.ts';
import { decodeFrames, probeSize } from './frames.ts';
import { type ClipResult, clipResultPath, type FrameResult, packQuad, runDir } from './results.ts';
import { type Split, splitOf } from './splits.ts';

interface Job {
  clip: BenchClip;
  sizes: number[];
  step: number;
  detector: string;
  options: Record<string, unknown>;
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
      split: { type: 'string' },
      detector: { type: 'string', default: 'classic' },
      options: { type: 'string', default: '{}' },
      'min-score': { type: 'string' },
      'min-area': { type: 'string' },
    },
  });
  const sizes = values.sizes.split(',').map(Number);
  const step = Number(values.step);
  if (!(values.detector in BENCH_DETECTORS)) {
    throw new Error(`Unknown detector ${values.detector}; known: ${Object.keys(BENCH_DETECTORS)}`);
  }
  const options = JSON.parse(values.options) as Record<string, unknown>;
  if (values['min-score'] !== undefined) options.minScore = Number(values['min-score']);
  if (values['min-area'] !== undefined) options.minArea = Number(values['min-area']);
  const split = values.split as Split | undefined;
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
      const clipSplit = splitOf(clip);
      if (split && clipSplit !== null && clipSplit !== split) continue;
      const file = clipResultPath(values.run, dataset, clip.id);
      if (existsSync(file)) skipped++;
      else {
        jobs.push({ clip, sizes, step, detector: values.detector, options, file: file.href });
      }
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

async function runClip({ clip, sizes, step, detector, options, file }: Job): Promise<number> {
  const selected = clip.frames.filter((_, i) => i % step === 0);
  const size = probeSize(selected[0].path);
  const last = probeSize(selected[selected.length - 1].path);
  if (last.width !== size.width || last.height !== size.height) {
    throw new Error('frames of one clip differ in size');
  }
  const detectors = sizes.map((s) => BENCH_DETECTORS[detector](s, options));
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
          for (let k = 0; k < 5; k++) BENCH_DETECTORS[detector](sizes[s], options).detect(work);
        }
        const t0 = performance.now();
        const detection = detectors[s].detect(work);
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
      results.push({ index: frame.index, truth: frame.quad && packQuad(frame.quad), runs });
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
    minScore: typeof options.minScore === 'number' ? options.minScore : null,
    minArea: typeof options.minArea === 'number' ? options.minArea : null,
    detector: { name: detector, options },
    frames: results,
  };
  const tmp = new URL(`${file}.tmp`);
  writeFileSync(tmp, JSON.stringify(result));
  renameSync(tmp, new URL(file));
  return results.length;
}
