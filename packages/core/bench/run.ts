/**
 * Runs a detector over real dataset frames and stores the raw per-frame results.
 *
 * Usage: `pnpm bench [--datasets smartdoc,midv500,midv2019,dizaj,rwmd,negatives]
 *   [--sizes 256,384,512] [--workers N] [--step 1] [--run full] [--clips <regex>]
 *   [--split tune|heldout] [--detector classic] [--options '<json>'] [--min-score <s>]
 *   [--min-area <a>] [--decode gray|rgba]`
 *
 * Each clip is processed by one worker thread: ffmpeg decodes every `step`-th frame at full
 * resolution to gray (video datasets; in colour if the detector needs it or `--decode rgba`, and
 * then converted with the library's `rgbaToGray`). Photos of the photo sets are decoded in colour
 * with their EXIF orientation applied (`decodePhoto`) and converted with `rgbaToGray`. The library's `downscaleGray` reduces it to each long side in `sizes`
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
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, renameSync, writeFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { parseArgs } from 'node:util';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';
import type { Quad } from '../src/geometry/index.ts';
import { downscaleGray, type GrayImage, type RgbaImage, rgbaToGray } from '../src/image.ts';
import { type BenchClip, DATASETS, type DatasetName, datasetKind, loadClips } from './datasets.ts';
import { BENCH_DETECTORS } from './detectors.ts';
import { decodeFrames, decodePhoto, downscaleRgba, probeSize } from './frames.ts';
import { type ClipResult, clipResultPath, type FrameResult, packQuad, runDir } from './results.ts';
import { type Split, splitOf } from './splits.ts';

/** Commit of this checkout (`+dirty` with uncommitted changes), recorded with every result. */
const COMMIT = (() => {
  try {
    const git = (args: string[]) => execFileSync('git', args, { encoding: 'utf8' }).trim();
    const head = git(['rev-parse', '--short', 'HEAD']);
    return git(['status', '--porcelain', '--', '.']) ? `${head}+dirty` : head;
  } catch {
    return 'unknown';
  }
})();

interface Job {
  clip: BenchClip;
  sizes: number[];
  step: number;
  detector: string;
  options: Record<string, unknown>;
  /** Frame decoding of video datasets: `gray` (ffmpeg luma) unless colour is needed or forced. */
  decode: 'gray' | 'rgba';
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
      decode: { type: 'string', default: 'gray' },
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
        jobs.push({
          clip,
          sizes,
          step,
          detector: values.detector,
          options,
          decode: values.decode === 'rgba' ? 'rgba' : 'gray',
          file: file.href,
        });
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

async function runClip(job: Job): Promise<number> {
  const { clip, sizes, step, detector, options, decode, file } = job;
  const selected = clip.frames.filter((_, i) => i % step === 0);
  const photo = datasetKind(clip.dataset) === 'photo';
  const factory = BENCH_DETECTORS[detector];
  const detectors = await Promise.all(sizes.map((s) => factory(s, options)));
  // Photo sets are decoded in colour and converted with the library's rgbaToGray, as the live
  // pipeline does (and as the Python photo scorer feeds the detector).
  const color = photo || decode === 'rgba' || detectors.some((d) => d.color);
  const small: (GrayImage | undefined)[] = sizes.map(() => undefined);
  const results: FrameResult[] = [];
  // Warm up the JIT on the first frame so that timings reflect the steady state.
  let warm = false;
  const onFrame = (gray: GrayImage, rgba: RgbaImage | null, i: number) => {
    const frame = selected[i];
    const runs: FrameResult['runs'] = {};
    for (let s = 0; s < sizes.length; s++) {
      const work = downscaleGray(gray, sizes[s], small[s]);
      if (work !== gray) small[s] = work;
      const workRgba = rgba && detectors[s].color ? downscaleRgba(rgba, sizes[s]) : undefined;
      if (!warm) for (let k = 0; k < 5; k++) detectors[s].detect(work, workRgba);
      const t0 = performance.now();
      const detection = detectors[s].detect(work, workRgba);
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
    const result: FrameResult = {
      index: frame.index,
      truth: frame.quad && packQuad(frame.quad),
      runs,
    };
    if (frame.id) result.id = frame.id;
    if (frame.flags) result.flags = frame.flags;
    if (frame.category) result.category = frame.category;
    results.push(result);
  };
  let size: { width: number; height: number };
  if (photo) {
    size = clip.reference;
    selected.forEach((frame, i) => {
      const rgba = decodePhoto(frame.path, 'rgba');
      if (rgba.width !== size.width || rgba.height !== size.height) {
        throw new Error(
          `${frame.id}: decoded ${rgba.width}x${rgba.height}, gt.jsonl says ${size.width}x${size.height}`,
        );
      }
      onFrame(rgbaToGray(rgba), rgba, i);
    });
  } else {
    size = probeSize(selected[0].path);
    const last = probeSize(selected[selected.length - 1].path);
    if (last.width !== size.width || last.height !== size.height) {
      throw new Error('frames of one clip differ in size');
    }
    const paths = selected.map((f) => f.path);
    if (color) {
      await decodeFrames(paths, 'rgba', size, (rgba, i) => onFrame(rgbaToGray(rgba), rgba, i));
    } else {
      await decodeFrames(paths, 'gray', size, (gray, i) => onFrame(gray, null, i));
    }
  }
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
    detector: {
      name: detector,
      options,
      decode: color ? 'rgba' : 'gray',
      ...(detectors[0].effective ?? {}),
      commit: COMMIT,
    },
    frames: results,
  };
  const tmp = new URL(`${file}.tmp`);
  writeFileSync(tmp, JSON.stringify(result));
  renameSync(tmp, new URL(file));
  return results.length;
}
