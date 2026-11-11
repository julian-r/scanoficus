import {
  createDetectorWorker,
  type DetectorWorkerLike,
  type DocumentDetector,
} from '@scanoficus/core/worker';
import type { NnDetectorOptions } from './detector.ts';

/** Configuration message sent to the worker entry before any request. */
export interface NnWorkerInit {
  init: Omit<NnDetectorOptions, 'model' | 'canvas'>;
}

/** Options for {@link createNnDetectorWorker}. */
export interface NnDetectorWorkerOptions extends Omit<NnDetectorOptions, 'model' | 'canvas'> {
  /** An already created module worker running `@scanoficus/detector-nn/worker-entry`. */
  worker?: Worker;
  /**
   * URL of the worker entry module. Default: `new URL('./worker-entry.js', import.meta.url)`
   * (recognized by Vite and webpack 5). Ignored if `worker` is given.
   */
  workerUrl?: string | URL;
  /** Transfer RGBA frame buffers instead of copying them (the caller's buffer is detached). */
  transferPixels?: boolean;
  /**
   * A request not answered within this time fails the worker (fall back to another detector).
   * The first request includes loading the model and compiling the shaders. Default 30000 ms.
   */
  timeoutMs?: number;
}

/**
 * Runs the neural detector in a module Web Worker (WebGL2 on an `OffscreenCanvas`, or the CPU
 * fallback there). Failure handling is the core's `createDetectorWorker`: a worker that cannot
 * load, crashes or stops answering rejects every pending and later request, so callers (such as
 * the `ScannerController`) can fall back.
 */
export function createNnDetectorWorker(options: NnDetectorWorkerOptions): DocumentDetector {
  const {
    worker: given,
    workerUrl,
    transferPixels,
    timeoutMs,
    graphUrl,
    weightsUrl,
    ...rest
  } = options;
  // Keep this exact `new Worker(new URL(…), { type: 'module' })` form: bundlers detect it.
  const worker =
    given ??
    (workerUrl
      ? new Worker(workerUrl, { type: 'module' })
      : new Worker(new URL('./worker-entry.js', import.meta.url), { type: 'module' }));
  // The worker resolves relative URLs against its own script, so send absolute ones.
  const base = typeof location !== 'undefined' ? location.href : undefined;
  const absolute = (url: string | URL | undefined) => url && new URL(url, base).href;
  const init: NnWorkerInit = {
    init: { ...rest, graphUrl: absolute(graphUrl), weightsUrl: absolute(weightsUrl) },
  };
  worker.postMessage(init);
  return createDetectorWorker({
    worker: worker as DetectorWorkerLike,
    transferPixels,
    timeoutMs: timeoutMs ?? 30_000,
  });
}
