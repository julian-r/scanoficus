import {
  createDetectorWorker,
  type DetectorWorkerLike,
  type DocumentDetector,
} from '@scanoficus/core/worker';
import type { NnBackend, NnDetectorOptions } from './detector.ts';
import type { GlPrecision } from './gl.ts';

/** Configuration message sent to the worker entry before any request. */
export interface NnWorkerInit {
  init: Omit<NnDetectorOptions, 'model' | 'canvas'>;
}

/** What the worker reports once its detector is initialized. */
export interface NnWorkerInfo {
  backend: NnBackend;
  /** WebGL2 activation precision, or `null` on the CPU. */
  precision: GlPrecision | null;
  /** Model download and backend initialization inside the worker (the cold start), in ms. */
  ms: number;
}

/** Status message posted by the worker entry after `init` (ignored by the core client). */
export type NnWorkerStatus = { nnReady: NnWorkerInfo } | { nnError: string };

/** The worker detector, with the outcome of the worker's initialization. */
export interface NnWorkerDetector extends DocumentDetector {
  /**
   * Resolves once the worker loaded the model and initialized its backend; rejects if that
   * failed (e.g. no WebGL2 on `OffscreenCanvas` with `backend: 'webgl2'`), if the worker
   * failed to load, or after `timeoutMs`.
   */
  ready(): Promise<NnWorkerInfo>;
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
export function createNnDetectorWorker(options: NnDetectorWorkerOptions): NnWorkerDetector {
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
  const timeout = timeoutMs ?? 30_000;
  const ready = new Promise<NnWorkerInfo>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`Neural detector worker not ready within ${timeout} ms`)),
      timeout,
    );
    const done = () => clearTimeout(timer);
    worker.addEventListener('message', (event: MessageEvent<NnWorkerStatus | object>) => {
      const data = event.data;
      if ('nnReady' in data) resolve(data.nnReady);
      else if ('nnError' in data) reject(new Error(data.nnError));
      else return;
      done();
    });
    worker.addEventListener('error', (event) => {
      done();
      reject(new Error(`Neural detector worker failed: ${event.message || 'no details'}`));
    });
  });
  ready.catch(() => {}); // observed through ready() or the detection requests
  worker.postMessage(init);
  const detector = createDetectorWorker({
    worker: worker as DetectorWorkerLike,
    transferPixels,
    timeoutMs: timeout,
  });
  return { detect: detector.detect, dispose: detector.dispose, ready: () => ready };
}
