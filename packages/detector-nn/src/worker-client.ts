import type { Detection } from '@scanoficus/core/detect';
import {
  createDetectorWorker,
  type DetectorFrame,
  type DetectorWorkerLike,
  type DocumentDetector,
} from '@scanoficus/core/worker';
import { type BundleSource, serializableBundle } from './bundle.ts';
import type { NnBackend, NnDetectorOptions } from './detector.ts';
import type { GlPrecision } from './gl.ts';

/** Detector options that can be posted to a worker (no functions, no loaded models). */
export type NnWorkerDetectorOptions = Omit<
  NnDetectorOptions,
  'model' | 'canvas' | 'presence' | 'bundle'
> & {
  /** A bundle source (URLs are made absolute) or the URL of a `bundle.json`. */
  bundle?: BundleSource | string | URL;
  /**
   * URL of an ES module whose default export is the {@link PresenceRule} (functions cannot be
   * posted to a worker). Bundlers: `new URL('./my-rule.js', import.meta.url)`.
   */
  presenceModule?: string | URL;
};

/** Configuration message sent to the worker entry before any request. */
export interface NnWorkerInit {
  init: Omit<NnWorkerDetectorOptions, 'presenceModule'> & { presenceModule?: string };
}

/** A still request (`detectStill`) to the worker entry. */
export interface NnStillRequest {
  nnStill: { id: number; frame: DetectorFrame };
}

/** What the worker reports once its detector is initialized. */
export interface NnWorkerInfo {
  backend: NnBackend;
  /** WebGL2 activation precision, or `null` on the CPU. */
  precision: GlPrecision | null;
  /** Model download and backend initialization inside the worker (the cold start), in ms. */
  ms: number;
}

/** Messages of the worker entry besides the core protocol (ignored by the core client). */
export type NnWorkerStatus =
  | { nnReady: NnWorkerInfo }
  | { nnError: string }
  | {
      nnStillResult: {
        id: number;
        detection?: Detection | null;
        error?: string;
        backend?: unknown;
      };
    };

/** The worker detector, with the outcome of the worker's initialization. */
export interface NnWorkerDetector extends DocumentDetector {
  /**
   * Resolves once the worker loaded the model and initialized its backend; rejects if that
   * failed (e.g. no WebGL2 on `OffscreenCanvas` with `backend: 'webgl2'`), if the worker
   * failed to load, or after `timeoutMs`.
   */
  ready(): Promise<NnWorkerInfo>;
  /** Still detection in the worker (see `NnDetector.detectStill`); consumes `ImageBitmap`s. */
  detectStill(frame: DetectorFrame): Promise<Detection | null>;
}

/** Options for {@link createNnDetectorWorker}. */
export interface NnDetectorWorkerOptions extends NnWorkerDetectorOptions {
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
export function createNnDetectorWorker(options: NnDetectorWorkerOptions = {}): NnWorkerDetector {
  const {
    worker: given,
    workerUrl,
    transferPixels,
    timeoutMs,
    graphUrl,
    weightsUrl,
    bundle,
    presenceModule,
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
    init: {
      ...rest,
      graphUrl: absolute(graphUrl),
      weightsUrl: absolute(weightsUrl),
      presenceModule: absolute(presenceModule),
      bundle:
        bundle === undefined
          ? undefined
          : typeof bundle === 'string' || bundle instanceof URL
            ? absolute(bundle)
            : serializableBundle(bundle),
    },
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
  // Still requests use their own messages (the core protocol only knows `detect`).
  const stills = new Map<
    number,
    { resolve: (d: Detection | null) => void; reject: (e: Error) => void; timer: number }
  >();
  let nextStill = 1;
  let disposed = false;
  worker.addEventListener('message', (event: MessageEvent<NnWorkerStatus | object>) => {
    if (!('nnStillResult' in event.data)) return;
    const { id, detection, error } = event.data.nnStillResult;
    const pending = stills.get(id);
    if (!pending) return;
    stills.delete(id);
    clearTimeout(pending.timer);
    if (error !== undefined) pending.reject(new Error(error));
    else pending.resolve(detection ?? null);
  });
  return {
    detect: detector.detect,
    ready: () => ready,
    detectStill(frame) {
      if (disposed) return Promise.reject(new Error('Detector disposed'));
      const id = nextStill++;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          stills.delete(id);
          reject(new Error(`Neural detector worker did not answer a still within ${timeout} ms`));
        }, timeout) as unknown as number;
        stills.set(id, { resolve, reject, timer });
        const request: NnStillRequest = { nnStill: { id, frame } };
        // Bitmaps are transferred (consumed); RGBA stills are copied (the caller keeps them).
        worker.postMessage(request, 'close' in frame ? [frame] : []);
      });
    },
    dispose() {
      disposed = true;
      for (const p of stills.values()) {
        clearTimeout(p.timer);
        p.reject(new Error('Detector disposed'));
      }
      stills.clear();
      detector.dispose();
    },
  };
}
