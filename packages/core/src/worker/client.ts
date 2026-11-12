import type { Detection, DetectOptions } from '../detect/detector.ts';
import type {
  DetectorFrame,
  DetectorWorkerMessage,
  DetectRequest,
  DocumentDetector,
} from './processor.ts';

/** The subset of the `Worker` API used by {@link createDetectorWorker}. */
export interface DetectorWorkerLike {
  postMessage(message: DetectRequest, transfer: Transferable[]): void;
  addEventListener(
    type: 'message',
    listener: (event: MessageEvent<DetectorWorkerMessage>) => void,
  ): void;
  addEventListener(type: 'error' | 'messageerror', listener: (event: Event) => void): void;
  terminate(): void;
}

/** Options for {@link createDetectorWorker}. */
export interface DetectorWorkerOptions {
  /**
   * An already created worker running `@scanoficus/core/worker-entry`, e.g. from a bundler's
   * worker import (Vite: `import DetectorWorker from '@scanoficus/core/worker-entry?worker'`).
   */
  worker?: DetectorWorkerLike;
  /**
   * URL of the worker entry module, started as a module worker (e.g. a copy of
   * `dist/worker-entry.js` and its chunks served by your app). Ignored if `worker` is given.
   *
   * Default: `new URL('./worker-entry.js', import.meta.url)`, i.e. the entry next to this
   * module. This works unbundled, with Vite (dev and build, from the package or from source)
   * and with webpack 5 / Next.js, which all recognize the `new Worker(new URL(…))` pattern.
   * Bundlers that do not (or dependency pre-bundling that moves this module) need `worker` or
   * `workerUrl`.
   */
  workerUrl?: string | URL;
  /**
   * Transfer the pixel buffer of RGBA frames instead of copying it. Faster, but the caller's
   * buffer is detached (length 0) afterwards. `ImageBitmap` frames are always transferred.
   * Default false.
   */
  transferPixels?: boolean;
  /**
   * A request not answered within this time fails the worker (it is terminated and every
   * further request rejects), so callers can fall back instead of waiting forever. Default
   * 10000 ms.
   */
  timeoutMs?: number;
}

/**
 * Runs document detection in a Web Worker so that the camera preview stays smooth. Requests are
 * answered in order; callers should keep at most one frame in flight and drop frames while
 * busy rather than queueing them.
 *
 * Failures are terminal and descriptive: if the worker script cannot be loaded, crashes,
 * cannot deserialize a message or stops answering, the worker is terminated and the pending
 * and all later `detect()` calls reject with an error naming the cause and the worker URL
 * (when known). Fall back to `createMainThreadDetector` in that case (the
 * `ScannerController` does this automatically for detectors it creates).
 */
export function createDetectorWorker(options: DetectorWorkerOptions = {}): DocumentDetector {
  let source: string;
  let worker: DetectorWorkerLike;
  if (options.worker) {
    worker = options.worker;
    source = 'the provided worker';
  } else if (options.workerUrl) {
    worker = new Worker(options.workerUrl, { type: 'module' }) as DetectorWorkerLike;
    source = String(options.workerUrl);
  } else {
    // Keep this exact `new Worker(new URL(…), { type: 'module' })` form: bundlers detect it.
    worker = new Worker(new URL('./worker-entry.js', import.meta.url), {
      type: 'module',
    }) as DetectorWorkerLike;
    source = `./worker-entry.js relative to ${import.meta.url}`;
  }
  const timeoutMs = options.timeoutMs ?? 10_000;
  const pending = new Map<
    number,
    {
      resolve: (d: Detection | null) => void;
      reject: (e: Error) => void;
      timer: ReturnType<typeof setTimeout>;
    }
  >();
  let nextId = 1;
  let loaded = false;
  /** Terminal failure: once set, every request rejects with it. */
  let failure: Error | null = null;

  const fail = (error: Error) => {
    if (failure) return;
    failure = error;
    worker.terminate();
    for (const p of pending.values()) {
      clearTimeout(p.timer);
      p.reject(error);
    }
    pending.clear();
  };

  worker.addEventListener('message', (event) => {
    const message = event.data;
    if ('ready' in message) {
      loaded = true;
      return;
    }
    loaded = true;
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    clearTimeout(request.timer);
    if ('error' in message) request.reject(new Error(message.error));
    else request.resolve(message.detection);
  });
  worker.addEventListener('error', (event) => {
    const e = event as ErrorEvent;
    const details = e.message || 'the browser gave no details';
    const location = e.filename ? ` at ${e.filename}:${e.lineno}:${e.colno}` : '';
    fail(
      new Error(
        loaded
          ? `Detector worker crashed: ${details}${location} (worker: ${source})`
          : `Detector worker failed to load: ${details}${location} (worker: ${source}). ` +
              'Check that the worker script is served (no 404) and that your bundler supports ' +
              '`new Worker(new URL(…, import.meta.url))`, or pass `worker` / `workerUrl`.',
      ),
    );
  });
  worker.addEventListener('messageerror', () => {
    fail(new Error(`Detector worker could not deserialize a message (worker: ${source})`));
  });

  return {
    detect(frame: DetectorFrame, detectOptions?: DetectOptions) {
      if (failure) return Promise.reject(failure);
      const id = nextId++;
      const transfer: Transferable[] = [];
      if ('close' in frame) transfer.push(frame);
      else if (options.transferPixels) transfer.push(frame.data.buffer);
      const request: DetectRequest = detectOptions
        ? { id, frame, options: detectOptions }
        : { id, frame };
      return new Promise<Detection | null>((resolve, reject) => {
        const timer = setTimeout(
          () =>
            fail(
              new Error(
                `Detector worker did not answer within ${timeoutMs} ms (worker: ${source})`,
              ),
            ),
          timeoutMs,
        );
        pending.set(id, { resolve, reject, timer });
        try {
          worker.postMessage(request, transfer);
        } catch (error) {
          pending.delete(id);
          clearTimeout(timer);
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      });
    },
    dispose() {
      fail(new Error('Detector disposed'));
    },
  };
}
