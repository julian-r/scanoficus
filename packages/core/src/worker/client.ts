import type { Detection, DetectOptions } from '../detect/detector.ts';
import {
  type DetectorFrame,
  type DetectRequest,
  type DetectResponse,
  type DocumentDetector,
  FrameProcessor,
} from './processor.ts';

/** The subset of the `Worker` API used by {@link createDetectorWorker}. */
export interface DetectorWorkerLike {
  postMessage(message: DetectRequest, transfer: Transferable[]): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<DetectResponse>) => void): void;
  addEventListener(type: 'error', listener: (event: Event) => void): void;
  terminate(): void;
}

/** Options for {@link createDetectorWorker}. */
export interface DetectorWorkerOptions {
  /**
   * Worker running `@scanoficus/core/worker-entry`. If omitted, a module worker is created from
   * the entry file next to this module (`new URL('./worker-entry.js', import.meta.url)`), which
   * works when the package is served as-is. Bundlers that pre-bundle dependencies may need the
   * worker created in application code instead, e.g. with Vite:
   * `import DetectorWorker from '@scanoficus/core/worker-entry?worker'` and
   * `{ worker: new DetectorWorker() }`.
   */
  worker?: DetectorWorkerLike;
  /**
   * Transfer the pixel buffer of RGBA frames instead of copying it. Faster, but the caller's
   * buffer is detached (length 0) afterwards. `ImageBitmap` frames are always transferred.
   * Default false.
   */
  transferPixels?: boolean;
}

/**
 * Runs document detection in a Web Worker so that the camera preview stays smooth. Requests are
 * answered in order; callers should keep at most one frame in flight and drop frames while
 * busy rather than queueing them.
 */
export function createDetectorWorker(options: DetectorWorkerOptions = {}): DocumentDetector {
  const worker =
    options.worker ??
    (new Worker(new URL('./worker-entry.js', import.meta.url), {
      type: 'module',
    }) as DetectorWorkerLike);
  const pending = new Map<
    number,
    { resolve: (d: Detection | null) => void; reject: (e: Error) => void }
  >();
  let nextId = 1;
  let disposed = false;

  const rejectAll = (error: Error) => {
    for (const p of pending.values()) p.reject(error);
    pending.clear();
  };
  worker.addEventListener('message', (event) => {
    const response = event.data;
    const request = pending.get(response.id);
    if (!request) return;
    pending.delete(response.id);
    if ('error' in response) request.reject(new Error(response.error));
    else request.resolve(response.detection);
  });
  worker.addEventListener('error', () => rejectAll(new Error('Detector worker failed')));

  return {
    detect(frame: DetectorFrame, detectOptions?: DetectOptions) {
      if (disposed) return Promise.reject(new Error('Detector disposed'));
      const id = nextId++;
      const transfer: Transferable[] = [];
      if ('close' in frame) transfer.push(frame);
      else if (options.transferPixels) transfer.push(frame.data.buffer);
      const request: DetectRequest = detectOptions
        ? { id, frame, options: detectOptions }
        : { id, frame };
      return new Promise<Detection | null>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        try {
          worker.postMessage(request, transfer);
        } catch (error) {
          pending.delete(id);
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      });
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      worker.terminate();
      rejectAll(new Error('Detector disposed'));
    },
  };
}

/**
 * {@link DocumentDetector} running on the calling thread, for environments without module
 * workers or `OffscreenCanvas` (and for Node). Same results as the worker, but detection
 * (≈5 ms at 384 px on a laptop, more on phones) blocks the thread; run it at a reduced rate.
 */
export function createMainThreadDetector(): DocumentDetector {
  let processor: FrameProcessor | null = new FrameProcessor();
  return {
    detect(frame: DetectorFrame, detectOptions?: DetectOptions) {
      if (!processor) return Promise.reject(new Error('Detector disposed'));
      try {
        return Promise.resolve(processor.process(frame, detectOptions));
      } catch (error) {
        return Promise.reject(error);
      }
    },
    dispose() {
      processor = null;
    },
  };
}
