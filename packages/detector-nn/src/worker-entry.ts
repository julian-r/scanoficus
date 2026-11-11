/**
 * Web Worker entry of the neural detector. Start it as a module worker and talk to it with
 * `createNnDetectorWorker`: the first message configures the detector (`{ init: options }`),
 * then it answers the detection protocol of `@scanoficus/core/worker` in order, and still
 * requests (`{ nnStill: { id, frame } }`) with `{ nnStillResult }`.
 *
 * @module
 */
import type { DetectorWorkerMessage, DetectRequest } from '@scanoficus/core/worker';
import { createNnDetector, type NnDetector, type PresenceRule } from './detector.ts';
import {
  checkPresenceModule,
  type NnStillRequest,
  type NnWorkerInit,
  type NnWorkerStatus,
} from './worker-client.ts';

interface NnWorkerScope {
  onmessage: ((event: MessageEvent<DetectRequest | NnWorkerInit | NnStillRequest>) => void) | null;
  postMessage(message: DetectorWorkerMessage | NnWorkerStatus): void;
}

const scope = globalThis as unknown as NnWorkerScope;
let detector: Promise<NnDetector> | null = null;
let queue: Promise<void> = Promise.resolve();
const message = (error: unknown) => (error instanceof Error ? error.message : String(error));

scope.onmessage = (event) => {
  const data = event.data;
  if ('init' in data) {
    const previous = detector;
    previous?.then(
      (d) => d.dispose(),
      () => {},
    );
    const { presenceModule, allowCrossOriginPresenceModule, ...options } = data.init;
    const started = performance.now();
    const current = (async () => {
      // A custom presence rule cannot be posted (functions do not clone): load its module.
      const url =
        presenceModule &&
        checkPresenceModule(
          presenceModule,
          (globalThis as unknown as { location: Location }).location.origin,
          allowCrossOriginPresenceModule,
        );
      const presence = url
        ? ((await import(/* @vite-ignore */ url)).default as PresenceRule)
        : undefined;
      return createNnDetector({ ...options, presence });
    })();
    detector = current;
    // Load the model and compile the shaders before the first frame arrives, and report the
    // outcome (the core client ignores these messages; `createNnDetectorWorker` reads them).
    current
      .then(async (d) => {
        const backend = await d.ready();
        scope.postMessage({
          nnReady: { backend, precision: d.precision, ms: performance.now() - started },
        });
      })
      .catch((error) => scope.postMessage({ nnError: message(error) }));
    return;
  }
  queue = queue.then(async () => {
    const still = 'nnStill' in data;
    const id = still ? data.nnStill.id : data.id;
    try {
      if (!detector) throw new Error('Neural detector worker was not initialized');
      const d = await detector;
      if (still) {
        const detection = await d.detectStill(data.nnStill.frame);
        scope.postMessage({ nnStillResult: { id, detection, backend: d.backend } });
      } else {
        scope.postMessage({ id, detection: await d.detect(data.frame, data.options) });
      }
    } catch (error) {
      scope.postMessage(
        still ? { nnStillResult: { id, error: message(error) } } : { id, error: message(error) },
      );
    }
  });
};
// Tells the client that the module loaded: errors after this are crashes, not load failures.
scope.postMessage({ ready: true });
