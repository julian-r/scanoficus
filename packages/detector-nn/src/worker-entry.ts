/**
 * Web Worker entry of the neural detector. Start it as a module worker and talk to it with
 * `createNnDetectorWorker`: the first message configures the detector (`{ init: options }`),
 * then it answers the detection protocol of `@scanoficus/core/worker` in order.
 *
 * @module
 */
import type { DetectorWorkerMessage, DetectRequest } from '@scanoficus/core/worker';
import { createNnDetector, type NnDetector } from './detector.ts';
import type { NnWorkerInit } from './worker-client.ts';

interface NnWorkerScope {
  onmessage: ((event: MessageEvent<DetectRequest | NnWorkerInit>) => void) | null;
  postMessage(message: DetectorWorkerMessage): void;
}

const scope = globalThis as unknown as NnWorkerScope;
let detector: NnDetector | null = null;
let queue: Promise<void> = Promise.resolve();

scope.onmessage = (event) => {
  const message = event.data;
  if ('init' in message) {
    detector?.dispose();
    detector = createNnDetector(message.init);
    // Start loading the model and compiling the shaders before the first frame arrives.
    detector.ready().catch(() => {});
    return;
  }
  const { id, frame, options } = message;
  queue = queue.then(async () => {
    try {
      if (!detector) throw new Error('Neural detector worker was not initialized');
      scope.postMessage({ id, detection: await detector.detect(frame, options) });
    } catch (error) {
      scope.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
    }
  });
};
// Tells the client that the module loaded: errors after this are crashes, not load failures.
scope.postMessage({ ready: true });
