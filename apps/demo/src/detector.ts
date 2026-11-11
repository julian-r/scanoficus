import { createDetectorWorker, type DocumentDetector } from '@scanoficus/core/worker';
// Vite bundles the worker entry (from the core sources) as a separate module worker.
import DetectorWorker from '@scanoficus/core/worker-entry?worker';

/** Detector running in a Vite-bundled Web Worker. */
export function createDemoDetector(): DocumentDetector {
  return createDetectorWorker({ worker: new DetectorWorker() });
}
