/**
 * Web Worker entry for document detection. Load it as a module worker and talk to it with
 * `createDetectorWorker` from `@scanoficus/core/worker`; importing this module registers the
 * message handler on the worker's global scope.
 *
 * @module
 */
import {
  type DetectorWorkerMessage,
  type DetectRequest,
  FrameProcessor,
  handleDetectRequest,
} from './processor.ts';

interface DetectorWorkerScope {
  onmessage: ((event: MessageEvent<DetectRequest>) => void) | null;
  postMessage(message: DetectorWorkerMessage): void;
}

const scope = globalThis as unknown as DetectorWorkerScope;
const processor = new FrameProcessor();
scope.onmessage = (event) => {
  scope.postMessage(handleDetectRequest(processor, event.data));
};
// Tells the client that the module loaded: errors after this are crashes, not load failures.
scope.postMessage({ ready: true });
