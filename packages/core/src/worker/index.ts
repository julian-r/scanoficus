/**
 * The {@link DocumentDetector} interface with a Web Worker implementation and a main-thread
 * fallback. The worker side lives in `@scanoficus/core/worker-entry`.
 *
 * @module
 */
export {
  createDetectorWorker,
  type DetectorWorkerLike,
  type DetectorWorkerOptions,
} from './client.ts';
export { createMainThreadDetector } from './main-thread.ts';
export {
  type DetectorFrame,
  type DetectorWorkerMessage,
  type DetectRequest,
  type DetectResponse,
  type DocumentDetector,
  FrameProcessor,
  handleDetectRequest,
} from './processor.ts';
