/**
 * The {@link DocumentDetector} interface with a Web Worker implementation and a main-thread
 * fallback. The worker side lives in `@scanoficus/core/worker-entry`.
 *
 * @module
 */
export {
  createDetectorWorker,
  createMainThreadDetector,
  type DetectorWorkerLike,
  type DetectorWorkerOptions,
} from './client.ts';
export {
  type DetectorFrame,
  type DetectRequest,
  type DetectResponse,
  type DocumentDetector,
  FrameProcessor,
  handleDetectRequest,
} from './processor.ts';
