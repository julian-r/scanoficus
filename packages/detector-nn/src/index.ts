/**
 * `@scanoficus/detector-nn`: an optional neural document detector for scanoficus. DocQuadNet
 * (349k parameters, 192×192 RGB input) runs on a small WebGL2 runtime with a CPU fallback and
 * plugs into the scanner as a `DocumentDetector`. The weights are fetched at runtime from URLs
 * you host; they are not part of the JavaScript bundle.
 *
 * @module
 */
export { CpuRunner } from './cpu.ts';
export { decodeHeads, type NnOutput } from './decode.ts';
export {
  createNnDetector,
  DEFAULT_THRESHOLD,
  type NnBackend,
  type NnDetector,
  type NnDetectorOptions,
} from './detector.ts';
export {
  ContextLostError,
  type GlFrame,
  type GlPrecision,
  GlRunner,
  type GlRunnerOptions,
  type GlSupport,
  probeGl,
} from './gl.ts';
export { createHybridDetector, type HybridDetectorOptions } from './hybrid.ts';
export {
  type Activation,
  type GraphNode,
  halfToFloat,
  loadModel,
  MODEL_FORMAT,
  type ModelGraph,
  type ModelSource,
  type NnModel,
  parseModel,
  type Shape,
  type Step,
  type WeightRef,
} from './model.ts';
export { bytesToInput, preprocessRgba } from './preprocess.ts';
export { type RefineOptions, refineQuad } from './refine.ts';
export {
  createNnDetectorWorker,
  type NnDetectorWorkerOptions,
  type NnWorkerDetector,
  type NnWorkerInfo,
  type NnWorkerInit,
  type NnWorkerStatus,
} from './worker-client.ts';
