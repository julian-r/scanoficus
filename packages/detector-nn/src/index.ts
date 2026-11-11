/**
 * `@scanoficus/detector-nn`: an optional neural document detector for scanoficus. DocQuadNet
 * (349k parameters, 192×192 RGB input) runs on a small WebGL2 runtime with a CPU fallback and
 * plugs into the scanner as a `DocumentDetector`. The weights are fetched at runtime from URLs
 * you host; they are not part of the JavaScript bundle.
 *
 * @module
 */

export {
  BUNDLE_FORMAT,
  type BundleSource,
  loadBundle,
  type NnBundle,
  type RefinerModel,
  type RefinerSettings,
  serializableBundle,
} from './bundle.ts';
export { DOCQUADNET } from './bundle-docquadnet.ts';
export {
  applyOffset,
  buildPyramid,
  type CornerFrame,
  cornerFrame,
  cropPlan,
  pyramidSizes,
  type RefinerPass,
  type RefinerResult,
  type RgbImage,
  refineCornersCpu,
  warpCrop,
  workSize,
} from './corners.ts';
export { CpuRunner } from './cpu.ts';
export { decodeHeads, fitQuadToFrame, type NnOutput } from './decode.ts';
export {
  createNnDetector,
  DEFAULT_THRESHOLD,
  type NnBackend,
  type NnDetector,
  type NnDetectorOptions,
  type PresenceRule,
  type PresenceSignals,
  presenceProbability,
  type RefinerUse,
} from './detector.ts';
export {
  ContextLostError,
  createGlContext,
  FrameChangedError,
  type GlFrame,
  type GlPrecision,
  GlRunner,
  type GlRunnerOptions,
  type GlSupport,
  probeGl,
} from './gl.ts';
export { createHybridDetector, HYBRID_DEFAULTS, type HybridDetectorOptions } from './hybrid.ts';
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
  verifyWeights,
  type WeightRef,
} from './model.ts';
export { bytesToInput, preprocessRgb, preprocessRgba, rgbaToRgb } from './preprocess.ts';
export { type RefineOptions, refineQuad } from './refine.ts';
export {
  checkPresenceModule,
  createNnDetectorWorker,
  type NnDetectorWorkerOptions,
  type NnStillRequest,
  type NnWorkerDetector,
  type NnWorkerDetectorOptions,
  type NnWorkerInfo,
  type NnWorkerInit,
  type NnWorkerStatus,
} from './worker-client.ts';
