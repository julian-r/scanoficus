/**
 * Classic (non-ML) document detection on gray frames, temporal quad tracking and a focus measure.
 *
 * @module
 */
export {
  CONFIDENT_SCORE,
  DEFAULT_MIN_SCORE,
  type Detection,
  type DetectOptions,
  DetectorContext,
  detectDocument,
  type ScoredQuad,
} from './detector.ts';
export { type Rect, sharpness } from './sharpness.ts';
export { QuadTracker, type QuadTrackerOptions, type TrackerState } from './tracker.ts';
