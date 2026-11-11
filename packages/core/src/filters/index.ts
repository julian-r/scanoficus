/**
 * Scan filters (CPU reference implementations): grayscale, auto-levels color enhancement,
 * shadow removal and adaptive binarization, combined into VisionKit-style {@link ScanFilter}s.
 *
 * @module
 */
export {
  adaptiveThreshold,
  applyFilter,
  type EnhanceOptions,
  enhanceColor,
  removeShadows,
  SCAN_FILTERS,
  type ScanFilter,
  type ShadowRemovalOptions,
  type ThresholdOptions,
  toGrayscale,
} from './filters.ts';
