/**
 * Detector implementations the benchmark runner can run by name (`pnpm bench --detector <name>
 * --options '<json>'`). Each entry turns the JSON options into a detector instance for one
 * processing size; the runner creates one instance per clip and size, so instances may keep
 * per-stream state (buffers, temporal priors).
 */
import {
  type Detection,
  type DetectOptions,
  DetectorContext,
  detectDocument,
} from '../src/detect/index.ts';
import type { GrayImage } from '../src/image.ts';

export interface BenchDetector {
  /**
   * Detects the document in `gray`, which is already downscaled to the processing long side.
   * Returns the ranked candidates (best first, including the winner) if it has any.
   */
  detect(gray: GrayImage): Detection | null;
}

export type BenchDetectorFactory = (
  size: number,
  options: Record<string, unknown>,
) => BenchDetector;

export const BENCH_DETECTORS: Record<string, BenchDetectorFactory> = {
  /** The library's `detectDocument`; `options` are `DetectOptions` (except `maxSide`). */
  classic(size, options) {
    const context = new DetectorContext();
    const detectOptions: DetectOptions = { ...options, maxSide: size, candidates: true };
    return { detect: (gray) => detectDocument(gray, detectOptions, context) };
  },
};
