/**
 * Tuned constants of the classic detector, in one object so that the benchmark can vary them
 * per experiment (`pnpm bench --options '{"params":{…}}'`, see docs/benchmarks.md). Internal:
 * not exported from the package, and the library never changes them. Values were chosen on the
 * tuning split (round 2, experiment log in docs/benchmarks.md).
 */
export const PARAMS = {
  /** Canny thresholds: high = max(1, highPerMedian · median gradient), low = high / 2. */
  highPerMedian: 2,
  /** Score multiplier for a quad with a side on the frame border. */
  framePenalty: 0.85,
  /**
   * Extra multiplier for a second frame side. 0.85 · 0.95 ≈ 0.81 stays above
   * `DEFAULT_MIN_SCORE`, so a well-supported document leaving the frame on two sides can pass.
   */
  secondFramePenalty: 0.95,
  /** Quads smaller than this fraction of the frame are penalized (∝ area^sizeExponent). */
  fullSizeArea: 0.25,
  sizeExponent: 0.15,
  /** Weight of the length-weighted mean support (the rest goes to the weakest side). */
  meanWeight: 0.9,
  /** Exponent of the paper-brightness factor (inside vs. surrounding band). */
  brightnessExponent: 0.5,
  /** Minimum score of an enclosing candidate, relative to the best (page over content). */
  enclosingScore: 0.85,
  /** Default `minArea`. */
  minArea: 0.04,
  /**
   * Weight of the yellow–blue chroma gradient against the luma gradient for RGBA input; 0 uses
   * the luma gradient only.
   */
  chromaWeight: 1,
};
