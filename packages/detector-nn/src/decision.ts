import type { PresenceRule, PresenceSignals } from './detector.ts';

/**
 * The data-driven presence decision of a model bundle (`decision.json`, round 3 of
 * scanoficus-ml: `scanoficus_ml/detectors.py` `OwnPipeline3`, features of
 * `scanoficus_ml/verify.py`): `score = bias + Σ weights[k] · features[k]`, a document if
 * `score > threshold`. A new round with new coefficients (same features) is a file swap.
 *
 * @module
 */

/** The fields of `decision.json` this runtime uses (others, e.g. the fit report, are ignored). */
export interface DecisionSpec {
  /** Export names of the main network and the refiner the decision was fitted with. */
  main: string;
  refiner: string;
  /** Refiner crop scales of the passes (pass 1 first). */
  scales: number[];
  features: string[];
  weights: number[];
  bias: number;
  /** Score threshold (the score is a logit; the document probability is its sigmoid). */
  threshold: number;
  /**
   * SHA-256 of the main network's and the refiner's weights the decision was fitted with
   * (added by `scripts/package_model.py`); `loadBundle` rejects other weights.
   */
  mainSha256: string;
  refinerSha256: string;
  /**
   * The scanoficus-ml file the decision was packaged from, e.g.
   * `export/docquadnet4/decision.json` (added by `scripts/package_model.py`; informational).
   */
  exportFile?: string;
}

/** A loaded decision. */
export interface Decision {
  spec: DecisionSpec;
  /** The decision's threshold as a probability: `sigmoid(spec.threshold)`. */
  threshold: number;
  /** Presence rule returning `sigmoid(score)` (0 if the refiner did not run). */
  rule: PresenceRule;
  /** The linear score of one frame and its features, or null without refiner outputs. */
  score(signals: PresenceSignals): { score: number; features: Record<string, number> } | null;
  /** The linear score of given feature values (`z` unclipped; the decision clips it). */
  scoreFeatures(features: Record<string, number>): number;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/**
 * The features `verify.features` defines and the decision can use; each needs only the main
 * network and the refiner's corner logits (no image work):
 * - `z`: presence logit as Python computes it, `log(p / (1 − p))` of the float32 sigmoid `p`
 *   (clamped to [1e-7, 1 − 1e-7]); the score clips it to [−12, 12];
 * - `cl_min` / `cl_mean`: min / mean of the last pass's corner logits;
 * - `cl1_min`: min of the first pass's corner logits.
 */
const FEATURES: Record<string, (s: PresenceSignals) => number | null> = {
  z: (s) => {
    const p = Math.min(Math.max(Math.fround(sigmoid(s.logit)), 1e-7), 1 - 1e-7);
    return Math.log(p / (1 - p));
  },
  cl_min: (s) => {
    const l = s.refiner?.at(-1)?.cornerLogits;
    return l ? Math.min(...l) : null;
  },
  cl_mean: (s) => {
    const l = s.refiner?.at(-1)?.cornerLogits;
    return l ? l.reduce((a, b) => a + b, 0) / l.length : null;
  },
  cl1_min: (s) => {
    const l = s.refiner?.[0]?.cornerLogits;
    return l ? Math.min(...l) : null;
  },
};

/** Supported feature names. */
export const DECISION_FEATURES: readonly string[] = Object.keys(FEATURES);

/** Validates a `decision.json` and builds its presence rule. */
export function createDecision(spec: DecisionSpec): Decision {
  const { features, weights, bias, threshold, scales } = spec ?? {};
  if (!Array.isArray(features) || !Array.isArray(weights) || features.length !== weights.length) {
    throw new Error('decision.json: features and weights must be arrays of the same length');
  }
  if (![bias, threshold, ...weights].every(Number.isFinite) || !Array.isArray(scales)) {
    throw new Error('decision.json: bias, threshold, weights and scales are required');
  }
  const unknown = features.filter((f) => !Object.hasOwn(FEATURES, f));
  if (unknown.length) {
    throw new Error(
      `decision.json uses features this runtime cannot compute: ${unknown.join(', ')} ` +
        `(supported: ${DECISION_FEATURES.join(', ')})`,
    );
  }
  /** As `OwnPipeline3.detect`: `bias + Σ w·f`, with `z` clipped to [−12, 12]. */
  const scoreFeatures = (values: Record<string, number>) => {
    let sum = 0;
    for (let k = 0; k < features.length; k++) {
      const v = values[features[k]];
      sum += weights[k] * (features[k] === 'z' ? Math.min(Math.max(v, -12), 12) : v);
    }
    return bias + sum;
  };
  const score = (signals: PresenceSignals) => {
    const values: Record<string, number> = {};
    for (const name of features) {
      const value = FEATURES[name](signals);
      if (value === null) return null;
      values[name] = value;
    }
    return { score: scoreFeatures(values), features: values };
  };
  return {
    spec,
    threshold: sigmoid(threshold),
    score,
    scoreFeatures,
    rule: (signals) => {
      const result = score(signals);
      return result ? sigmoid(result.score) : 0;
    },
  };
}
