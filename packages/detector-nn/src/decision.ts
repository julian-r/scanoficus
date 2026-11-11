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
}

/** A loaded decision. */
export interface Decision {
  spec: DecisionSpec;
  /** The decision's threshold as a probability: `sigmoid(spec.threshold)`. */
  threshold: number;
  /** Presence rule returning `sigmoid(score)` (0 if the refiner did not run). */
  rule: PresenceRule;
  /** The linear score and its features for one frame, or null without refiner outputs. */
  score(signals: PresenceSignals): { score: number; features: Record<string, number> } | null;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/**
 * The features `verify.features` defines and the decision can use; each needs only the main
 * network and the refiner's corner logits (no image work):
 * - `z`: presence logit as Python computes it, `log(p / (1 − p))` of the float32 sigmoid `p`
 *   (clamped to [1e-7, 1 − 1e-7]), then clipped to [−12, 12] by the decision;
 * - `cl_min` / `cl_mean`: min / mean of the last pass's corner logits;
 * - `cl1_min`: min of the first pass's corner logits.
 */
const FEATURES: Record<string, (s: PresenceSignals) => number | null> = {
  z: (s) => {
    const p = Math.min(Math.max(Math.fround(sigmoid(s.logit)), 1e-7), 1 - 1e-7);
    return Math.min(Math.max(Math.log(p / (1 - p)), -12), 12);
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
  const score = (signals: PresenceSignals) => {
    const values: Record<string, number> = {};
    let sum = 0;
    for (let k = 0; k < features.length; k++) {
      const value = FEATURES[features[k]](signals);
      if (value === null) return null;
      values[features[k]] = value;
      sum += weights[k] * value;
    }
    return { score: bias + sum, features: values };
  };
  return {
    spec,
    threshold: sigmoid(threshold),
    score,
    rule: (signals) => {
      const result = score(signals);
      return result ? sigmoid(result.score) : 0;
    },
  };
}
