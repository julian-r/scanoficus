import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { Point } from '@scanoficus/core/geometry';
import { describe, expect, it } from 'vitest';
import { DOCQUADNET3 } from '../src/bundle-docquadnet3.ts';
import {
  buildPyramid,
  CpuRunner,
  cornerFrame,
  createDecision,
  createNnDetector,
  cropPlan,
  type DecisionSpec,
  type NnBundle,
  type PresenceSignals,
  parseModel,
  rgbaToRgb,
  warpCrop,
} from '../src/index.ts';
import { renderScene, type SceneCase } from './scene.ts';

interface Round3Case extends SceneCase {
  work: [number, number];
  numpy: {
    presence: number;
    passes: { quad: [number, number][]; cropSha256: string; heads: number[][] }[];
    features: Record<string, number>;
    score: number;
    accept: boolean;
    quad: [number, number][];
  };
  onnx: { score: number; accept: boolean; quad: [number, number][] };
}
const fixtures = JSON.parse(
  readFileSync(new URL('fixtures/round3.json', import.meta.url), 'utf8'),
) as { threshold: number; cases: Round3Case[] };
const file = (url: string | URL | undefined) => readFileSync(fileURLToPath(url as URL));
const refinerSource = DOCQUADNET3.refiner as NonNullable<typeof DOCQUADNET3.refiner>;
const spec = JSON.parse(file(DOCQUADNET3.decisionUrl).toString()) as DecisionSpec;
const decision = createDecision(spec);
const bundle: NnBundle = {
  name: 'docquadnet3',
  threshold: decision.threshold,
  decision,
  main: parseModel(
    JSON.parse(file(DOCQUADNET3.main.graphUrl).toString()),
    file(DOCQUADNET3.main.weightsUrl),
  ),
  refiner: {
    model: parseModel(
      JSON.parse(file(refinerSource.graphUrl).toString()),
      file(refinerSource.weightsUrl),
    ),
    scales: refinerSource.scales,
    crop: refinerSource.crop,
    workSide: refinerSource.workSide,
    coordinates: refinerSource.coordinates,
  },
};
const sha256 = (data: Uint8Array) => createHash('sha256').update(data).digest('hex');

function quadError(a: readonly Point[], b: [number, number][]): number {
  let best = Number.POSITIVE_INFINITY;
  for (let shift = 0; shift < 4; shift++) {
    let worst = 0;
    for (let i = 0; i < 4; i++) {
      const p = a[(i + shift) % 4];
      worst = Math.max(worst, Math.hypot(p.x - b[i][0], p.y - b[i][1]));
    }
    best = Math.min(best, worst);
  }
  return best;
}

describe('round-3 bundle', () => {
  it('reads the decision from decision.json', () => {
    expect(bundle.refiner?.model.outputShape).toEqual([3, 1, 1]);
    expect(spec.features).toEqual(['z', 'cl_min', 'cl_mean', 'cl1_min']);
    expect(decision.threshold).toBeCloseTo(1 / (1 + Math.exp(-spec.threshold)), 12);
    expect(fixtures.threshold).toBe(spec.threshold);
    expect(DOCQUADNET3.threshold).toBeUndefined();
  });

  it('is data-driven: new coefficients are a file swap; unknown features are refused', () => {
    const signals = {
      logit: 3,
      refiner: [
        { scale: 0.4, offsets: [], cornerLogits: [1, 2, 3, 4] },
        { scale: 0.2, offsets: [], cornerLogits: [2, 2, 4, 4] },
      ],
    } as unknown as PresenceSignals;
    const swapped = createDecision({ ...spec, weights: [1, 0, 0, 0], bias: 0 });
    expect(swapped.score(signals)?.score).toBeCloseTo(3, 5);
    const r = decision.score(signals);
    expect(r?.features).toEqual({ z: r?.features.z, cl_min: 2, cl_mean: 3, cl1_min: 1 });
    expect(() => createDecision({ ...spec, features: ['z', 'es_min'], weights: [1, 1] })).toThrow(
      /es_min/,
    );
    expect(decision.rule({ ...signals, refiner: null })).toBe(0);
  });
});

describe('round-3 parity (CPU vs scanoficus-ml)', () => {
  const refiner = new CpuRunner(bundle.refiner?.model as NonNullable<NnBundle['refiner']>['model']);
  const worst = { heads: 0, feature: 0, score: 0, scoreOnnx: 0, quad: 0, quadOnnx: 0 };
  for (const c of fixtures.cases) {
    it(c.name, async () => {
      const frame = renderScene(c);
      const rgb = { width: c.width, height: c.height, data: rgbaToRgb(frame) };
      // Pass inputs from Python: bit-exact crops, 3 outputs per crop.
      const levels = buildPyramid(rgb, refinerSource.workSide);
      expect([levels[0].width, levels[0].height]).toEqual(c.work);
      for (const [k, pass] of c.numpy.passes.entries()) {
        const q = pass.quad.map(([x, y]) => ({ x, y }));
        const crops = new Uint8Array(4 * 64 * 64 * 3);
        const input = new Float32Array(4 * 3 * 4096);
        for (let n = 0; n < 4; n++) {
          const { level, m } = cropPlan(cornerFrame(q, n, spec.scales[k]), levels.length, 64);
          const out = crops.subarray(n * 12288, (n + 1) * 12288);
          warpCrop(levels[level], m, 64, out);
          for (let ch = 0; ch < 3; ch++) {
            for (let i = 0; i < 4096; i++) input[(n * 3 + ch) * 4096 + i] = out[i * 3 + ch] / 255;
          }
        }
        expect(sha256(crops)).toBe(pass.cropSha256);
        const heads = refiner.run(input, 4);
        pass.heads.flat().forEach((v, i) => {
          worst.heads = Math.max(worst.heads, Math.abs(heads[i] - v));
        });
      }
      // The whole detector with the built-in decision (scores captured through the rule).
      let captured: ReturnType<typeof decision.score> = null;
      const detector = createNnDetector({
        bundle,
        backend: 'cpu',
        clipToFrame: false,
        presence: (s) => {
          captured = decision.score(s);
          return decision.rule(s);
        },
      });
      const detection = await detector.detect(frame); // live: the refiner runs for the decision
      const result = captured as unknown as { score: number; features: Record<string, number> };
      expect(result).not.toBeNull();
      for (const [name, value] of Object.entries(c.numpy.features)) {
        // From Python's pass inputs the refiner outputs agree to 1e-5 (above); end to end, the
        // main network's 1e-6 differences move the crops by 1e-3 px and flip some 8-bit crop
        // levels, which moves the corner logits by up to a few hundredths.
        const v = name === 'z' ? Math.min(Math.max(value, -12), 12) : value;
        worst.feature = Math.max(worst.feature, Math.abs(result.features[name] - v));
        expect(Math.abs(result.features[name] - v)).toBeLessThan(name === 'z' ? 1e-3 : 0.05);
      }
      worst.score = Math.max(worst.score, Math.abs(result.score - c.numpy.score));
      worst.scoreOnnx = Math.max(worst.scoreOnnx, Math.abs(result.score - c.onnx.score));
      expect(Math.abs(result.score - c.numpy.score)).toBeLessThan(0.05);
      expect(!!detection).toBe(c.numpy.accept);
      expect(c.onnx.accept).toBe(c.numpy.accept);
      if (detection) {
        worst.quad = Math.max(worst.quad, quadError(detection.quad, c.numpy.quad));
        worst.quadOnnx = Math.max(worst.quadOnnx, quadError(detection.quad, c.onnx.quad));
        expect(quadError(detection.quad, c.numpy.quad)).toBeLessThan(0.05);
        expect(Math.abs(detection.score - 1 / (1 + Math.exp(-result.score)))).toBeLessThan(1e-12);
      }
    });
  }
  it('reports the worst errors', () => {
    console.log(
      `round 3 CPU vs NumPy: refiner outputs ${worst.heads.toExponential(2)}, features ` +
        `${worst.feature.toFixed(4)}, score ` +
        `${worst.score.toExponential(2)}, quad ${worst.quad.toExponential(2)} px; vs ONNX: score ` +
        `${worst.scoreOnnx.toFixed(4)}, quad ${worst.quadOnnx.toFixed(4)} px`,
    );
    expect(worst.heads).toBeLessThan(1e-4);
  });
});
