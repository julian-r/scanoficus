import { describe, expect, it } from 'vitest';
import { DetectorContext, detectDocument, sharpness } from '../src/detect/index.ts';
import { createGray, rgbaToGray } from '../src/image.ts';
import { evaluateDetection } from './support/metrics.ts';
import { generateScene, type SceneClass } from './support/synthetic.ts';

/**
 * Regression thresholds per synthetic class on 40 seeds (the full `pnpm eval` uses 300 seeds;
 * see the README for those numbers). They sit a few points below the measured rates so that
 * the suite is stable but still catches real regressions.
 */
const THRESHOLDS: Record<Exclude<SceneClass, 'none'>, { success: number; cornerError: number }> = {
  easy: { success: 1, cornerError: 0.002 },
  perspective: { success: 1, cornerError: 0.002 },
  noisy: { success: 0.97, cornerError: 0.003 },
  shadow: { success: 0.97, cornerError: 0.003 },
  border: { success: 0.95, cornerError: 0.01 },
  textured: { success: 0.92, cornerError: 0.005 },
  'low-contrast': { success: 0.75, cornerError: 0.02 },
};
const SEEDS = 40;

describe('detectDocument on synthetic scenes', () => {
  const context = new DetectorContext();

  for (const [kind, threshold] of Object.entries(THRESHOLDS) as [
    Exclude<SceneClass, 'none'>,
    { success: number; cornerError: number },
  ][]) {
    it(`detects documents in '${kind}' scenes`, () => {
      let success = 0;
      const errors: number[] = [];
      for (let seed = 1000; seed < 1000 + SEEDS; seed++) {
        const scene = generateScene(kind, seed);
        const detection = detectDocument(rgbaToGray(scene.image), {}, context);
        const outcome = evaluateDetection(scene.quad, detection?.quad ?? null, 384, 288);
        if (outcome.iou >= 0.9) {
          success++;
          errors.push(outcome.cornerError);
        }
      }
      expect(success / SEEDS).toBeGreaterThanOrEqual(threshold.success);
      const meanError = errors.reduce((a, b) => a + b, 0) / errors.length;
      expect(meanError).toBeLessThan(threshold.cornerError);
    });
  }

  it('rarely reports documents in document-free scenes', () => {
    let falsePositives = 0;
    for (let seed = 1000; seed < 1000 + SEEDS; seed++) {
      const scene = generateScene('none', seed);
      if (detectDocument(rgbaToGray(scene.image), {}, context)) falsePositives++;
    }
    expect(falsePositives / SEEDS).toBeLessThanOrEqual(0.05);
  });
});

describe('detectDocument API', () => {
  it('returns quads in input coordinates for inputs larger than maxSide', () => {
    const scene = generateScene('easy', 7, 768, 576);
    const detection = detectDocument(rgbaToGray(scene.image), { maxSide: 384 });
    expect(detection).not.toBeNull();
    const outcome = evaluateDetection(scene.quad, detection!.quad, 768, 576);
    expect(outcome.iou).toBeGreaterThan(0.97);
  });

  it('orders corners TL, TR, BR, BL and returns ranked candidates on request', () => {
    const scene = generateScene('easy', 3);
    const detection = detectDocument(rgbaToGray(scene.image), { candidates: true })!;
    const [tl, tr, br, bl] = detection.quad;
    expect(tl.x).toBeLessThan(tr.x);
    expect(bl.x).toBeLessThan(br.x);
    expect(tl.y).toBeLessThan(bl.y);
    expect(tr.y).toBeLessThan(br.y);
    expect(detection.candidates?.[0]).toEqual({ quad: detection.quad, score: detection.score });
    const scores = detection.candidates!.map((c) => c.score);
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
  });

  it('returns null for a blank frame and for tiny images', () => {
    const blank = createGray(320, 240);
    blank.data.fill(128);
    expect(detectDocument(blank)).toBeNull();
    expect(detectDocument(createGray(8, 8))).toBeNull();
  });

  it('respects minArea', () => {
    const scene = generateScene('easy', 11);
    const gray = rgbaToGray(scene.image);
    expect(detectDocument(gray, { minArea: 0.95 })).toBeNull();
  });
});

describe('sharpness', () => {
  it('is zero for flat images and decreases with blur', () => {
    const flat = createGray(50, 50);
    flat.data.fill(90);
    expect(sharpness(flat)).toBe(0);
    const sharpScene = rgbaToGray(generateScene('easy', 5).image);
    const blurred = createGray(sharpScene.width, sharpScene.height);
    const w = sharpScene.width;
    for (let i = 0; i < blurred.data.length; i++) {
      const a = sharpScene.data;
      blurred.data[i] =
        (a[i] * 4 +
          (a[i - 1] ?? a[i]) +
          (a[i + 1] ?? a[i]) +
          (a[i - w] ?? a[i]) +
          (a[i + w] ?? a[i])) /
        8;
    }
    expect(sharpness(blurred)).toBeLessThan(sharpness(sharpScene) * 0.8);
  });
});
