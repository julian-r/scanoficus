import { readFileSync } from 'node:fs';
import { CONFIDENT_SCORE } from '@scanoficus/core/detect';
import type { Quad } from '@scanoficus/core/geometry';
import { rgbaToGray } from '@scanoficus/core/image';
import type { DocumentDetector } from '@scanoficus/core/worker';
import { describe, expect, it, vi } from 'vitest';
import { createHybridDetector, createNnDetector, parseModel, refineQuad } from '../src/index.ts';
import { renderScene, type SceneCase } from './scene.ts';

const fixtures = JSON.parse(
  readFileSync(new URL('fixtures/cases.json', import.meta.url), 'utf8'),
) as {
  cases: (SceneCase & {
    quad: [number, number][];
    refined: [number, number][];
    detected: boolean;
  })[];
};
const toQuad = (q: [number, number][]) => q.map(([x, y]) => ({ x, y })) as unknown as Quad;
const maxDistance = (a: Quad, b: [number, number][]) =>
  Math.max(...a.map((p, i) => Math.hypot(p.x - b[i][0], p.y - b[i][1])));
const truthError = (a: Quad, page: [number, number][]) =>
  Math.max(...a.map((p) => Math.min(...page.map(([x, y]) => Math.hypot(p.x - x, p.y - y)))));

describe('refineQuad', () => {
  const positives = fixtures.cases.filter((c) => c.detected);
  for (const c of positives) {
    it(`matches scanoficus_ml.refine on ${c.name}`, () => {
      const gray = rgbaToGray(renderScene(c));
      const refined = refineQuad(gray, toQuad(c.quad));
      expect(maxDistance(refined, c.refined)).toBeLessThan(0.5);
    });
  }

  it('moves predicted corners onto the page edges', () => {
    const c = positives[0];
    const page = c.page as [number, number][];
    const gray = rgbaToGray(renderScene(c));
    const before = truthError(toQuad(c.quad), page);
    const after = truthError(refineQuad(gray, toQuad(c.quad)), page);
    expect(before).toBeGreaterThan(3);
    expect(after).toBeLessThan(0.5);
  });

  it('keeps the quad without edges', () => {
    const flat = { width: 192, height: 192, data: new Uint8Array(192 * 192).fill(128) };
    const quad = toQuad([
      [40, 40],
      [150, 40],
      [150, 150],
      [40, 150],
    ]);
    const refined = refineQuad(flat, quad);
    expect(
      maxDistance(
        refined,
        [...quad].map((p) => [p.x, p.y]),
      ),
    ).toBe(0);
  });
});

describe('detectors', () => {
  const model = parseModel(
    JSON.parse(
      readFileSync(
        new URL('../../../apps/demo/public/models/docquadnet/docquadnet.json', import.meta.url),
        'utf8',
      ),
    ),
    readFileSync(
      new URL('../../../apps/demo/public/models/docquadnet/docquadnet.bin', import.meta.url),
    ),
  );

  it('refines network detections when asked', async () => {
    const c = fixtures.cases[0];
    const detector = createNnDetector({ model, backend: 'cpu', refine: true });
    const detection = await detector.detect(renderScene(c));
    expect(detection).not.toBeNull();
    expect(truthError(detection?.quad as Quad, c.page as [number, number][])).toBeLessThan(0.5);
    detector.dispose();
  });

  const fake = (score: number | null): DocumentDetector & { calls: number } => {
    const detector = {
      calls: 0,
      detect: vi.fn(async () => {
        detector.calls++;
        return score === null ? null : { quad: toQuad(fixtures.cases[0].quad), score };
      }),
      dispose: vi.fn(),
    };
    return detector;
  };

  it('hybrid: confident classic detections skip the network', async () => {
    for (const score of [CONFIDENT_SCORE, Math.min(1, CONFIDENT_SCORE + 0.05)]) {
      const classic = fake(score);
      const nn = fake(0.99);
      const hybrid = createHybridDetector({ classic, nn });
      const frame = renderScene(fixtures.cases[0]);
      expect((await hybrid.detect(frame))?.score).toBe(score);
      // The RGBA frame itself (not a gray copy) reaches the classic detector: colour edges.
      expect(classic.detect).toHaveBeenCalledWith(frame, { minScore: 0 });
      expect(nn.calls).toBe(0);
      hybrid.dispose();
      expect(classic.dispose).toHaveBeenCalled();
      expect(nn.dispose).toHaveBeenCalled();
    }
  });

  it('hybrid: a custom confidence score', async () => {
    const classic = fake(0.9);
    const nn = fake(0.99);
    const hybrid = createHybridDetector({ classic, nn, confidentScore: 0.95 });
    expect((await hybrid.detect(renderScene(fixtures.cases[0])))?.score).toBe(0.99);
    expect(classic.calls).toBe(1);
    expect(nn.calls).toBe(1);
  });

  it('hybrid: the default confidence score is CONFIDENT_SCORE', async () => {
    const classic = fake(CONFIDENT_SCORE);
    const nn = fake(0.99);
    const hybrid = createHybridDetector({ classic, nn });
    const frame = renderScene(fixtures.cases[0]);
    expect((await hybrid.detect(frame))?.score).toBe(CONFIDENT_SCORE);
    expect(nn.calls).toBe(0);
    hybrid.dispose();
    expect(classic.dispose).toHaveBeenCalled();
    expect(nn.dispose).toHaveBeenCalled();
  });

  it('hybrid: unsure or missing classic detections go to the network', async () => {
    for (const score of [CONFIDENT_SCORE - 0.01, 0, null]) {
      const nn = fake(0.99);
      const hybrid = createHybridDetector({ classic: fake(score), nn });
      expect((await hybrid.detect(renderScene(fixtures.cases[0])))?.score).toBe(0.99);
      expect(nn.calls).toBe(1);
    }
    const veto = createHybridDetector({ classic: fake(CONFIDENT_SCORE - 0.01), nn: fake(null) });
    expect(await veto.detect(renderScene(fixtures.cases[0]))).toBeNull();
  });
});
