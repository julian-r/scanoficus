import { readFileSync } from 'node:fs';
import { CONFIDENT_SCORE, DEFAULT_MIN_SCORE } from '@scanoficus/core/detect';
import type { Quad } from '@scanoficus/core/geometry';
import type { DocumentDetector } from '@scanoficus/core/worker';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  createGlContext,
  createHybridDetector,
  createNnDetector,
  fitQuadToFrame,
  parseModel,
} from '../src/index.ts';
import { DOCQUADNET_GRAPH_URL, DOCQUADNET_WEIGHTS_URL } from '../src/model-files.ts';
import { renderScene, type SceneCase } from './scene.ts';

const quad = (...points: [number, number][]) =>
  points.map(([x, y]) => ({ x, y })) as unknown as Quad;
const cases = JSON.parse(readFileSync(new URL('fixtures/cases.json', import.meta.url), 'utf8'))
  .cases as SceneCase[];
const model = parseModel(
  JSON.parse(readFileSync(DOCQUADNET_GRAPH_URL, 'utf8')),
  readFileSync(DOCQUADNET_WEIGHTS_URL),
);

describe('fitQuadToFrame', () => {
  it('keeps quads inside the frame', () => {
    const q = quad([10, 10], [90, 12], [88, 70], [12, 68]);
    expect(fitQuadToFrame(q, 100, 80)).toEqual(q);
  });

  it('clamps corners outside the frame onto its border', () => {
    const fitted = fitQuadToFrame(quad([-20, 10], [90, -5], [120, 70], [12, 95]), 100, 80);
    expect(fitted).toEqual(quad([0, 10], [90, 0], [100, 70], [12, 80]));
  });

  it('rejects non-convex, self-intersecting, degenerate and non-finite quads', () => {
    expect(fitQuadToFrame(quad([10, 10], [90, 10], [50, 20], [10, 70]), 100, 80)).toBeNull();
    expect(fitQuadToFrame(quad([10, 10], [90, 70], [90, 10], [10, 70]), 100, 80)).toBeNull();
    expect(fitQuadToFrame(quad([10, 10], [14, 10], [14, 14], [10, 14]), 100, 80)).toBeNull();
    // Two corners clamped onto the same frame corner.
    expect(fitQuadToFrame(quad([-30, -10], [-10, -30], [90, 70], [10, 70]), 100, 80)).toBeNull();
    expect(fitQuadToFrame(quad([Number.NaN, 0], [90, 0], [90, 70], [10, 70]), 100, 80)).toBeNull();
  });

  it('detect() clips, infer() keeps the raw quad', async () => {
    const partial = cases.find((c) => c.name.startsWith('partial')) as SceneCase;
    const detector = createNnDetector({ model, backend: 'cpu' });
    const raw = await detector.infer(renderScene(partial));
    expect(Math.min(...raw.quad.map((p) => p.x))).toBeLessThan(0);
    const detection = await detector.detect(renderScene(partial));
    for (const p of detection?.quad ?? []) {
      expect(p.x).toBeGreaterThanOrEqual(0);
      expect(p.x).toBeLessThanOrEqual(partial.width);
      expect(p.y).toBeGreaterThanOrEqual(0);
      expect(p.y).toBeLessThanOrEqual(partial.height);
    }
    const unclipped = createNnDetector({ model, backend: 'cpu', clipToFrame: false });
    const rawDetection = await unclipped.detect(renderScene(partial));
    expect(Math.min(...(rawDetection?.quad ?? []).map((p) => p.x))).toBeLessThan(0);
  });

  it('detect() works on frames smaller than the network input (upscaling)', async () => {
    const tiny = cases.find((c) => c.name.startsWith('tiny')) as SceneCase;
    const detector = createNnDetector({ model, backend: 'cpu' });
    expect(await detector.detect(renderScene(tiny))).not.toBeNull();
  });
});

describe('createHybridDetector: network errors', () => {
  const classic = (score: number | null): DocumentDetector => ({
    detect: async () =>
      score === null ? null : { quad: quad([1, 1], [9, 1], [9, 9], [1, 9]), score },
    dispose: () => {},
  });
  const failing: DocumentDetector = {
    detect: async () => {
      throw new Error('network down');
    },
    dispose: () => {},
  };
  const frame = renderScene(cases[0]);

  it('answers with the classic result if it reaches DEFAULT_MIN_SCORE', async () => {
    const onError = vi.fn();
    const score = (DEFAULT_MIN_SCORE + CONFIDENT_SCORE) / 2;
    const hybrid = createHybridDetector({ classic: classic(score), nn: failing, onError });
    expect((await hybrid.detect(frame))?.score).toBe(score);
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: 'network down' }));
  });

  it('answers null below DEFAULT_MIN_SCORE or without a classic quad', async () => {
    for (const score of [DEFAULT_MIN_SCORE - 0.01, null]) {
      const onError = vi.fn();
      const hybrid = createHybridDetector({ classic: classic(score), nn: failing, onError });
      expect(await hybrid.detect(frame)).toBeNull();
      expect(onError).toHaveBeenCalledTimes(1);
    }
  });
});

describe('createGlContext', () => {
  afterEach(() => vi.unstubAllGlobals());
  const gl = { isWebGl2: true };

  it('falls back to a document canvas when OffscreenCanvas has no WebGL2 (Safari 16.4)', () => {
    const element = { width: 0, height: 0, getContext: () => gl };
    vi.stubGlobal(
      'OffscreenCanvas',
      class {
        getContext() {
          return null;
        }
      },
    );
    vi.stubGlobal('document', { createElement: () => element });
    const context = createGlContext();
    expect(context?.canvas).toBe(element);
    expect(context?.gl).toBe(gl);
  });

  it('prefers OffscreenCanvas and survives a throwing getContext', () => {
    vi.stubGlobal(
      'OffscreenCanvas',
      class {
        getContext() {
          return gl;
        }
      },
    );
    expect(createGlContext()?.gl).toBe(gl);
    vi.stubGlobal(
      'OffscreenCanvas',
      class {
        getContext() {
          throw new Error('no');
        }
      },
    );
    expect(createGlContext()).toBeNull(); // no document in a worker
  });
});
