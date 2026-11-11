import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { Point } from '@scanoficus/core/geometry';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DOCQUADNET2 } from '../../../apps/demo/src/bundles/docquadnet2.ts';
import {
  createNnDetector,
  loadBundle,
  type NnBundle,
  type PresenceSignals,
  parseModel,
} from '../src/index.ts';
import { renderScene, type SceneCase } from './scene.ts';

interface Case extends SceneCase {
  main: { presence: number; rawQuad: [number, number][] };
  refined: [number, number][];
}
const fixtures = JSON.parse(
  readFileSync(new URL('fixtures/refiner.json', import.meta.url), 'utf8'),
) as { cases: Case[] };
const byName = (name: string) => fixtures.cases.find((c) => c.name === name) as Case;
const file = (url: string | URL | undefined) => readFileSync(fileURLToPath(url as URL));
const refinerSource = DOCQUADNET2.refiner as NonNullable<typeof DOCQUADNET2.refiner>;
const bundle: NnBundle = {
  name: 'docquadnet2',
  threshold: DOCQUADNET2.threshold as number,
  decision: null,
  main: parseModel(
    JSON.parse(file(DOCQUADNET2.main.graphUrl).toString()),
    file(DOCQUADNET2.main.weightsUrl),
  ),
  refiner: {
    model: parseModel(
      JSON.parse(file(refinerSource.graphUrl).toString()),
      file(refinerSource.weightsUrl),
    ),
    scales: refinerSource.scales,
    crop: refinerSource.crop,
    workSide: refinerSource.workSide,
  },
};

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

describe('round-2 pipeline (CPU backend)', () => {
  it('detectStill runs the refiner (full pipeline vs pipeline.py)', async () => {
    const detector = createNnDetector({ bundle, backend: 'cpu', clipToFrame: false });
    for (const name of ['page-640x480', 'tilted-1280x720', 'large-2000x1500']) {
      const c = byName(name);
      const detection = await detector.detectStill(renderScene(c));
      expect(detection?.score).toBeCloseTo(c.main.presence, 4);
      expect(quadError(detection?.quad ?? [], c.refined)).toBeLessThan(0.02);
    }
  });

  it('serializes concurrent calls (each frame refined on its own pixels)', async () => {
    const detector = createNnDetector({ bundle, backend: 'cpu', clipToFrame: false });
    const a = byName('page-640x480');
    const b = byName('partial-500x375');
    const order: string[] = [];
    const [da, db] = await Promise.all([
      detector.detectStill(renderScene(a)).finally(() => order.push('a')),
      detector.detectStill(renderScene(b)).finally(() => order.push('b')),
    ]);
    expect(quadError(da?.quad ?? [], a.refined)).toBeLessThan(0.02);
    expect(quadError(db?.quad ?? [], b.refined)).toBeLessThan(0.02);
    expect(order).toEqual(['a', 'b']);
  });

  it('live detect() returns the raw quad unless live refinement is on', async () => {
    const c = byName('page-640x480');
    const live = await createNnDetector({ bundle, backend: 'cpu' }).detect(renderScene(c));
    expect(quadError(live?.quad ?? [], c.main.rawQuad)).toBeLessThan(0.01);
    const refined = await createNnDetector({
      bundle,
      backend: 'cpu',
      refiner: { live: true },
    }).detect(renderScene(c));
    expect(quadError(refined?.quad ?? [], c.refined)).toBeLessThan(0.02);
    const off = await createNnDetector({ bundle, backend: 'cpu', refiner: false }).detectStill(
      renderScene(c),
    );
    expect(quadError(off?.quad ?? [], c.main.rawQuad)).toBeLessThan(0.01);
  });

  it('uses the bundle threshold (0.98 for round 2)', async () => {
    const tiny = byName('tiny-160x120'); // presence 0.68
    expect(await createNnDetector({ bundle, backend: 'cpu' }).detect(renderScene(tiny))).toBeNull();
    const lenient = createNnDetector({ bundle, backend: 'cpu', threshold: 0.5 });
    expect(await lenient.detect(renderScene(tiny))).not.toBeNull();
  });

  it('calls a custom presence rule with all signals, after the refiner', async () => {
    const seen: PresenceSignals[] = [];
    const c = byName('page-640x480');
    const detector = createNnDetector({
      bundle,
      backend: 'cpu',
      presence: (s) => {
        seen.push(s);
        return s.refinedQuad ? 0.999 : 0;
      },
    });
    const detection = await detector.detectStill(renderScene(c));
    expect(detection?.score).toBe(0.999);
    const s = seen[0];
    expect(s.still).toBe(true);
    expect(s.threshold).toBe(0.98);
    expect(s.logit).toBeCloseTo(Math.log(c.main.presence / (1 - c.main.presence)), 3);
    expect(s.refiner?.map((p) => p.scale)).toEqual([0.4, 0.2]);
    expect(s.refiner?.[0].offsets).toHaveLength(4);
    expect(quadError(s.rawQuad, c.main.rawQuad)).toBeLessThan(0.01);
    expect(s.frame.width).toBe(640);
    // The rule also rejects: no refined quad on live frames → score 0.
    expect(await detector.detect(renderScene(c))).toBeNull();
    // Rules run for frames below the threshold too (the empty frame).
    seen.length = 0;
    expect(await detector.detectStill(renderScene(byName('empty-192x192')))).not.toBeNull();
    expect(seen).toHaveLength(1);
  });
});

describe('loadBundle', () => {
  afterEach(() => vi.unstubAllGlobals());
  const dir = new URL('../../../apps/demo/models/docquadnet2/', import.meta.url);
  const serve = (override: Record<string, Uint8Array> = {}) => {
    vi.stubGlobal('fetch', async (url: URL | string) => {
      const name = String(url).replace('https://cdn.test/m/', '');
      try {
        return new Response((override[name] ?? readFileSync(new URL(name, dir))) as BodyInit);
      } catch {
        return new Response(null, { status: 404 });
      }
    });
  };

  it('loads a self-hosted bundle.json with the refiner and checks every weights file', async () => {
    serve();
    const loaded = await loadBundle('https://cdn.test/m/bundle.json');
    expect(loaded.name).toBe('docquadnet2');
    expect(loaded.threshold).toBe(0.98);
    expect(loaded.refiner?.model.batch).toBe(4);
    expect(loaded.refiner?.scales).toEqual([0.4, 0.2]);
    const graph = JSON.parse(readFileSync(new URL('docquadnet2-refiner.json', dir), 'utf8'));
    const tampered = new Uint8Array(readFileSync(new URL(graph.weights.file, dir)));
    tampered[100] ^= 1;
    serve({ [graph.weights.file]: tampered });
    await expect(loadBundle('https://cdn.test/m/bundle.json')).rejects.toThrow(/SHA-256/);
  });
});
