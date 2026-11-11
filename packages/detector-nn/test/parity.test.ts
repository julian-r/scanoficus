import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  bytesToInput,
  CpuRunner,
  createNnDetector,
  decodeHeads,
  halfToFloat,
  parseModel,
  preprocessRgba,
} from '../src/index.ts';
import { renderScene, type SceneCase } from './scene.ts';

const modelDir = new URL('../../../apps/demo/public/models/docquadnet/', import.meta.url);
const graphJson = readFileSync(new URL('docquadnet.json', modelDir), 'utf8');
const weightBytes = readFileSync(new URL('docquadnet.bin', modelDir));
const fixtures = JSON.parse(
  readFileSync(new URL('fixtures/cases.json', import.meta.url), 'utf8'),
) as {
  graphSha256: string;
  weightsSha256: string;
  threshold: number;
  cases: (SceneCase & {
    frameSha256: string;
    inputSha256: string;
    heads: number[];
    presence: number;
    detected: boolean;
    quad: [number, number][];
  })[];
};
const model = parseModel(JSON.parse(graphJson), weightBytes);
const sha256 = (data: Uint8Array | string) => createHash('sha256').update(data).digest('hex');

/** Largest distance between corresponding corners after matching the start corner. */
function quadError(a: { x: number; y: number }[], b: [number, number][]): number {
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

describe('model', () => {
  it('matches the fixture model files', () => {
    expect(sha256(graphJson)).toBe(fixtures.graphSha256);
    expect(sha256(weightBytes)).toBe(fixtures.weightsSha256);
  });

  it('lowers the 44-node graph', () => {
    expect(model.graph.nodes).toHaveLength(44);
    const convs = model.steps.filter((s) => s.kind === 'conv');
    expect(convs).toHaveLength(36); // 34 convolutions + 2 linear layers
    expect(convs.filter((s) => s.depthwise)).toHaveLength(11);
    expect(convs.filter((s) => s.residual)).toHaveLength(7); // every add fused
    expect(model.steps).toHaveLength(36);
    expect(model.outputShape).toEqual([9, 1, 1]);
    expect(model.size).toBe(192);
  });

  it('decodes fp16', () => {
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x7bff)).toBe(65504);
    expect(halfToFloat(0x0001)).toBe(2 ** -24);
    expect(halfToFloat(0x7c00)).toBe(Number.POSITIVE_INFINITY);
  });

  it('rejects unknown formats and out-of-range weights', () => {
    const graph = JSON.parse(graphJson);
    expect(() => parseModel({ ...graph, format: 'x' }, weightBytes)).toThrow(/format/);
    expect(() => parseModel(graph, weightBytes.subarray(0, 1000))).toThrow(/reference/);
  });
});

describe('parity with the NumPy reference', () => {
  const runner = new CpuRunner(model);
  let worst = 0;
  for (const c of fixtures.cases) {
    it(c.name, () => {
      const frame = renderScene(c);
      const rgb = new Uint8Array(frame.width * frame.height * 3);
      for (let i = 0; i < frame.width * frame.height; i++) {
        rgb.set(frame.data.subarray(4 * i, 4 * i + 3), 3 * i);
      }
      expect(sha256(rgb)).toBe(c.frameSha256);
      // Preprocessing is bit-exact with scanoficus_ml.data.preprocess.
      const bytes = preprocessRgba(frame, 192);
      expect(sha256(bytes)).toBe(c.inputSha256);
      const heads = runner.run(bytesToInput(bytes));
      const error = Math.max(...c.heads.map((v, i) => Math.abs(v - heads[i])));
      worst = Math.max(worst, error);
      expect(error).toBeLessThan(1e-3);
      const out = decodeHeads(heads, c.width, c.height);
      expect(Math.abs(out.presence - c.presence)).toBeLessThan(1e-4);
      expect(out.presence > fixtures.threshold).toBe(c.detected);
      expect(quadError([...out.quad], c.quad)).toBeLessThan(0.01);
    });
  }
  it('reports the worst error', () => {
    console.log(
      `CPU vs NumPy: max abs error ${worst.toExponential(2)} over ${fixtures.cases.length} frames`,
    );
    expect(worst).toBeGreaterThan(0);
  });
});

describe('createNnDetector (CPU backend)', () => {
  it('detects the page and rejects an empty frame', async () => {
    const detector = createNnDetector({ model, backend: 'cpu' });
    expect(await detector.ready()).toBe('cpu');
    const page = fixtures.cases[0];
    const detection = await detector.detect(renderScene(page));
    expect(detection).not.toBeNull();
    expect(detection?.score).toBeCloseTo(page.presence, 4);
    expect(quadError([...(detection?.quad ?? [])], page.quad)).toBeLessThan(0.01);
    const empty = fixtures.cases.find((c) => c.name.startsWith('empty')) as SceneCase;
    expect(await detector.detect(renderScene(empty))).toBeNull();
    detector.dispose();
    await expect(detector.detect(renderScene(page))).rejects.toThrow(/disposed/);
  });

  it('applies the threshold', async () => {
    const detector = createNnDetector({ model, backend: 'cpu', threshold: 0.99999 });
    expect(await detector.detect(renderScene(fixtures.cases[0]))).toBeNull();
    const out = await detector.infer(renderScene(fixtures.cases[0]));
    expect(out.presence).toBeGreaterThan(0.99);
    detector.dispose();
  });

  it('rejects a missing model source', async () => {
    const detector = createNnDetector({ backend: 'cpu' });
    await expect(detector.ready()).rejects.toThrow(/graphUrl/);
  });
});
