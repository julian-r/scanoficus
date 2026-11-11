import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { Point, Quad } from '@scanoficus/core/geometry';
import { describe, expect, it } from 'vitest';
import { DOCQUADNET2 } from '../../../apps/demo/src/bundles/docquadnet2.ts';
import {
  buildPyramid,
  CpuRunner,
  cornerFrame,
  cropPlan,
  parseModel,
  preprocessRgba,
  refineCornersCpu,
  rgbaToRgb,
  warpCrop,
} from '../src/index.ts';
import { bytesToInput } from '../src/preprocess.ts';
import { renderScene, type SceneCase } from './scene.ts';

/** Fixtures of scripts/make_refiner_fixtures.py (pipeline.py with the NumPy refiner). */
interface RefinerCase extends SceneCase {
  frameSha256: string;
  main: { heads: number[]; presence: number; rawQuad: [number, number][] };
  work: { width: number; height: number; sha256: string };
  passes: { quad: [number, number][]; scale: number; cropSha256: string; offsets: number[][] }[];
  refined: [number, number][];
  refinedOnnx: [number, number][];
}
const fixtures = JSON.parse(
  readFileSync(new URL('fixtures/refiner.json', import.meta.url), 'utf8'),
) as { scales: number[]; workSide: number; cases: RefinerCase[] };
const read = (url: string | URL | undefined) => readFileSync(fileURLToPath(url as URL));
const load = (source: { graphUrl?: string | URL; weightsUrl?: string | URL }) =>
  parseModel(JSON.parse(read(source.graphUrl).toString()), read(source.weightsUrl));
const main = load(DOCQUADNET2.main);
const refiner = load(DOCQUADNET2.refiner as NonNullable<typeof DOCQUADNET2.refiner>);
const settings = { scales: fixtures.scales, crop: 64, workSide: fixtures.workSide };
const sha256 = (data: Uint8Array) => createHash('sha256').update(data).digest('hex');
const points = (q: [number, number][]): Point[] => q.map(([x, y]) => ({ x, y }));

/** Largest corner distance after matching the start corner (Python orders quads differently). */
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

const frames = new Map<string, ReturnType<typeof renderScene>>();
const frameOf = (c: RefinerCase) => {
  if (!frames.has(c.name)) frames.set(c.name, renderScene(c));
  return frames.get(c.name) as ReturnType<typeof renderScene>;
};

describe('round-2 bundle', () => {
  it('has a batch-4 refiner that maps 3×64×64 crops to 2 offsets', () => {
    expect(refiner.batch).toBe(4);
    expect(refiner.outputShape).toEqual([2, 1, 1]);
    expect(DOCQUADNET2.threshold).toBe(0.98);
    expect(DOCQUADNET2.refiner?.scales).toEqual([0.4, 0.2]);
  });
});

describe('corner refiner parity (CPU vs pipeline.py)', () => {
  const runner = new CpuRunner(refiner);
  const mainRunner = new CpuRunner(main);
  let worstOffset = 0;
  let worstPipeline = 0;
  let worstOnnx = 0;
  for (const c of fixtures.cases) {
    it(c.name, () => {
      const frame = frameOf(c);
      const rgb = { width: c.width, height: c.height, data: rgbaToRgb(frame) };
      expect(sha256(rgb.data)).toBe(c.frameSha256);
      // Main network, round 2.
      const heads = mainRunner.run(bytesToInput(preprocessRgba(frame, 192)));
      expect(Math.max(...c.main.heads.map((v, i) => Math.abs(v - heads[i])))).toBeLessThan(1e-4);
      // Work image (single INTER_AREA resize) is bit-exact.
      const levels = buildPyramid(rgb, settings.workSide);
      expect([levels[0].width, levels[0].height]).toEqual([c.work.width, c.work.height]);
      expect(sha256(levels[0].data)).toBe(c.work.sha256);
      // Each pass: the crops from Python's pass input are bit-exact; offsets within 1e-5.
      for (const pass of c.passes) {
        const q = points(pass.quad);
        const crops = new Uint8Array(4 * 64 * 64 * 3);
        const input = new Float32Array(4 * 3 * 64 * 64);
        for (let n = 0; n < 4; n++) {
          const { level, m } = cropPlan(cornerFrame(q, n, pass.scale), levels.length, 64);
          const out = crops.subarray(n * 64 * 64 * 3, (n + 1) * 64 * 64 * 3);
          warpCrop(levels[level], m, 64, out);
          for (let ch = 0; ch < 3; ch++) {
            for (let i = 0; i < 4096; i++) input[(n * 3 + ch) * 4096 + i] = out[i * 3 + ch] / 255;
          }
        }
        expect(sha256(crops)).toBe(pass.cropSha256);
        const offsets = runner.run(input, 4);
        pass.offsets.forEach(([dx, dy], n) => {
          const e = Math.max(Math.abs(offsets[2 * n] - dx), Math.abs(offsets[2 * n + 1] - dy));
          worstOffset = Math.max(worstOffset, e);
        });
      }
      // Whole refinement from Python's raw quad.
      const result = refineCornersCpu(
        rgb,
        points(c.main.rawQuad) as unknown as Quad,
        settings,
        (input, n) => runner.run(input, n),
      );
      const error = quadError(result.quad, c.refined);
      // Without a document (empty, noise) the refiner chases texture: tiny offset differences
      // of pass 1 change pass-2 crops, so only documents get the tight tolerance.
      const document = c.main.presence > 0.5;
      if (document) {
        worstPipeline = Math.max(worstPipeline, error);
        worstOnnx = Math.max(worstOnnx, quadError(result.quad, c.refinedOnnx));
      }
      expect(error).toBeLessThan(document ? 0.01 : 0.25);
      expect(result.passes.map((p) => p.scale)).toEqual([0.4, 0.2]);
    });
  }
  it('reports the worst errors', () => {
    console.log(
      `refiner CPU vs NumPy: offsets ${worstOffset.toExponential(2)} half-crop units, ` +
        `refined quad (documents) ${worstPipeline.toExponential(2)} px; vs ONNX pipeline ` +
        `${worstOnnx.toFixed(4)} px`,
    );
    expect(worstOffset).toBeLessThan(1e-5);
  });
});
