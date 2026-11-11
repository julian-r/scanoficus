import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { CpuRunner, parseModel } from '../src/index.ts';

/** Tiny random-weight graphs from scripts/make_synthetic.py with NumPy reference outputs. */
const load = (name: string) => {
  const dir = new URL(`fixtures/synthetic/${name}`, import.meta.url);
  const graph = JSON.parse(readFileSync(`${dir.pathname}.json`, 'utf8'));
  return {
    model: parseModel(graph, readFileSync(`${dir.pathname}.bin`)),
    input: Float32Array.from(graph.test.input as number[]),
    output: graph.test.output as number[],
  };
};

describe('synthetic graphs (CPU vs NumPy)', () => {
  it('odd-mix lowers to an unfused add, a fused residual and a sigmoid', () => {
    const { model } = load('odd-mix');
    expect(model.height).toBe(13);
    expect(model.width).toBe(9);
    expect(model.steps.filter((s) => s.kind === 'add')).toHaveLength(1);
    expect(model.steps.filter((s) => s.kind === 'conv' && s.residual)).toHaveLength(1);
    expect(model.steps.filter((s) => s.kind === 'sigmoid')).toHaveLength(1);
    expect(model.outputShape).toEqual([3, 1, 1]);
  });

  for (const name of ['odd-mix', 'two-channels']) {
    it(name, () => {
      const { model, input, output } = load(name);
      const result = new CpuRunner(model).run(input);
      expect(result).toHaveLength(output.length);
      output.forEach((v, i) => {
        expect(Math.abs(result[i] - v)).toBeLessThan(1e-5 * Math.max(1, Math.abs(v)));
      });
    });
  }
});
