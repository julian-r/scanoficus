import { type Activation, type NnModel, type Step, weightsF32 } from './model.ts';

/**
 * Plain TypeScript interpreter of a lowered model (float32 storage, CHW). Slow (≈ 100 ms per
 * frame) but dependency-free; it is the fallback without WebGL2 and the reference of the tests.
 */
export class CpuRunner {
  private readonly model: NnModel;
  private readonly params: ({ weight: Float32Array; bias: Float32Array } | null)[];

  constructor(model: NnModel) {
    this.model = model;
    this.params = model.steps.map((s) =>
      s.kind === 'conv'
        ? { weight: weightsF32(model, s.weight), bias: weightsF32(model, s.bias) }
        : null,
    );
  }

  /**
   * Runs the network on `batch` inputs of `C×H×W` (NCHW) and returns the outputs one after the
   * other (9 values per frame for the main network, 2 per crop for the refiner).
   */
  run(input: Float32Array, batch = 1): Float32Array {
    const { model } = this;
    const size = model.channels * model.height * model.width;
    if (input.length !== batch * size)
      throw new RangeError(`Expected ${batch * size} input values`);
    const outputs: Float32Array[] = [];
    for (let n = 0; n < batch; n++) {
      const env = new Map<string, Float32Array>([
        [model.graph.input.name, input.subarray(n * size, (n + 1) * size)],
      ]);
      const get = (name: string) => env.get(name) as Float32Array;
      model.steps.forEach((step, i) => {
        env.set(step.output, runStep(step, this.params[i], get));
      });
      outputs.push(get(model.output));
    }
    if (batch === 1) return outputs[0];
    const out = new Float32Array(outputs.reduce((a, o) => a + o.length, 0));
    let offset = 0;
    for (const o of outputs) {
      out.set(o, offset);
      offset += o.length;
    }
    return out;
  }
}

function runStep(
  step: Step,
  params: { weight: Float32Array; bias: Float32Array } | null,
  get: (name: string) => Float32Array,
): Float32Array {
  if (step.kind === 'add') {
    const a = get(step.a);
    const b = get(step.b);
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    return out;
  }
  if (step.kind === 'sigmoid') {
    const a = get(step.input);
    return a.map((v) => 1 / (1 + Math.exp(-v)));
  }
  const { weight, bias } = params as { weight: Float32Array; bias: Float32Array };
  const input = get(step.input);
  const [cin, ih, iw] = step.inShape;
  const [cout, oh, ow] = step.outShape;
  const [kh, kw] = step.kernel;
  const { stride, pad } = step;
  const plane = oh * ow;
  const out = new Float32Array(cout * plane);
  for (let o = 0; o < cout; o++) out.fill(bias[o], o * plane, (o + 1) * plane);
  const pointwise = kh === 1 && kw === 1 && stride === 1 && pad === 0;
  for (let o = 0; o < cout; o++) {
    const dst = o * plane;
    const groupIn = step.depthwise ? [o] : null;
    const count = groupIn ? 1 : cin;
    for (let j = 0; j < count; j++) {
      const c = groupIn ? o : j;
      const src = c * ih * iw;
      const wBase = (o * count + j) * kh * kw;
      if (pointwise) {
        const w = weight[wBase];
        for (let p = 0; p < plane; p++) out[dst + p] += w * input[src + p];
        continue;
      }
      for (let ky = 0; ky < kh; ky++) {
        for (let kx = 0; kx < kw; kx++) {
          const w = weight[wBase + ky * kw + kx];
          for (let oy = 0; oy < oh; oy++) {
            const y = oy * stride + ky - pad;
            if (y < 0 || y >= ih) continue;
            const row = src + y * iw;
            const orow = dst + oy * ow;
            for (let ox = 0; ox < ow; ox++) {
              const x = ox * stride + kx - pad;
              if (x >= 0 && x < iw) out[orow + ox] += w * input[row + x];
            }
          }
        }
      }
    }
  }
  if (step.residual) {
    const r = get(step.residual);
    for (let i = 0; i < out.length; i++) out[i] += r[i];
  }
  activate(out, step.act);
  return out;
}

function activate(x: Float32Array, act: Activation): void {
  if (act === 'relu6') {
    for (let i = 0; i < x.length; i++) x[i] = Math.min(6, Math.max(0, x[i]));
  } else if (act === 'hswish') {
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      x[i] = (v * Math.min(6, Math.max(0, v + 3))) / 6;
    }
  }
}
