/**
 * The raw DocQuadNet model format (`scanoficus-docquadnet-raw/1`): a JSON graph of NCHW
 * operations whose weights live in one little-endian fp16 blob, and its lowering into a flat
 * list of steps shared by the CPU and WebGL2 runtimes.
 *
 * @module
 */

/** Slice of the fp16 weight blob, in elements. */
export interface WeightRef {
  offset: number;
  length: number;
  shape: number[];
}

/** Activation fused into a convolution or linear layer. */
export type Activation = 'none' | 'relu6' | 'hswish';

/** One node of the raw graph (see `scanoficus_ml/rawgraph.py`). */
export type GraphNode =
  | {
      op: 'conv';
      in: string;
      out: string;
      weight: WeightRef;
      bias: WeightRef;
      stride: number;
      pad: number;
      groups: number;
      act: Activation;
    }
  | { op: 'add'; a: string; b: string; out: string }
  | { op: 'flatten'; in: string; out: string }
  | { op: 'linear'; in: string; out: string; weight: WeightRef; bias: WeightRef; act: Activation }
  | { op: 'sigmoid'; in: string; out: string };

/** The graph JSON (`docquadnet.json`). Unknown fields are ignored. */
export interface ModelGraph {
  format: string;
  dtype: string;
  input: { name: string; shape: number[] };
  nodes: GraphNode[];
  outputs: { heads: string };
}

/** Shape of an activation: channels, height, width (batch 1). */
export type Shape = readonly [c: number, h: number, w: number];

/**
 * A lowered operation. Linear layers become convolutions whose kernel covers the whole input
 * (PyTorch's C-major `flatten` order equals the `[out, C, H, W]` weight layout), so the runtimes
 * only implement convolution, residual add and sigmoid. A residual add that directly follows the
 * convolution producing one of its operands (with no activation) is fused into it.
 */
export type Step =
  | {
      kind: 'conv';
      input: string;
      output: string;
      inShape: Shape;
      outShape: Shape;
      kernel: readonly [kh: number, kw: number];
      stride: number;
      pad: number;
      depthwise: boolean;
      act: Activation;
      weight: WeightRef;
      bias: WeightRef;
      /** Tensor added to the result (fused residual), or null. */
      residual: string | null;
    }
  | { kind: 'add'; a: string; b: string; output: string; shape: Shape }
  | { kind: 'sigmoid'; input: string; output: string; shape: Shape };

/** A loaded model: graph, fp16 weights (raw bits) and its lowered plan. */
export interface NnModel {
  readonly graph: ModelGraph;
  /** fp16 weights as stored (IEEE 754 binary16 bit patterns). */
  readonly weights: Uint16Array;
  /** Square input side (192 for DocQuadNet). */
  readonly size: number;
  /** Input channels (3 = RGB). */
  readonly channels: number;
  readonly steps: readonly Step[];
  /** Name of the output tensor (9 values: 8 corner coordinates and the presence logit). */
  readonly output: string;
  readonly outputShape: Shape;
}

/** The format identifier this runtime understands. */
export const MODEL_FORMAT = 'scanoficus-docquadnet-raw/1';

/** Where to fetch a model from (resolved against the document / worker location). */
export interface ModelSource {
  /** URL of the graph JSON (`docquadnet.json`). */
  graphUrl: string | URL;
  /** URL of the fp16 weight blob (`docquadnet.bin`). */
  weightsUrl: string | URL;
}

/**
 * Fetches and validates a model. The weights are not part of any JavaScript bundle: host
 * `docquadnet.json` and `docquadnet.bin` next to your app (served with compression, see the
 * package README).
 */
export async function loadModel(source: ModelSource, init?: RequestInit): Promise<NnModel> {
  const get = async (url: string | URL) => {
    const response = await fetch(url, init);
    if (!response.ok) throw new Error(`Failed to load ${String(url)}: HTTP ${response.status}`);
    return response;
  };
  const [graph, weights] = await Promise.all([
    get(source.graphUrl).then((r) => r.json() as Promise<unknown>),
    get(source.weightsUrl).then((r) => r.arrayBuffer()),
  ]);
  return parseModel(graph, weights);
}

/** Validates a graph and fp16 weight blob and lowers the graph into runtime steps. */
export function parseModel(graph: unknown, weights: ArrayBuffer | ArrayBufferView): NnModel {
  const g = graph as ModelGraph;
  if (!g || g.format !== MODEL_FORMAT) {
    throw new Error(`Unsupported model format ${JSON.stringify(g?.format)} (need ${MODEL_FORMAT})`);
  }
  if (g.dtype !== 'float16') throw new Error(`Unsupported weight dtype ${g.dtype}`);
  const bytes =
    weights instanceof ArrayBuffer
      ? new Uint8Array(weights)
      : new Uint8Array(weights.buffer, weights.byteOffset, weights.byteLength);
  if (bytes.byteLength % 2) throw new Error('Weight blob has an odd number of bytes');
  // Copy into an aligned buffer and read little-endian regardless of the platform.
  const fp16 = new Uint16Array(bytes.byteLength / 2);
  for (let i = 0; i < fp16.length; i++) fp16[i] = bytes[2 * i] | (bytes[2 * i + 1] << 8);

  const [batch, channels, height, width] = g.input.shape;
  if (batch !== 1 || height !== width) throw new Error('Expected a 1×C×S×S input');
  const shapes = new Map<string, Shape>([[g.input.name, [channels, height, width]]]);
  const shapeOf = (name: string): Shape => {
    const s = shapes.get(name);
    if (!s) throw new Error(`Tensor ${name} is used before it is defined`);
    return s;
  };
  const checkRef = (ref: WeightRef, expected: number[], what: string) => {
    const n = expected.reduce((a, b) => a * b, 1);
    if (ref.length !== n || ref.offset < 0 || ref.offset + ref.length > fp16.length) {
      throw new Error(`Bad ${what} reference (expected ${expected.join('×')})`);
    }
  };
  const steps: Step[] = [];
  const aliases = new Map<string, string>();
  /** Uses of each tensor, to know whether a convolution output may be fused into an add. */
  const uses = new Map<string, number>();
  const use = (name: string) => uses.set(name, (uses.get(name) ?? 0) + 1);
  for (const node of g.nodes) {
    if (node.op === 'add') {
      use(node.a);
      use(node.b);
    } else use(node.in);
  }
  use(g.outputs.heads);

  for (const node of g.nodes) {
    switch (node.op) {
      case 'conv':
      case 'linear': {
        const inShape = shapeOf(node.in);
        const [c, h, w] = inShape;
        let outShape: Shape;
        let kernel: [number, number];
        let depthwise = false;
        let stride = 1;
        let pad = 0;
        if (node.op === 'conv') {
          const [cout, cinGroup, kh, kw] = node.weight.shape;
          depthwise = node.groups !== 1;
          if (depthwise && (node.groups !== c || cout !== c || cinGroup !== 1)) {
            throw new Error('Only depthwise grouped convolutions are supported');
          }
          if (!depthwise && cinGroup !== c) throw new Error(`Channel mismatch at ${node.out}`);
          ({ stride, pad } = node);
          kernel = [kh, kw];
          outShape = [
            cout,
            Math.floor((h + 2 * pad - kh) / stride) + 1,
            Math.floor((w + 2 * pad - kw) / stride) + 1,
          ];
          checkRef(node.weight, [cout, cinGroup, kh, kw], 'weight');
          checkRef(node.bias, [cout], 'bias');
        } else {
          const [cout, cin] = node.weight.shape;
          if (cin !== c * h * w) throw new Error(`Linear input mismatch at ${node.out}`);
          kernel = [h, w];
          outShape = [cout, 1, 1];
          checkRef(node.weight, [cout, cin], 'weight');
          checkRef(node.bias, [cout], 'bias');
        }
        if (!['none', 'relu6', 'hswish'].includes(node.act)) {
          throw new Error(`Unsupported activation ${node.act}`);
        }
        steps.push({
          kind: 'conv',
          input: node.in,
          output: node.out,
          inShape,
          outShape,
          kernel,
          stride,
          pad,
          depthwise,
          act: node.act,
          weight: node.weight,
          bias: node.bias,
          residual: null,
        });
        shapes.set(node.out, outShape);
        break;
      }
      case 'add': {
        const shape = shapeOf(node.a);
        const sb = shapeOf(node.b);
        if (shape.some((v, i) => v !== sb[i])) throw new Error(`Add shape mismatch at ${node.out}`);
        const last = steps[steps.length - 1];
        const fusable =
          last?.kind === 'conv' &&
          last.act === 'none' &&
          last.residual === null &&
          (last.output === node.a || last.output === node.b) &&
          uses.get(last.output) === 1;
        if (fusable) {
          last.residual = last.output === node.b ? node.a : node.b;
          last.output = node.out;
        } else {
          steps.push({ kind: 'add', a: node.a, b: node.b, output: node.out, shape });
        }
        shapes.set(node.out, shape);
        break;
      }
      case 'flatten':
        // A view: linear layers read the [C, H, W] tensor in C-major order directly.
        shapes.set(node.out, shapeOf(node.in));
        aliases.set(node.out, resolve(node.in));
        break;
      case 'sigmoid': {
        const shape = shapeOf(node.in);
        steps.push({ kind: 'sigmoid', input: node.in, output: node.out, shape });
        shapes.set(node.out, shape);
        break;
      }
      default:
        throw new Error(`Unsupported op ${(node as { op: string }).op}`);
    }
  }
  // Replace flatten outputs by the tensors they view.
  for (const step of steps) {
    if (step.kind === 'conv') {
      step.input = resolve(step.input);
      if (step.residual) step.residual = resolve(step.residual);
    } else if (step.kind === 'add') {
      step.a = resolve(step.a);
      step.b = resolve(step.b);
    } else step.input = resolve(step.input);
  }
  const output = resolve(g.outputs.heads);
  return {
    graph: g,
    weights: fp16,
    size: height,
    channels,
    steps,
    output,
    outputShape: shapeOf(g.outputs.heads),
  };

  function resolve(name: string): string {
    return aliases.get(name) ?? name;
  }
}

/** Converts IEEE 754 binary16 bits to a number. */
export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const mantissa = h & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 31) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/** Decodes a weight slice to float32. */
export function weightsF32(model: NnModel, ref: WeightRef): Float32Array {
  const out = new Float32Array(ref.length);
  for (let i = 0; i < ref.length; i++) out[i] = halfToFloat(model.weights[ref.offset + i]);
  return out;
}
