import type { RgbaImage } from '@scanoficus/core/image';
import type { NnModel, Shape, Step } from './model.ts';
import { linearAreaTable } from './preprocess.ts';

/**
 * Minimal WebGL2 inference runtime for the lowered DocQuadNet graph.
 *
 * Layout: a `C × H × W` activation is one texture of `W × (H · ⌈C / 4⌉)` texels; texel
 * `(x, s · H + y)` holds channels `4s … 4s + 3` at `(y, x)` (padding channels are 0). Every step
 * is one full-screen draw into the output texture's framebuffer: dense convolution (also used for
 * the linear layers, whose kernel covers the whole input), depthwise convolution (both with
 * fused bias, residual add and activation), elementwise add / sigmoid, and the preprocessing
 * (2×2 box halvings, then an area resize to the input size). Weights are fp16 textures holding
 * the exact stored values; arithmetic is float32 (`highp`); activations are stored as float32
 * or float16 textures. Only the final 9 outputs are read back.
 */

/** Storage precision of activations. */
export type GlPrecision = 'float32' | 'float16';

/** Frames the runtime can upload as a texture. */
export type GlFrame = ImageBitmap | RgbaImage | HTMLCanvasElement | OffscreenCanvas;

/** Thrown while the WebGL context is lost; the runtime rebuilds itself after a restore. */
export class ContextLostError extends Error {
  constructor() {
    super('WebGL context lost');
    this.name = 'ContextLostError';
  }
}

const VERTEX = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out vec4 o;
`;

const ACTIVATE = `
vec4 activate(vec4 v, int act) {
  if (act == 1) return clamp(v, 0.0, 6.0);
  if (act == 2) return v * clamp(v + 3.0, 0.0, 6.0) / 6.0;
  return v;
}`;

/** Dense convolution: weights texel (((ky·kw + kx)·inSlices + k)·4 + j, s), bias in the last column. */
const CONV = `${HEADER}${ACTIVATE}
uniform sampler2D u_in, u_w, u_res;
uniform ivec2 u_inSize, u_outSize, u_kernel;
uniform int u_inSlices, u_stride, u_pad, u_act, u_residual;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int s = p.y / u_outSize.y;
  int oy = p.y - s * u_outSize.y;
  int ox = p.x;
  vec4 acc = texelFetch(u_w, ivec2(u_kernel.x * u_kernel.y * u_inSlices * 4, s), 0);
  for (int ky = 0; ky < u_kernel.y; ky++) {
    int iy = oy * u_stride + ky - u_pad;
    if (iy < 0 || iy >= u_inSize.y) continue;
    for (int kx = 0; kx < u_kernel.x; kx++) {
      int ix = ox * u_stride + kx - u_pad;
      if (ix < 0 || ix >= u_inSize.x) continue;
      int c = (ky * u_kernel.x + kx) * u_inSlices * 4;
      for (int k = 0; k < u_inSlices; k++, c += 4) {
        vec4 v = texelFetch(u_in, ivec2(ix, k * u_inSize.y + iy), 0);
        acc += mat4(
          texelFetch(u_w, ivec2(c, s), 0),
          texelFetch(u_w, ivec2(c + 1, s), 0),
          texelFetch(u_w, ivec2(c + 2, s), 0),
          texelFetch(u_w, ivec2(c + 3, s), 0)) * v;
      }
    }
  }
  if (u_residual == 1) acc += texelFetch(u_res, p, 0);
  o = activate(acc, u_act);
}`;

/** Depthwise convolution: weights texel (ky·kw + kx, s), bias in the last column. */
const DEPTHWISE = `${HEADER}${ACTIVATE}
uniform sampler2D u_in, u_w, u_res;
uniform ivec2 u_inSize, u_outSize, u_kernel;
uniform int u_stride, u_pad, u_act, u_residual;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int s = p.y / u_outSize.y;
  int oy = p.y - s * u_outSize.y;
  int ox = p.x;
  vec4 acc = texelFetch(u_w, ivec2(u_kernel.x * u_kernel.y, s), 0);
  for (int ky = 0; ky < u_kernel.y; ky++) {
    int iy = oy * u_stride + ky - u_pad;
    if (iy < 0 || iy >= u_inSize.y) continue;
    for (int kx = 0; kx < u_kernel.x; kx++) {
      int ix = ox * u_stride + kx - u_pad;
      if (ix < 0 || ix >= u_inSize.x) continue;
      acc += texelFetch(u_w, ivec2(ky * u_kernel.x + kx, s), 0) *
        texelFetch(u_in, ivec2(ix, s * u_inSize.y + iy), 0);
    }
  }
  if (u_residual == 1) acc += texelFetch(u_res, p, 0);
  o = activate(acc, u_act);
}`;

const ELEMENTWISE = `${HEADER}
uniform sampler2D u_a, u_b;
uniform int u_mode;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 a = texelFetch(u_a, p, 0);
  o = u_mode == 0 ? a + texelFetch(u_b, p, 0) : 1.0 / (1.0 + exp(-a));
}`;

/** Exact 2×2 box mean of 8-bit values, rounded half up (OpenCV's INTER_AREA 2× path). */
const HALVE = `${HEADER}
uniform sampler2D u_src;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) * 2;
  vec4 s = texelFetch(u_src, p, 0) + texelFetch(u_src, p + ivec2(1, 0), 0) +
    texelFetch(u_src, p + ivec2(0, 1), 0) + texelFetch(u_src, p + ivec2(1, 1), 0);
  o = floor((floor(s * 255.0 + 0.5) + 2.0) / 4.0) / 255.0;
}`;

/**
 * Resize to the network input, rounded to 8-bit levels: area (fractional coverage, round half
 * to even) when both
 * axes shrink, else OpenCV's bilinear emulation of INTER_AREA in its exact integer arithmetic,
 * with the per-axis tables of `linearAreaTable` (row 0: x, row 1: y).
 */
const RESIZE = `${HEADER}
precision highp isampler2D;
uniform sampler2D u_src;
uniform isampler2D u_tab;
uniform ivec2 u_srcSize;
uniform vec2 u_scale;
uniform int u_linear;
ivec3 px(int x, int y) {
  return ivec3(floor(texelFetch(u_src, ivec2(x, y), 0).rgb * 255.0 + 0.5));
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (u_linear == 1) {
    ivec4 tx = texelFetch(u_tab, ivec2(p.x, 0), 0);
    ivec4 ty = texelFetch(u_tab, ivec2(p.y, 1), 0);
    ivec3 h0 = px(tx.x, ty.x) * tx.y + px(tx.w, ty.x) * tx.z;
    ivec3 h1 = px(tx.x, ty.w) * tx.y + px(tx.w, ty.w) * tx.z;
    ivec3 v = (((ty.y * (h0 >> 4)) >> 16) + ((ty.z * (h1 >> 4)) >> 16) + 2) >> 2;
    o = vec4(vec3(clamp(v, 0, 255)) / 255.0, 0.0);
    return;
  }
  vec2 lo = vec2(p) * u_scale;
  vec2 hi = min(lo + u_scale, vec2(u_srcSize));
  ivec2 start = ivec2(floor(lo));
  ivec2 end = min(ivec2(ceil(hi)), u_srcSize);
  vec3 acc = vec3(0.0);
  for (int y = start.y; y < end.y; y++) {
    float wy = min(float(y + 1), hi.y) - max(float(y), lo.y);
    vec3 row = vec3(0.0);
    for (int x = start.x; x < end.x; x++) {
      float wx = min(float(x + 1), hi.x) - max(float(x), lo.x);
      row += wx * vec3(px(x, y));
    }
    acc += wy * row;
  }
  vec2 cell = hi - lo;
  // Integer levels in, round half to even out: exact for integer factors (e.g. 4:1, where
  // OpenCV's scalar path rounds the float mean half to even).
  o = vec4(roundEven(acc / (cell.x * cell.y)) / 255.0, 0.0);
}`;

interface Target {
  texture: WebGLTexture;
  framebuffer: WebGLFramebuffer;
  width: number;
  height: number;
}

interface Programs {
  conv: WebGLProgram;
  depthwise: WebGLProgram;
  elementwise: WebGLProgram;
  halve: WebGLProgram;
  resize: WebGLProgram;
}

/** WebGL2 capabilities relevant to the runtime. */
export interface GlSupport {
  /** Activation precisions that can be rendered to, best first. */
  precisions: GlPrecision[];
  maxTextureSize: number;
}

/** Probes a context: float32 needs `EXT_color_buffer_float`, float16 also `EXT_color_buffer_half_float`. */
export function probeGl(gl: WebGL2RenderingContext): GlSupport {
  const precisions: GlPrecision[] = [];
  const full = gl.getExtension('EXT_color_buffer_float');
  if (full && renderable(gl, gl.RGBA32F)) precisions.push('float32');
  if ((full || gl.getExtension('EXT_color_buffer_half_float')) && renderable(gl, gl.RGBA16F)) {
    precisions.push('float16');
  }
  return { precisions, maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number };
}

function renderable(gl: WebGL2RenderingContext, format: number): boolean {
  const texture = gl.createTexture();
  const framebuffer = gl.createFramebuffer();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texStorage2D(gl.TEXTURE_2D, 1, format, 4, 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(framebuffer);
  gl.deleteTexture(texture);
  return ok;
}

/** Options for {@link GlRunner}. */
export interface GlRunnerOptions {
  /** Activation storage: `'auto'` (default) prefers float32, then float16. */
  precision?: 'auto' | GlPrecision;
  /**
   * Canvas to create the context on. Default: a new 1×1 `OffscreenCanvas`, or a `<canvas>` where
   * `OffscreenCanvas` has no WebGL2 (Safari 16.4–16.x) and a document exists.
   */
  canvas?: OffscreenCanvas | HTMLCanvasElement;
}

const CONTEXT_ATTRIBUTES: WebGLContextAttributes = {
  alpha: false,
  antialias: false,
  depth: false,
  stencil: false,
  premultipliedAlpha: false,
  preserveDrawingBuffer: false,
};

/**
 * Creates a WebGL2 context on `canvas`, or on a new canvas: an `OffscreenCanvas` first, then a
 * document `<canvas>` if that gives no WebGL2 context. Returns `null` without WebGL2.
 */
export function createGlContext(
  canvas?: OffscreenCanvas | HTMLCanvasElement,
): { canvas: OffscreenCanvas | HTMLCanvasElement; gl: WebGL2RenderingContext } | null {
  const attempt = (c: OffscreenCanvas | HTMLCanvasElement) => {
    try {
      const gl = c.getContext('webgl2', CONTEXT_ATTRIBUTES) as WebGL2RenderingContext | null;
      return gl ? { canvas: c, gl } : null;
    } catch {
      return null;
    }
  };
  if (canvas) return attempt(canvas);
  if (typeof OffscreenCanvas !== 'undefined') {
    const found = attempt(new OffscreenCanvas(1, 1));
    if (found) return found;
  }
  if (typeof document !== 'undefined') {
    const element = document.createElement('canvas');
    element.width = element.height = 1;
    return attempt(element);
  }
  return null;
}

/**
 * Runs a model with WebGL2. The constructor compiles the shaders, uploads the weights (the
 * cold start) and runs a self-test: GL errors, framebuffer completeness and, if the graph has a
 * `selfTest` record, one inference on its constant input against the stored outputs. It throws
 * (releasing everything it created) if WebGL2 or a renderable float format is missing or the
 * self-test fails, so callers can fall back to the CPU instead of getting wrong answers.
 *
 * Context loss: while the context is lost, {@link run} throws {@link ContextLostError}; after
 * the browser restores the context, the next run rebuilds all GPU resources from the model kept
 * in memory. {@link dispose} releases every GPU resource and the context.
 */
export class GlRunner {
  readonly precision: GlPrecision;
  private readonly model: NnModel;
  private readonly canvas: OffscreenCanvas | HTMLCanvasElement;
  private readonly ownsCanvas: boolean;
  private gl: WebGL2RenderingContext;
  /** `lost`: the GPU objects are gone and must be rebuilt once the context is back. */
  private state: 'ready' | 'lost' | 'disposed' = 'ready';
  private programs!: Programs;
  private uniforms = new Map<WebGLProgram, Map<string, WebGLUniformLocation | null>>();
  private weights: WebGLTexture[] = [];
  /** Activation targets and which one each tensor uses. */
  private targets: Target[] = [];
  private tensorTarget = new Map<string, number>();
  private frameTexture: WebGLTexture | null = null;
  private levels: Target[] = [];
  /**
   * Integer coefficient table of the upscaling resize and the sizes it was made for. Always
   * bound (a 1×1 placeholder at first): an integer sampler left on the unit of `u_src` would
   * make every resize draw fail with INVALID_OPERATION.
   */
  private table: { texture: WebGLTexture; key: string } | null = null;
  /** Frame sizes whose preprocessing was checked for GL errors. */
  private checkedSizes = new Set<string>();
  private readonly maxTextureSize: number;
  private readonly onLost = (event: Event) => {
    event.preventDefault(); // allows the browser to restore the context
    if (this.state !== 'disposed') this.state = 'lost';
  };

  constructor(model: NnModel, options: GlRunnerOptions = {}) {
    this.model = model;
    this.ownsCanvas = !options.canvas;
    const context = createGlContext(options.canvas);
    if (!context) throw new Error('WebGL2 is not available');
    this.canvas = context.canvas;
    const gl = context.gl;
    this.gl = gl;
    try {
      const support = probeGl(gl);
      const wanted = options.precision ?? 'auto';
      const precision = wanted === 'auto' ? support.precisions[0] : wanted;
      if (!precision || !support.precisions.includes(precision)) {
        throw new Error(
          `WebGL2 cannot render to ${wanted === 'auto' ? 'float' : wanted} textures ` +
            '(needs EXT_color_buffer_float or EXT_color_buffer_half_float)',
        );
      }
      this.precision = precision;
      this.maxTextureSize = support.maxTextureSize;
      this.canvas.addEventListener('webglcontextlost', this.onLost as EventListener);
      this.build();
    } catch (error) {
      // Leave nothing behind: GPU objects, the listener and (if ours) the context.
      this.canvas.removeEventListener('webglcontextlost', this.onLost as EventListener);
      if (!gl.isContextLost()) this.release();
      if (this.ownsCanvas) gl.getExtension('WEBGL_lose_context')?.loseContext();
      this.state = 'disposed';
      throw error;
    }
  }

  /** True while the context is lost (runs throw {@link ContextLostError}). */
  get lost(): boolean {
    return this.gl.isContextLost();
  }

  /** Largest frame side that can be uploaded as a texture. */
  get maxFrameSize(): number {
    return this.maxTextureSize;
  }

  /** Preprocesses `frame` on the GPU and runs the network. Returns the 9 raw outputs. */
  run(frame: GlFrame): Float32Array {
    this.ensure();
    this.preprocess(frame);
    return this.execute();
  }

  /** Runs the network on an already preprocessed `C × H × W` input (CHW). */
  runTensor(input: Float32Array): Float32Array {
    this.ensure();
    return this.runTensorUnchecked(input);
  }

  private runTensorUnchecked(input: Float32Array): Float32Array {
    const { width, height, channels } = this.model;
    const plane = width * height;
    if (channels > 4) throw new RangeError('Inputs with more than 4 channels are not supported');
    if (input.length !== channels * plane) throw new RangeError('Bad input length');
    const texels = new Float32Array(plane * 4);
    for (let c = 0; c < channels; c++) {
      for (let i = 0; i < plane; i++) texels[i * 4 + c] = input[c * plane + i];
    }
    const gl = this.gl;
    const target = this.target(this.model.graph.input.name);
    gl.bindTexture(gl.TEXTURE_2D, target.texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.FLOAT, texels);
    return this.execute();
  }

  /** Reads back the last network input as 8-bit CHW planes (debugging and tests). */
  readInput(): Uint8Array {
    this.ensure();
    const { width, height, channels } = this.model;
    const plane = width * height;
    const texels = this.read(this.target(this.model.graph.input.name), width, height);
    const out = new Uint8Array(channels * plane);
    for (let c = 0; c < channels; c++) {
      for (let i = 0; i < plane; i++) out[c * plane + i] = Math.round(texels[i * 4 + c] * 255);
    }
    return out;
  }

  /** Releases all GPU resources (and the context if the runner created the canvas). */
  dispose(): void {
    if (this.state === 'disposed') return;
    const usable = this.state === 'ready' && !this.gl.isContextLost();
    this.state = 'disposed';
    this.canvas.removeEventListener('webglcontextlost', this.onLost as EventListener);
    if (usable) this.release();
    if (this.ownsCanvas) {
      this.gl.getExtension('WEBGL_lose_context')?.loseContext();
      this.canvas.width = this.canvas.height = 1;
    }
  }

  private ensure(): void {
    if (this.state === 'disposed') throw new Error('Runner disposed');
    if (this.gl.isContextLost()) {
      this.state = 'lost';
      throw new ContextLostError();
    }
    if (this.state === 'lost') {
      // All objects of the lost context are invalid: rebuild from scratch.
      this.uniforms.clear();
      this.weights = [];
      this.targets = [];
      this.levels = [];
      this.frameTexture = null;
      this.table = null;
      probeGl(this.gl); // re-enables the float render extensions
      this.build();
      this.state = 'ready';
    }
  }

  /**
   * Compiles the programs, uploads the weights, allocates the activation textures and runs the
   * self-test; throws on any failure.
   */
  private build(): void {
    const gl = this.gl;
    gl.getError(); // clear stale errors
    this.programs = {
      conv: link(gl, CONV),
      depthwise: link(gl, DEPTHWISE),
      elementwise: link(gl, ELEMENTWISE),
      halve: link(gl, HALVE),
      resize: link(gl, RESIZE),
    };
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    for (const step of this.model.steps) {
      if (step.kind === 'conv') this.weights.push(this.uploadWeights(step));
    }
    this.table = { texture: this.uploadTable(new Int32Array(4), 1), key: '' };
    this.checkedSizes.clear();
    this.allocate();
    this.selfTest();
  }

  /**
   * Checks that the GPU path works: no GL error, and the stored outputs for a constant input are
   * reproduced (relative tolerance 1e-3 for float32 storage, 1e-2 for float16).
   */
  private selfTest(): void {
    const gl = this.gl;
    const check = (what: string) => {
      const error = gl.getError();
      if (error !== gl.NO_ERROR && !gl.isContextLost()) {
        throw new Error(`WebGL2 self-test failed: GL error 0x${error.toString(16)} ${what}`);
      }
    };
    check('during initialization');
    const test = this.model.graph.selfTest;
    if (!test) return;
    const { width, height, channels } = this.model;
    const out = this.runTensorUnchecked(
      new Float32Array(channels * width * height).fill(test.input),
    );
    check('during the test inference');
    const tolerance = this.precision === 'float32' ? 1e-3 : 1e-2;
    test.heads.forEach((expected, i) => {
      const error = Math.abs(out[i] - expected);
      if (!(error <= tolerance * Math.max(1, Math.abs(expected)))) {
        throw new Error(
          `WebGL2 self-test failed (${this.precision}): output ${i} is ${out[i]}, expected ` +
            `${expected}`,
        );
      }
    });
  }

  private release(): void {
    const gl = this.gl;
    for (const p of Object.values(this.programs ?? {})) gl.deleteProgram(p);
    for (const t of this.weights) gl.deleteTexture(t);
    for (const t of [...this.targets, ...this.levels]) {
      gl.deleteFramebuffer(t.framebuffer);
      gl.deleteTexture(t.texture);
    }
    if (this.frameTexture) gl.deleteTexture(this.frameTexture);
    if (this.table) gl.deleteTexture(this.table.texture);
    this.weights = [];
    this.targets = [];
    this.levels = [];
    this.frameTexture = null;
    this.table = null;
    this.uniforms.clear();
  }

  /** Uploads an integer coefficient table (`n × 2` texels of RGBA32I). */
  private uploadTable(data: Int32Array, n: number): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const rows = data.length / (4 * n);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32I, n, rows, 0, gl.RGBA_INTEGER, gl.INT, data);
    setNearest(gl);
    return texture;
  }

  /** Packs a convolution's fp16 weights and bias into one RGBA16F texture (exact values). */
  private uploadWeights(step: Extract<Step, { kind: 'conv' }>): WebGLTexture {
    const gl = this.gl;
    const w = this.model.weights;
    const [cin] = step.inShape;
    const [cout] = step.outShape;
    const [kh, kw] = step.kernel;
    const taps = kh * kw;
    const outSlices = Math.ceil(cout / 4);
    const inSlices = Math.ceil(cin / 4);
    const width = step.depthwise ? taps + 1 : taps * inSlices * 4 + 1;
    if (width > this.maxTextureSize) throw new Error('Layer too large for MAX_TEXTURE_SIZE');
    const data = new Uint16Array(width * outSlices * 4);
    const { offset } = step.weight;
    for (let o = 0; o < cout; o++) {
      const s = o >> 2;
      const lane = o & 3;
      const row = s * width;
      if (step.depthwise) {
        for (let t = 0; t < taps; t++) data[(row + t) * 4 + lane] = w[offset + o * taps + t];
      } else {
        for (let i = 0; i < cin; i++) {
          for (let t = 0; t < taps; t++) {
            const col = (t * inSlices + (i >> 2)) * 4 + (i & 3);
            data[(row + col) * 4 + lane] = w[offset + (o * cin + i) * taps + t];
          }
        }
      }
      data[(row + width - 1) * 4 + lane] = w[step.bias.offset + o];
    }
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, width, outSlices);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, outSlices, gl.RGBA, gl.HALF_FLOAT, data);
    setNearest(gl);
    return texture;
  }

  /**
   * Assigns activation textures with a static liveness plan: a tensor's texture returns to the
   * pool (per texture size) after its last use, so peak memory is a few tensors, not all 40.
   */
  private allocate(): void {
    const { steps, graph, width, height, channels } = this.model;
    const lastUse = new Map<string, number>();
    steps.forEach((step, i) => {
      for (const name of inputsOf(step)) lastUse.set(name, i);
    });
    lastUse.set(this.model.output, steps.length);
    const free = new Map<string, number[]>();
    const acquire = (name: string, shape: Shape) => {
      const width = shape[2];
      const height = shape[1] * Math.ceil(shape[0] / 4);
      const key = `${width}x${height}`;
      const index = free.get(key)?.pop() ?? this.targets.push(this.createTarget(width, height)) - 1;
      this.tensorTarget.set(name, index);
    };
    this.tensorTarget.clear();
    acquire(graph.input.name, [channels, height, width]);
    steps.forEach((step, i) => {
      acquire(step.output, step.kind === 'conv' ? step.outShape : step.shape);
      for (const name of new Set(inputsOf(step))) {
        if (lastUse.get(name) !== i) continue;
        const t = this.targets[this.tensorTarget.get(name) as number];
        const key = `${t.width}x${t.height}`;
        free.set(key, [...(free.get(key) ?? []), this.tensorTarget.get(name) as number]);
      }
    });
  }

  private createTarget(width: number, height: number, format?: number): Target {
    if (width > this.maxTextureSize || height > this.maxTextureSize) {
      throw new Error('Activation too large for MAX_TEXTURE_SIZE');
    }
    const gl = this.gl;
    const internal = format ?? (this.precision === 'float32' ? gl.RGBA32F : gl.RGBA16F);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, internal, width, height);
    setNearest(gl);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE && !gl.isContextLost()) {
      gl.deleteFramebuffer(framebuffer);
      gl.deleteTexture(texture);
      throw new Error(`Framebuffer incomplete (0x${status.toString(16)}) for ${width}×${height}`);
    }
    return { texture, framebuffer, width, height };
  }

  private target(name: string): Target {
    return this.targets[this.tensorTarget.get(name) as number];
  }

  private uniform(program: WebGLProgram, name: string): WebGLUniformLocation | null {
    let map = this.uniforms.get(program);
    if (!map) {
      map = new Map();
      this.uniforms.set(program, map);
    }
    if (!map.has(name)) map.set(name, this.gl.getUniformLocation(program, name));
    return map.get(name) as WebGLUniformLocation | null;
  }

  private bindTexture(program: WebGLProgram, name: string, unit: number, texture: WebGLTexture) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(this.uniform(program, name), unit);
  }

  private draw(target: Target): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
    gl.viewport(0, 0, target.width, target.height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Uploads the frame, halves it while possible and resizes it into the input tensor. */
  private preprocess(frame: GlFrame): void {
    const gl = this.gl;
    const { width: mw, height: mh } = this.model;
    let width = frame.width;
    let height = frame.height;
    if (width > this.maxTextureSize || height > this.maxTextureSize) {
      throw new RangeError('Frame larger than MAX_TEXTURE_SIZE');
    }
    if (!(width >= 1 && height >= 1)) throw new RangeError('Empty frame');
    this.frameTexture ??= gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.frameTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    if ('data' in frame) {
      const { data } = frame;
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, frame);
    }
    setNearest(gl);
    let source = this.frameTexture as WebGLTexture;
    let level = 0;
    const { halve, resize } = this.programs;
    while (width >= 2 * mw && height >= 2 * mh && width % 2 === 0 && height % 2 === 0) {
      width /= 2;
      height /= 2;
      let target = this.levels[level];
      if (!target || target.width !== width || target.height !== height) {
        if (target) {
          gl.deleteFramebuffer(target.framebuffer);
          gl.deleteTexture(target.texture);
        }
        target = this.createTarget(width, height, gl.RGBA8);
        this.levels[level] = target;
      }
      gl.useProgram(halve);
      this.bindTexture(halve, 'u_src', 0, source);
      this.draw(target);
      source = target.texture;
      level++;
    }
    // OpenCV uses true area interpolation only when neither axis grows.
    const linear = width < mw || height < mh;
    if (linear) {
      const key = `${width}x${height}`;
      if (this.table?.key !== key) {
        if (this.table) gl.deleteTexture(this.table.texture);
        const n = Math.max(mw, mh);
        const data = new Int32Array(n * 2 * 4);
        data.set(linearAreaTable(width, mw, true), 0);
        data.set(linearAreaTable(height, mh, false), n * 4);
        this.table = { texture: this.uploadTable(data, n), key };
      }
    }
    // Bind after any upload: uploads go through the active texture unit.
    gl.useProgram(resize);
    this.bindTexture(resize, 'u_src', 0, source);
    this.bindTexture(resize, 'u_tab', 1, this.table?.texture as WebGLTexture);
    gl.uniform1i(this.uniform(resize, 'u_linear'), linear ? 1 : 0);
    gl.uniform2i(this.uniform(resize, 'u_srcSize'), width, height);
    gl.uniform2f(this.uniform(resize, 'u_scale'), width / mw, height / mh);
    this.draw(this.target(this.model.graph.input.name));
    // A failed draw would silently leave the previous input in place: check each new frame size.
    const size = `${frame.width}x${frame.height}`;
    if (!this.checkedSizes.has(size)) {
      const error = gl.getError();
      if (error !== gl.NO_ERROR && !gl.isContextLost()) {
        throw new Error(`WebGL2 preprocessing failed: GL error 0x${error.toString(16)} (${size})`);
      }
      this.checkedSizes.add(size);
    }
  }

  /** Runs every step and reads back the output tensor. */
  private execute(): Float32Array {
    const gl = this.gl;
    let w = 0;
    for (const step of this.model.steps) {
      const out = this.target(step.output);
      if (step.kind === 'conv') {
        const program = step.depthwise ? this.programs.depthwise : this.programs.conv;
        gl.useProgram(program);
        const u = (name: string) => this.uniform(program, name);
        this.bindTexture(program, 'u_in', 0, this.target(step.input).texture);
        this.bindTexture(program, 'u_w', 1, this.weights[w++]);
        const residual = step.residual ? this.target(step.residual).texture : null;
        this.bindTexture(program, 'u_res', 2, residual ?? this.weights[0]);
        gl.uniform1i(u('u_residual'), residual ? 1 : 0);
        gl.uniform2i(u('u_inSize'), step.inShape[2], step.inShape[1]);
        gl.uniform2i(u('u_outSize'), step.outShape[2], step.outShape[1]);
        gl.uniform2i(u('u_kernel'), step.kernel[1], step.kernel[0]);
        if (!step.depthwise) gl.uniform1i(u('u_inSlices'), Math.ceil(step.inShape[0] / 4));
        gl.uniform1i(u('u_stride'), step.stride);
        gl.uniform1i(u('u_pad'), step.pad);
        gl.uniform1i(u('u_act'), step.act === 'relu6' ? 1 : step.act === 'hswish' ? 2 : 0);
      } else {
        const program = this.programs.elementwise;
        gl.useProgram(program);
        const a = step.kind === 'add' ? step.a : step.input;
        this.bindTexture(program, 'u_a', 0, this.target(a).texture);
        const b = step.kind === 'add' ? this.target(step.b).texture : this.weights[0];
        this.bindTexture(program, 'u_b', 1, b);
        gl.uniform1i(this.uniform(program, 'u_mode'), step.kind === 'add' ? 0 : 1);
      }
      this.draw(out);
    }
    const [c, h, wd] = this.model.outputShape;
    const texels = this.read(this.target(this.model.output), wd, h * Math.ceil(c / 4));
    // Unpack [C, H, W] from the packed layout.
    const out = new Float32Array(c * h * wd);
    for (let ch = 0; ch < c; ch++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < wd; x++) {
          out[(ch * h + y) * wd + x] = texels[(((ch >> 2) * h + y) * wd + x) * 4 + (ch & 3)];
        }
      }
    }
    if (this.gl.isContextLost()) throw new ContextLostError();
    return out;
  }

  private read(target: Target, width: number, height: number): Float32Array {
    const gl = this.gl;
    const texels = new Float32Array(width * height * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, texels);
    return texels;
  }
}

function inputsOf(step: Step): string[] {
  if (step.kind === 'add') return [step.a, step.b];
  if (step.kind === 'sigmoid') return [step.input];
  return step.residual ? [step.input, step.residual] : [step.input];
}

function setNearest(gl: WebGL2RenderingContext): void {
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

/**
 * Compiles and links a program. The compile status is only queried if linking fails (querying
 * it right away would serialize parallel shader compilation); the error then carries the
 * compile logs of both shaders.
 */
function link(gl: WebGL2RenderingContext, fragment: string): WebGLProgram {
  const program = gl.createProgram();
  const shaders: WebGLShader[] = [];
  for (const [type, source] of [
    [gl.VERTEX_SHADER, VERTEX],
    [gl.FRAGMENT_SHADER, fragment],
  ] as const) {
    const shader = gl.createShader(type);
    if (!shader) throw new Error('Failed to create shader');
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    gl.attachShader(program, shader);
    shaders.push(shader);
  }
  gl.linkProgram(program);
  const linked = gl.getProgramParameter(program, gl.LINK_STATUS);
  const logs = linked
    ? []
    : shaders.map((shader, i) => {
        const compiled = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
        const name = i === 0 ? 'vertex' : 'fragment';
        return `${name} shader ${compiled ? 'compiled' : `failed: ${gl.getShaderInfoLog(shader)}`}`;
      });
  for (const shader of shaders) gl.deleteShader(shader);
  if (!linked) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Shader link error: ${log}; ${logs.join('; ')}`);
  }
  return program;
}

export function createCanvas(): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(1, 1);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 1;
  return canvas;
}
