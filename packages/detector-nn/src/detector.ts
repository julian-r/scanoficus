import type { Detection, DetectOptions } from '@scanoficus/core/detect';
import { orderQuad, type Quad } from '@scanoficus/core/geometry';
import { type GrayImage, type RgbaImage, rgbaToGray } from '@scanoficus/core/image';
import type { DetectorFrame, DocumentDetector } from '@scanoficus/core/worker';
import { type BundleSource, loadBundle, type NnBundle } from './bundle.ts';
import { DOCQUADNET } from './bundle-docquadnet.ts';
import { type RefinerPass, type RgbImage, refineCornersCpu } from './corners.ts';
import { CpuRunner } from './cpu.ts';
import { decodeHeads, fitQuadToFrame, type NnOutput } from './decode.ts';
import {
  ContextLostError,
  createCanvas,
  FrameChangedError,
  type GlPrecision,
  GlRunner,
} from './gl.ts';
import { loadModel, type NnModel } from './model.ts';
import { bytesToInput, preprocessRgb, rgbaToRgb } from './preprocess.ts';
import { type RefineOptions, refineQuad } from './refine.ts';

/** Presence threshold of the round-1 model (scanoficus-ml `export/docquadnet/threshold.json`). */
export const DEFAULT_THRESHOLD = 0.85;

/** What a presence rule gets for one frame. */
export interface PresenceSignals {
  /** Presence logit of the main network and its sigmoid. */
  logit: number;
  presence: number;
  /** The 9 raw outputs of the main network. */
  heads: Float32Array;
  /** The network's quad (frame pixels, may extend beyond the frame). */
  rawQuad: Quad;
  /** The quad after the corner refiner, if it ran for this frame. */
  refinedQuad: Quad | null;
  /** The refiner's offsets per pass (half-crop units, crop frame), if it ran. */
  refiner: RefinerPass[] | null;
  /**
   * The frame (an `ImageBitmap` is still open during the call; do not close or keep it). For
   * stills this is the full-resolution image.
   */
  frame: DetectorFrame;
  /** True for the still re-detection (`detectStill`), false for live frames. */
  still: boolean;
  /** The detector's threshold. */
  threshold: number;
  /**
   * WebGL2 activation precision the frame's networks ran with, or null on the CPU (round-3
   * decisions add `FLOAT16_SCORE_OFFSET` on float16).
   */
  precision: GlPrecision | null;
}

/**
 * Decides document presence: returns a score in [0, 1]; the frame reports a document if the
 * score exceeds the detector's threshold, and the score becomes `Detection.score`. With the
 * default ({@link presenceProbability}) frames below the threshold skip the refiner; with any
 * other rule the refiner runs first (when enabled) so the rule sees its result.
 */
export type PresenceRule = (signals: PresenceSignals) => number;

/** The default presence rule: the main network's presence probability. */
export const presenceProbability: PresenceRule = (s) => s.presence;

/** When the corner refiner of a bundle runs. */
export interface RefinerUse {
  /** On live frames (`detect`). Default false (it costs two refiner passes per frame). */
  live?: boolean;
  /** On stills (`detectStill`: captured photos and imported images). Default true. */
  still?: boolean;
}

/** Inference backend. */
export type NnBackend = 'webgl2' | 'cpu';

/** Options for {@link createNnDetector}. */
export interface NnDetectorOptions {
  /**
   * The model bundle: a {@link BundleSource} (`DOCQUADNET`, the default; `DOCQUADNET2` from
   * `@scanoficus/detector-nn/docquadnet2`, round 2 with the corner refiner; or your own URLs),
   * the URL of a self-hosted `bundle.json`, or a loaded bundle (`loadBundle`). `graphUrl`,
   * `weightsUrl` and `model` describe a main network without a refiner instead.
   */
  bundle?: BundleSource | string | URL | NnBundle | Promise<NnBundle>;
  /**
   * Corner refiner of the bundle: `true` / `{ live, still }` (default: stills only), `false`
   * to never run it.
   */
  refiner?: boolean | RefinerUse;
  /** Presence decision (default {@link presenceProbability} against `threshold`). */
  presence?: PresenceRule;
  /**
   * URL of the graph JSON (relative URLs resolve against the page / worker location). Default:
   * the copy shipped in the package (`models/docquadnet.json`; see `loadModel`).
   */
  graphUrl?: string | URL;
  /**
   * URL of the fp16 weights (≈ 690 kB; not part of the JS bundle). Default: the file named by
   * the graph, relative to `graphUrl`, or the package's copy with the default graph.
   */
  weightsUrl?: string | URL;
  /** An already loaded model instead of the URLs (e.g. shared by several detectors). */
  model?: NnModel | Promise<NnModel>;
  /**
   * A document is reported if its presence score exceeds this. Default: the bundle's threshold
   * (round 1: {@link DEFAULT_THRESHOLD} 0.85, where 0.9–0.95 trades recall for fewer false
   * positives; round 2: 0.98).
   */
  threshold?: number;
  /**
   * `'auto'` (default): WebGL2, falling back to the CPU if WebGL2 or float render targets are
   * missing, initialization or a run fails, and for frames during a context loss.
   * `'webgl2'`: WebGL2 only (errors reject). `'cpu'`: plain TypeScript (≈ 100 ms per frame).
   */
  backend?: 'auto' | NnBackend;
  /** WebGL2 activation precision: `'auto'` (default) = float32 if renderable, else float16. */
  precision?: 'auto' | GlPrecision;
  /**
   * Network first, then classic edge refinement near the predicted quad (see `refineQuad`).
   * `true` uses the tuned defaults. Default false.
   */
  refine?: boolean | RefineOptions;
  /** Canvas for the WebGL2 context. Default: a new `OffscreenCanvas` (or `<canvas>`). */
  canvas?: OffscreenCanvas | HTMLCanvasElement;
  /**
   * `detect()` clamps the quad's corners to the frame and returns `null` for a non-convex or
   * degenerate quad (below 1 % of the frame), so the scanner never gets corners it cannot crop.
   * `infer()` always returns the raw quad. Default true.
   */
  clipToFrame?: boolean;
}

/** A {@link DocumentDetector} backed by DocQuadNet, with access to the raw network output. */
export interface NnDetector extends DocumentDetector {
  /**
   * The backend that answered the last frame (before the first frame: the one initialized, or
   * `null`). `'cpu'` while the CPU answers during a WebGL context loss or after a WebGL2 failure.
   */
  readonly backend: NnBackend | null;
  /** Why WebGL2 is not (or no longer) used, with backend `'auto'`; `null` otherwise. */
  readonly glError: string | null;
  /** WebGL2 activation precision in use, or `null` (CPU, or not initialized yet). */
  readonly precision: GlPrecision | null;
  /**
   * Loads the model and initializes the backend (shader compilation, weight upload): the cold
   * start. Called implicitly by the first detection.
   */
  ready(): Promise<NnBackend>;
  /**
   * Runs the network: the quad (in frame pixels) and presence, regardless of the threshold, and
   * without refinement. Consumes `ImageBitmap` frames like `detect`.
   */
  infer(frame: DetectorFrame): Promise<NnOutput>;
  /**
   * Detection on a full-resolution still (captured photo, imported image): like `detect`, with
   * the corner refiner when the bundle has one (on the still downscaled to the refiner's work
   * size). The `ScannerController` calls it for the re-detection after capture and for
   * `addImage()`. Consumes `ImageBitmap` frames.
   */
  detectStill(frame: DetectorFrame, options?: DetectOptions): Promise<Detection | null>;
  /** The loaded bundle (after `ready()`). */
  readonly bundle: NnBundle | null;
}

type Canvas2D = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

/**
 * Creates the neural document detector. The returned detector implements the core
 * `DocumentDetector` interface: `detect()` resolves with `{ quad, score }` in frame pixels, where
 * `score` is the presence probability, or `null` below the threshold. `DetectOptions` of the
 * classic detector are ignored. Works on the main thread and in workers (`OffscreenCanvas`);
 * see `createNnDetectorWorker` to run it in a worker.
 */
export function createNnDetector(options: NnDetectorOptions = {}): NnDetector {
  const mode = options.backend ?? 'auto';
  const refine = options.refine === true ? {} : options.refine || null;
  /**
   * When the refiner runs: stills by default; with a bundle decision (round 3) also live frames,
   * because the decision needs the refiner's corner scores.
   */
  const refinerUse = (b: NnBundle): Required<RefinerUse> =>
    options.refiner === false
      ? { live: false, still: false }
      : {
          live: options.refiner === true || (options.refiner?.live ?? !!b.decision),
          still: options.refiner === true || (options.refiner?.still ?? true),
        };
  /** The presence rule: the option, else the bundle's decision, else the presence > threshold. */
  const ruleFor = (b: NnBundle): PresenceRule =>
    options.presence ?? b.decision?.rule ?? presenceProbability;
  let bundlePromise: Promise<NnBundle> | null = null;
  let init: Promise<NnBackend> | null = null;
  let bundle: NnBundle | null = null;
  let gl: GlRunner | null = null;
  let cpu: CpuRunner | null = null;
  let cpuRefiner: CpuRunner | null = null;
  let disposed = false;
  let lastBackend: NnBackend | null = null;
  let glError: string | null = null;
  let canvas2d: { canvas: OffscreenCanvas | HTMLCanvasElement; ctx: Canvas2D } | null = null;

  const getBundle = (): Promise<NnBundle> => {
    if (bundlePromise) return bundlePromise;
    const b = options.bundle;
    if (b instanceof Promise) bundlePromise = b;
    else if (b && typeof b === 'object' && 'main' in b && isLoaded(b)) {
      bundlePromise = Promise.resolve(b);
    } else if (b) bundlePromise = loadBundle(b as BundleSource | string | URL);
    else if (options.model || options.graphUrl) {
      const model = options.model
        ? Promise.resolve(options.model)
        : loadModel({ graphUrl: options.graphUrl, weightsUrl: options.weightsUrl });
      bundlePromise = model.then((main) => ({
        name: 'custom',
        threshold: DEFAULT_THRESHOLD,
        main,
        refiner: null,
        decision: null,
      }));
    } else bundlePromise = loadBundle(DOCQUADNET);
    return bundlePromise;
  };

  const fallBack = (error: unknown) => {
    glError = error instanceof Error ? error.message : String(error);
    console.warn(`[detector-nn] WebGL2 not used, running on the CPU: ${glError}`);
  };

  const ready = (): Promise<NnBackend> => {
    init ??= getBundle().then((b) => {
      if (disposed) throw new Error('Detector disposed');
      if (b.decision && !options.presence) {
        // The decision scores the refiner's corner logits: without the refiner it would reject
        // every frame.
        const use = refinerUse(b);
        if (!use.live || !use.still) {
          throw new Error(
            `The ${b.name} bundle's decision needs the refiner on live frames and stills ` +
              '(refiner: { live: true, still: true }, the default); pass a presence rule to ' +
              'run without it',
          );
        }
      }
      if (b.decision && options.threshold !== undefined) {
        console.warn(
          `[detector-nn] threshold ${options.threshold} is compared with sigmoid(score) of the ` +
            `${b.name} decision (its own threshold: ${b.threshold.toFixed(4)} = ` +
            `sigmoid(${b.decision.spec.threshold}))`,
        );
      }
      bundle = b;
      if (mode !== 'cpu') {
        const use = refinerUse(b);
        const useRefiner = b.refiner && (use.live || use.still);
        try {
          gl = new GlRunner(b.main, {
            precision: options.precision,
            canvas: options.canvas,
            refiner: useRefiner ? b.refiner?.model : undefined,
          });
        } catch (error) {
          if (mode === 'webgl2') throw error;
          fallBack(error);
        }
      }
      return gl ? 'webgl2' : 'cpu';
    });
    // A failure (network, WebGL2 with backend 'webgl2') is not cached: the next call retries.
    init.catch(() => {
      if (!bundle) bundlePromise = null;
      init = null;
    });
    return init;
  };

  const context2d = (width: number, height: number): Canvas2D => {
    if (!canvas2d) {
      const canvas = createCanvas();
      const ctx = canvas.getContext('2d', { willReadFrequently: true }) as Canvas2D | null;
      if (!ctx) throw new Error('2D canvas context unavailable');
      canvas2d = { canvas, ctx };
    }
    if (canvas2d.canvas.width !== width || canvas2d.canvas.height !== height) {
      canvas2d.canvas.width = width;
      canvas2d.canvas.height = height;
    }
    return canvas2d.ctx;
  };

  /** RGBA pixels of a bitmap drawn (and optionally resized) by the browser. */
  const drawBitmap = (bitmap: ImageBitmap, width: number, height: number): RgbaImage => {
    const ctx = context2d(width, height);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, 0, 0, width, height);
    return ctx.getImageData(0, 0, width, height);
  };

  /**
   * One frame in flight: what the networks run on (`source`: the frame, or for a bitmap that is
   * too large, a browser-drawn copy at `sx × sy` of its size), the pixels read from it (once),
   * and the token of its upload to the GPU.
   */
  interface FrameJob {
    frame: DetectorFrame;
    source: DetectorFrame;
    sx: number;
    sy: number;
    rgba: RgbaImage | null;
    rgb: RgbImage | null;
    token: number | null;
  }

  /**
   * Bitmaps above the GPU's texture limit (or, on the CPU, above 2 × the refiner's work side)
   * are drawn by the browser at a long side of 2 × workSide (3200) first, so a 50 MP still is
   * never read at full size. This path is not bit-exact with the Python pipeline (the browser's
   * filter instead of one area resize). RGBA frames are used as they are.
   */
  const prepare = (frame: DetectorFrame): FrameJob => {
    const job: FrameJob = {
      frame,
      source: frame,
      sx: 1,
      sy: 1,
      rgba: null,
      rgb: null,
      token: null,
    };
    if (!isBitmap(frame)) return job;
    const reduced = 2 * (bundle?.refiner?.workSide ?? 1600);
    const long = Math.max(frame.width, frame.height);
    const limit = gl ? gl.maxFrameSize : reduced;
    if (long <= limit) return job;
    const k = Math.min(limit, reduced) / long;
    const width = Math.max(1, Math.round(frame.width * k));
    const height = Math.max(1, Math.round(frame.height * k));
    job.rgba = drawBitmap(frame, width, height);
    job.source = job.rgba;
    job.sx = width / frame.width;
    job.sy = height / frame.height;
    return job;
  };

  const rgbaOf = (job: FrameJob): RgbaImage => {
    job.rgba ??= isBitmap(job.source)
      ? drawBitmap(job.source, job.source.width, job.source.height)
      : job.source;
    return job.rgba;
  };

  const rgbOf = (job: FrameJob): RgbImage => {
    if (!job.rgb) {
      const rgba = rgbaOf(job);
      job.rgb = { width: rgba.width, height: rgba.height, data: rgbaToRgb(rgba) };
    }
    return job.rgb;
  };

  const runCpu = (m: NnModel, job: FrameJob): Float32Array => {
    cpu ??= new CpuRunner(m);
    lastBackend = 'cpu';
    return cpu.run(bytesToInput(preprocessRgb(rgbOf(job), m.width, m.height)));
  };

  /** A GL failure: context loss → this frame on the CPU; anything else → CPU from now on. */
  const glFailed = (error: unknown) => {
    if (mode === 'webgl2') throw error;
    if (!(error instanceof ContextLostError)) {
      gl?.dispose();
      gl = null;
      fallBack(error);
    }
  };

  const run = (m: NnModel, job: FrameJob): Float32Array => {
    if (gl) {
      try {
        const { source } = job;
        let heads: Float32Array;
        if (Math.max(source.width, source.height) <= gl.maxFrameSize) {
          const result = gl.runFrame(source);
          heads = result.heads;
          job.token = result.token;
        } else {
          heads = gl.runTensor(bytesToInput(preprocessRgb(rgbOf(job), m.width, m.height)));
        }
        lastBackend = 'webgl2';
        return heads;
      } catch (error) {
        glFailed(error);
      }
    }
    return runCpu(m, job);
  };

  /** The corner refiner on the job's frame (GPU if that frame is uploaded, else the CPU). */
  const runRefiner = (b: NnBundle, job: FrameJob, out: NnOutput) => {
    const refiner = b.refiner as NonNullable<NnBundle['refiner']>;
    const { sx, sy } = job;
    // Round 3 starts from the network's normalized corners (in network order), round 2 from the
    // quad in frame pixels (scaled to the source if the browser reduced a large bitmap).
    const scaled = (refiner.coordinates === 'normalized'
      ? [0, 1, 2, 3].map((i) => ({ x: out.heads[2 * i], y: out.heads[2 * i + 1] }))
      : out.quad.map((p) => ({ x: p.x * sx, y: p.y * sy }))) as unknown as Quad;
    let result: ReturnType<typeof refineCornersCpu> | null = null;
    if (gl?.hasRefiner && job.token !== null) {
      try {
        result = gl.refine(scaled, refiner, job.token);
      } catch (error) {
        // Another frame on the GPU (cannot happen while process() is serialized): CPU.
        if (!(error instanceof FrameChangedError)) glFailed(error);
      }
    }
    if (!result) {
      cpuRefiner ??= new CpuRunner(refiner.model);
      const runner = cpuRefiner;
      lastBackend = 'cpu';
      result = refineCornersCpu(rgbOf(job), scaled, refiner, (input, n) => runner.run(input, n));
    }
    if (sx === 1 && sy === 1) return result;
    return {
      ...result,
      quad: result.quad.map((p) => ({ x: p.x / sx, y: p.y / sy })) as unknown as Quad,
    };
  };

  /** Runs the network without consuming the frame. */
  const evaluate = async (job: FrameJob): Promise<NnOutput> => {
    await ready();
    if (disposed) throw new Error('Detector disposed');
    const m = (bundle as NnBundle).main;
    const { frame } = job;
    if (!(frame.width >= 1 && frame.height >= 1)) throw new RangeError('Empty frame');
    if (!isBitmap(frame) && frame.data?.length !== frame.width * frame.height * 4) {
      throw new TypeError('Expected an ImageBitmap or RGBA pixels with ImageData layout');
    }
    return decodeHeads(run(m, job), frame.width, frame.height);
  };

  /** Gray frame at ≤ `longSide` for refinement (bitmaps are downscaled by the browser). */
  const grayFor = (job: FrameJob, longSide: number): GrayImage => {
    const { frame } = job;
    const scale = Math.min(1, longSide / Math.max(frame.width, frame.height));
    if (isBitmap(frame)) {
      const w = Math.max(1, Math.round(frame.width * scale));
      const h = Math.max(1, Math.round(frame.height * scale));
      return rgbaToGray(drawBitmap(frame, w, h));
    }
    return rgbaToGray(frame); // refineQuad area-downscales it
  };

  /**
   * Frames are processed one at a time, in call order: the GPU holds one uploaded frame and the
   * refiner reads it, so two concurrent calls must not interleave.
   */
  let tail: Promise<unknown> = Promise.resolve();
  const serialized = <T>(task: () => Promise<T>): Promise<T> => {
    const next = tail.then(task, task);
    tail = next.catch(() => {});
    return next;
  };

  /** The whole pipeline for one frame: network, refiner, presence rule, edge refinement. */
  const process = async (frame: DetectorFrame, still: boolean): Promise<Detection | null> => {
    try {
      await ready();
      if (disposed) throw new Error('Detector disposed');
      const job = prepare(frame);
      const out = await evaluate(job);
      const b = bundle as NnBundle;
      const threshold = options.threshold ?? b.threshold;
      const signals: PresenceSignals = {
        logit: out.heads[8],
        presence: out.presence,
        heads: out.heads,
        rawQuad: out.quad,
        refinedQuad: null,
        refiner: null,
        frame,
        still,
        threshold,
        precision: null,
      };
      const rule = ruleFor(b);
      const use = refinerUse(b);
      // The default rule decides before the (costly) refiner.
      if (rule === presenceProbability && !(out.presence > threshold)) return null;
      if (b.refiner && (still ? use.still : use.live)) {
        const refined = runRefiner(b, job, out);
        signals.refinedQuad = orderQuad(refined.quad);
        signals.refiner = refined.passes;
      }
      signals.precision =
        lastBackend === 'webgl2' ? ((gl as GlRunner | null)?.precision ?? null) : null;
      const score = rule(signals);
      if (!(score > threshold)) return null;
      let quad: Quad = signals.refinedQuad ?? out.quad;
      if (refine) {
        const longSide = refine.longSide ?? 1024;
        const gray = grayFor(job, longSide);
        const sx = gray.width / frame.width;
        const sy = gray.height / frame.height;
        const scaled = quad.map((p) => ({ x: p.x * sx, y: p.y * sy })) as unknown as Quad;
        quad = refineQuad(gray, scaled, refine).map((p) => ({
          x: p.x / sx,
          y: p.y / sy,
        })) as unknown as Quad;
      }
      if (options.clipToFrame !== false) {
        const fitted = fitQuadToFrame(quad, frame.width, frame.height);
        if (!fitted) return null;
        quad = fitted;
      }
      return { quad, score };
    } finally {
      if (isBitmap(frame)) frame.close();
    }
  };

  return {
    get backend() {
      return lastBackend ?? (gl ? 'webgl2' : bundle ? 'cpu' : null);
    },
    get glError() {
      return glError;
    },
    get precision() {
      return (gl as GlRunner | null)?.precision ?? null;
    },
    get bundle() {
      return bundle;
    },
    ready,
    infer: (frame) =>
      serialized(async () => {
        try {
          await ready();
          if (disposed) throw new Error('Detector disposed');
          return await evaluate(prepare(frame));
        } finally {
          if (isBitmap(frame)) frame.close();
        }
      }),
    detect: (frame) => serialized(() => process(frame, false)),
    detectStill: (frame) => serialized(() => process(frame, true)),
    dispose() {
      disposed = true;
      (gl as GlRunner | null)?.dispose();
      gl = null;
      cpu = null;
      cpuRefiner = null;
      if (canvas2d) canvas2d.canvas.width = canvas2d.canvas.height = 0;
      canvas2d = null;
    },
  };
}

function isLoaded(b: object): b is NnBundle {
  const main = (b as { main?: unknown }).main as { steps?: unknown } | undefined;
  return Array.isArray(main?.steps);
}

export function isBitmap(frame: unknown): frame is ImageBitmap {
  return typeof ImageBitmap !== 'undefined' && frame instanceof ImageBitmap;
}
