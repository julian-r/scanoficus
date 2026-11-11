import type { Detection, DetectOptions } from '@scanoficus/core/detect';
import { orderQuad, type Quad } from '@scanoficus/core/geometry';
import { type GrayImage, type RgbaImage, rgbaToGray } from '@scanoficus/core/image';
import type { DetectorFrame, DocumentDetector } from '@scanoficus/core/worker';
import { type BundleSource, loadBundle, type NnBundle } from './bundle.ts';
import { DOCQUADNET } from './bundle-docquadnet.ts';
import { type RefinerPass, refineCornersCpu } from './corners.ts';
import { CpuRunner } from './cpu.ts';
import { decodeHeads, fitQuadToFrame, type NnOutput } from './decode.ts';
import { ContextLostError, createCanvas, type GlPrecision, GlRunner } from './gl.ts';
import { loadModel, type NnModel } from './model.ts';
import { bytesToInput, preprocessRgba, rgbaToRgb } from './preprocess.ts';
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
  const refinerUse: Required<RefinerUse> =
    options.refiner === false
      ? { live: false, still: false }
      : {
          live: options.refiner === true || (options.refiner?.live ?? false),
          still: options.refiner === true || (options.refiner?.still ?? true),
        };
  const rule = options.presence ?? presenceProbability;
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
      bundle = b;
      if (mode !== 'cpu') {
        const useRefiner = b.refiner && (refinerUse.live || refinerUse.still);
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

  /** RGBA pixels of a frame; bitmaps are drawn (and optionally downscaled) by the browser. */
  const pixels = (frame: DetectorFrame, width = frame.width, height = frame.height): RgbaImage => {
    if (!isBitmap(frame)) return frame;
    const ctx = context2d(width, height);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(frame, 0, 0, width, height);
    return ctx.getImageData(0, 0, width, height);
  };

  /** Whether the frame of the last main-network run is on the GPU (for the refiner). */
  let frameOnGpu = false;

  const runCpu = (m: NnModel, frame: DetectorFrame): Float32Array => {
    cpu ??= new CpuRunner(m);
    lastBackend = 'cpu';
    frameOnGpu = false;
    return cpu.run(bytesToInput(preprocessRgba(pixels(frame), m.width, m.height)));
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

  const run = (m: NnModel, frame: DetectorFrame): Float32Array => {
    if (gl) {
      try {
        const fits = Math.max(frame.width, frame.height) <= gl.maxFrameSize;
        const heads = fits
          ? gl.run(frame)
          : gl.runTensor(bytesToInput(preprocessRgba(pixels(frame), m.width, m.height)));
        lastBackend = 'webgl2';
        frameOnGpu = fits;
        return heads;
      } catch (error) {
        glFailed(error);
      }
    }
    return runCpu(m, frame);
  };

  /** The corner refiner on the frame of the last run (GPU if it is there, else the CPU). */
  const runRefiner = (b: NnBundle, frame: DetectorFrame, quad: Quad) => {
    const refiner = b.refiner as NonNullable<NnBundle['refiner']>;
    if (gl?.hasRefiner && frameOnGpu) {
      try {
        return gl.refine(quad, refiner);
      } catch (error) {
        glFailed(error);
      }
    }
    cpuRefiner ??= new CpuRunner(refiner.model);
    const runner = cpuRefiner;
    const rgba = pixels(frame);
    lastBackend = 'cpu';
    return refineCornersCpu(
      { width: rgba.width, height: rgba.height, data: rgbaToRgb(rgba) },
      quad,
      refiner,
      (input, n) => runner.run(input, n),
    );
  };

  /** Runs the network without consuming the frame. */
  const evaluate = async (frame: DetectorFrame): Promise<NnOutput> => {
    await ready();
    if (disposed) throw new Error('Detector disposed');
    const m = (bundle as NnBundle).main;
    if (!(frame.width >= 1 && frame.height >= 1)) throw new RangeError('Empty frame');
    if (!isBitmap(frame) && frame.data?.length !== frame.width * frame.height * 4) {
      throw new TypeError('Expected an ImageBitmap or RGBA pixels with ImageData layout');
    }
    return decodeHeads(run(m, frame), frame.width, frame.height);
  };

  /** Gray frame at ≤ `longSide` for refinement (bitmaps are downscaled by the browser). */
  const grayFor = (frame: DetectorFrame, longSide: number): GrayImage => {
    const scale = Math.min(1, longSide / Math.max(frame.width, frame.height));
    if (isBitmap(frame)) {
      const w = Math.max(1, Math.round(frame.width * scale));
      const h = Math.max(1, Math.round(frame.height * scale));
      return rgbaToGray(pixels(frame, w, h));
    }
    return rgbaToGray(frame); // refineQuad area-downscales it
  };

  /** The whole pipeline for one frame: network, refiner, presence rule, edge refinement. */
  const process = async (frame: DetectorFrame, still: boolean): Promise<Detection | null> => {
    try {
      const out = await evaluate(frame);
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
      };
      // The default rule decides before the (costly) refiner.
      if (rule === presenceProbability && !(out.presence > threshold)) return null;
      if (b.refiner && (still ? refinerUse.still : refinerUse.live)) {
        const refined = runRefiner(b, frame, out.quad);
        signals.refinedQuad = orderQuad(refined.quad);
        signals.refiner = refined.passes;
      }
      const score = rule(signals);
      if (!(score > threshold)) return null;
      let quad: Quad = signals.refinedQuad ?? out.quad;
      if (refine) {
        const longSide = refine.longSide ?? 1024;
        const gray = grayFor(frame, longSide);
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
    async infer(frame) {
      try {
        return await evaluate(frame);
      } finally {
        if (isBitmap(frame)) frame.close();
      }
    },
    detect: (frame) => process(frame, false),
    detectStill: (frame) => process(frame, true),
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
