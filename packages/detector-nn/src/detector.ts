import type { Quad } from '@scanoficus/core/geometry';
import { type GrayImage, type RgbaImage, rgbaToGray } from '@scanoficus/core/image';
import type { DetectorFrame, DocumentDetector } from '@scanoficus/core/worker';
import { CpuRunner } from './cpu.ts';
import { decodeHeads, fitQuadToFrame, type NnOutput } from './decode.ts';
import { ContextLostError, createCanvas, type GlPrecision, GlRunner } from './gl.ts';
import { loadModel, type NnModel } from './model.ts';
import { bytesToInput, preprocessRgba } from './preprocess.ts';
import { type RefineOptions, refineQuad } from './refine.ts';

/** Presence threshold chosen on validation (scanoficus-ml `export/docquadnet/threshold.json`). */
export const DEFAULT_THRESHOLD = 0.85;

/** Inference backend. */
export type NnBackend = 'webgl2' | 'cpu';

/** Options for {@link createNnDetector}. */
export interface NnDetectorOptions {
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
   * A document is reported if its presence probability exceeds this. Default
   * {@link DEFAULT_THRESHOLD} (0.85); 0.9–0.95 trades recall for fewer false positives.
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
}

type Canvas2D = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

/**
 * Creates the neural document detector. The returned detector implements the core
 * `DocumentDetector` interface: `detect()` resolves with `{ quad, score }` in frame pixels, where
 * `score` is the presence probability, or `null` below the threshold. `DetectOptions` of the
 * classic detector are ignored. Works on the main thread and in workers (`OffscreenCanvas`);
 * see `createNnDetectorWorker` to run it in a worker.
 */
export function createNnDetector(options: NnDetectorOptions): NnDetector {
  const mode = options.backend ?? 'auto';
  const threshold = options.threshold ?? DEFAULT_THRESHOLD;
  const refine = options.refine === true ? {} : options.refine || null;
  let modelPromise: Promise<NnModel> | null = null;
  let init: Promise<NnBackend> | null = null;
  let model: NnModel | null = null;
  let gl: GlRunner | null = null;
  let cpu: CpuRunner | null = null;
  let disposed = false;
  let lastBackend: NnBackend | null = null;
  let glError: string | null = null;
  let canvas2d: { canvas: OffscreenCanvas | HTMLCanvasElement; ctx: Canvas2D } | null = null;

  const getModel = () => {
    modelPromise ??= options.model
      ? Promise.resolve(options.model)
      : loadModel({ graphUrl: options.graphUrl, weightsUrl: options.weightsUrl });
    return modelPromise;
  };

  const fallBack = (error: unknown) => {
    glError = error instanceof Error ? error.message : String(error);
    console.warn(`[detector-nn] WebGL2 not used, running on the CPU: ${glError}`);
  };

  const ready = (): Promise<NnBackend> => {
    init ??= getModel().then((m) => {
      if (disposed) throw new Error('Detector disposed');
      model = m;
      if (mode !== 'cpu') {
        try {
          gl = new GlRunner(m, { precision: options.precision, canvas: options.canvas });
        } catch (error) {
          if (mode === 'webgl2') throw error;
          fallBack(error);
        }
      }
      return gl ? 'webgl2' : 'cpu';
    });
    // A failure (network, WebGL2 with backend 'webgl2') is not cached: the next call retries.
    init.catch(() => {
      if (!model) modelPromise = null;
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

  const runCpu = (m: NnModel, frame: DetectorFrame): Float32Array => {
    cpu ??= new CpuRunner(m);
    lastBackend = 'cpu';
    return cpu.run(bytesToInput(preprocessRgba(pixels(frame), m.width, m.height)));
  };

  const run = (m: NnModel, frame: DetectorFrame): Float32Array => {
    if (gl) {
      try {
        const heads =
          Math.max(frame.width, frame.height) > gl.maxFrameSize
            ? gl.runTensor(bytesToInput(preprocessRgba(pixels(frame), m.width, m.height)))
            : gl.run(frame);
        lastBackend = 'webgl2';
        return heads;
      } catch (error) {
        if (mode === 'webgl2') throw error;
        // During a context loss, use the CPU for this frame and retry WebGL2 later; any other
        // failure switches to the CPU permanently.
        if (!(error instanceof ContextLostError)) {
          gl.dispose();
          gl = null;
          fallBack(error);
        }
      }
    }
    return runCpu(m, frame);
  };

  /** Runs the network without consuming the frame. */
  const evaluate = async (frame: DetectorFrame): Promise<NnOutput> => {
    await ready();
    if (disposed) throw new Error('Detector disposed');
    const m = model as NnModel;
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

  return {
    get backend() {
      return lastBackend ?? (gl ? 'webgl2' : model ? 'cpu' : null);
    },
    get glError() {
      return glError;
    },
    get precision() {
      return (gl as GlRunner | null)?.precision ?? null;
    },
    ready,
    async infer(frame) {
      try {
        return await evaluate(frame);
      } finally {
        if (isBitmap(frame)) frame.close();
      }
    },
    async detect(frame) {
      try {
        const out = await evaluate(frame);
        if (!(out.presence > threshold)) return null;
        let quad: Quad = out.quad;
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
        return { quad, score: out.presence };
      } finally {
        if (isBitmap(frame)) frame.close();
      }
    },
    dispose() {
      disposed = true;
      (gl as GlRunner | null)?.dispose();
      gl = null;
      cpu = null;
      if (canvas2d) canvas2d.canvas.width = canvas2d.canvas.height = 0;
      canvas2d = null;
    },
  };
}

export function isBitmap(frame: unknown): frame is ImageBitmap {
  return typeof ImageBitmap !== 'undefined' && frame instanceof ImageBitmap;
}
