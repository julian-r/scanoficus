import type { Detection, DetectOptions } from '@scanoficus/core/detect';
import {
  createDetectorWorker,
  type DetectorFrame,
  type DocumentDetector,
} from '@scanoficus/core/worker';

/**
 * Detector choice of the demo (stored in localStorage) and a `DocumentDetector` that builds the
 * chosen detector, reports what it runs on and falls back to the classic detector when the
 * network cannot be used. `@scanoficus/detector-nn` is loaded lazily, so the classic path does
 * not download it.
 */

export type DetectorChoice = 'classic' | 'nn' | 'nn-refine' | 'hybrid' | 'nn2-refiner';

export const DETECTOR_LABELS: Record<DetectorChoice, string> = {
  classic: 'Classic',
  nn: 'Neural network',
  'nn-refine': 'Neural + refine',
  hybrid: 'Hybrid',
  'nn2-refiner': 'Neural v2 + refiner',
};

const STORAGE_KEY = 'scanoficus-detector';

let webGl2: boolean | null = null;

/**
 * True if this page can render to a float texture with WebGL2 (what the network needs): the
 * extension alone is not enough, the framebuffer must be complete. Probed once per page.
 */
export function hasWebGl2(): boolean {
  if (webGl2 !== null) return webGl2;
  webGl2 = false;
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) return false;
    for (const [extension, format] of [
      ['EXT_color_buffer_float', gl.RGBA32F],
      ['EXT_color_buffer_half_float', gl.RGBA16F],
    ] as const) {
      if (webGl2 || !gl.getExtension(extension)) continue;
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, format, 1, 1);
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      webGl2 = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    }
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  } catch {
    webGl2 = false;
  }
  return webGl2;
}

export function loadChoice(): DetectorChoice {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored && Object.hasOwn(DETECTOR_LABELS, stored)) return stored as DetectorChoice;
  return hasWebGl2() ? 'nn-refine' : 'classic';
}

/**
 * Detection size for a choice: refinement was tuned on frames with a long side of 1024 px, so
 * "Neural + refine" gets 768-px frames (live and stills); the others keep the default 384.
 */
export function detectionMaxSide(choice: DetectorChoice): number | undefined {
  return choice === 'nn-refine' ? 768 : undefined;
}

export function saveChoice(choice: DetectorChoice): void {
  localStorage.setItem(STORAGE_KEY, choice);
}

/** What the active detector runs on, for the debug overlay. */
export interface DetectorStatus {
  choice: DetectorChoice;
  /** Human-readable description, e.g. "Neural + refine · worker · WebGL2 float32". */
  running: string;
  /** Model download + shader compilation + weight upload, in ms (network only). */
  coldStartMs: number | null;
  /** Median detect time over the last 60 frames, in ms. */
  p50Ms: number | null;
  /** Time of the last still detection (full-resolution still with the corner refiner), in ms. */
  stillMs: number | null;
  /** Size of that still. */
  stillSize: string | null;
  /** Why the network is not used (fallback to Classic), if it is not. */
  error: string | null;
  /** Transient trouble that did not end the network: context losses, failed frames. */
  notice: string | null;
}

let current: DetectorStatus | null = null;
const listeners = new Set<(status: DetectorStatus) => void>();

/** Subscribes to status changes of the most recently created demo detector. */
export function onDetectorStatus(listener: (status: DetectorStatus) => void): () => void {
  listeners.add(listener);
  if (current) listener(current);
  return () => listeners.delete(listener);
}

/**
 * Creates the chosen detector. The network runs in a worker when `OffscreenCanvas` WebGL2
 * works there, else on the main thread; if neither works or the model cannot be loaded, the
 * classic worker detector takes over and the reason is reported.
 */
export function createDemoDetector(choice: DetectorChoice): DocumentDetector {
  const status: DetectorStatus = {
    choice,
    running: choice === 'classic' ? 'Classic · worker' : `${DETECTOR_LABELS[choice]} · loading…`,
    coldStartMs: null,
    p50Ms: null,
    stillMs: null,
    stillSize: null,
    error: null,
    notice: null,
  };
  let lostFrames = 0;
  current = status;
  const publish = () => {
    if (current !== status) return;
    for (const listener of listeners) listener({ ...status });
  };
  const times: number[] = [];
  let disposed = false;
  let detector: (DocumentDetector & { detectStill?: DocumentDetector['detectStill'] }) | null =
    null;

  const fallback = (reason: string): DocumentDetector => {
    status.error = reason;
    status.running = 'Classic · worker (fallback)';
    status.coldStartMs = null;
    publish();
    return createDetectorWorker();
  };

  const build = async (): Promise<DocumentDetector> => {
    if (choice === 'classic') return createDetectorWorker();
    const started = performance.now();
    let nn: typeof import('@scanoficus/detector-nn');
    try {
      nn = await import('@scanoficus/detector-nn');
    } catch (error) {
      return fallback(`Could not load the detector module: ${message(error)}`);
    }
    // The model files come with the package (`models/`); Vite emits them as content-hashed
    // assets, so the service worker's cache-first strategy can never serve a stale pair.
    // Round 2 (with the corner refiner on stills; threshold 0.98 from the bundle) is a
    // separate entry, so its files are only fetched when this choice is used.
    const v2 =
      choice === 'nn2-refiner'
        ? (await import('@scanoficus/detector-nn/docquadnet2')).DOCQUADNET2
        : undefined;
    const options = {
      refine: choice === 'nn-refine',
      backend: 'webgl2' as const, // never the slow CPU path in a live loop
      ...(v2 ? { bundle: v2, refiner: { live: false, still: true } } : {}),
    };
    let network: DocumentDetector | null = null;
    let workerError = '';
    if (typeof OffscreenCanvas !== 'undefined') {
      const worker = nn.createNnDetectorWorker({ ...options, timeoutMs: 60_000 });
      try {
        const info = await worker.ready();
        network = worker;
        status.running = `${DETECTOR_LABELS[choice]} · worker · ${describe(info)}`;
      } catch (error) {
        worker.dispose();
        workerError = message(error);
      }
    } else workerError = 'no OffscreenCanvas';
    if (!network) {
      const main = nn.createNnDetector(options);
      try {
        await main.ready();
        network = main;
        status.running = `${DETECTOR_LABELS[choice]} · main thread (worker: ${workerError}) · ${describe(main)}`;
      } catch (error) {
        main.dispose();
        return fallback(`Neural network unavailable: ${message(error)} (worker: ${workerError})`);
      }
    }
    status.coldStartMs = performance.now() - started;
    if (choice === 'hybrid') {
      network = nn.createHybridDetector({
        nn: network,
        classic: createDetectorWorker(),
        // The hybrid answers such frames with the classic result; just report it.
        onError: (error) => {
          status.notice = `network error (classic answered): ${message(error)}`;
          publish();
        },
      });
      status.running = status.running.replace('Hybrid', 'Hybrid (classic first)');
    }
    publish();
    return network;
  };

  const building = build().then((d) => {
    detector = d;
    if (disposed) d.dispose();
    return d;
  });
  publish();

  /** Full-resolution still detection (round 2 with refiner; others use the downscaled path). */
  const detectStill = async (
    still: DetectorFrame,
    options?: DetectOptions,
  ): Promise<Detection | null> => {
    const d = detector ?? (await building);
    if (disposed || !d.detectStill) {
      if ('close' in still) still.close();
      throw new Error('No still detection: the controller uses the downscaled frame');
    }
    const t = performance.now();
    const size = `${still.width}×${still.height}`;
    const result = await d.detectStill(still, options);
    status.stillMs = performance.now() - t;
    status.stillSize = size;
    publish();
    return result;
  };

  return {
    ...(choice === 'nn2-refiner' ? { detectStill } : {}),
    async detect(frame: DetectorFrame, options?: DetectOptions) {
      const d = detector ?? (await building);
      if (disposed) throw new Error('Detector disposed');
      const t = performance.now();
      try {
        // Test hook for the browser check: make the next detection fail with this message.
        const hook = globalThis as { __scanoficusDemoFault?: string };
        const fault = hook.__scanoficusDemoFault;
        if (fault) delete hook.__scanoficusDemoFault;
        const result = await (fault ? Promise.reject(new Error(fault)) : d.detect(frame, options));
        times.push(performance.now() - t);
        if (times.length > 60) times.shift();
        status.p50Ms = [...times].sort((a, b) => a - b)[Math.floor(times.length / 2)];
        publish();
        return result;
      } catch (error) {
        if (choice === 'classic' || status.error || disposed) throw error;
        if (/context lost/i.test(message(error))) {
          // Transient: the library rebuilds its GPU state after the browser restores the
          // context. Skip this frame and keep the network.
          lostFrames++;
          status.notice = `WebGL context lost: ${lostFrames} frame(s) skipped`;
          publish();
          return null;
        }
        // The network failed at runtime: continue with the classic detector.
        d.dispose();
        detector = fallback(`Neural network failed: ${message(error)}`);
        times.length = 0;
        return detector.detect(frame, options);
      }
    },
    dispose() {
      disposed = true;
      detector?.dispose();
    },
  };
}

function describe(info: { backend: string | null; precision: string | null }): string {
  if (info.backend === 'webgl2') return `WebGL2 ${info.precision ?? ''}`.trim();
  return info.backend === 'cpu' ? 'CPU' : 'unknown backend';
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
