import type { DetectOptions } from '@scanoficus/core/detect';
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

export type DetectorChoice = 'classic' | 'nn' | 'nn-refine' | 'hybrid';

export const DETECTOR_LABELS: Record<DetectorChoice, string> = {
  classic: 'Classic',
  nn: 'Neural network',
  'nn-refine': 'Neural + refine',
  hybrid: 'Hybrid',
};

const STORAGE_KEY = 'scanoficus-detector';

/** True if this page can render to float textures with WebGL2 (what the network needs). */
export function hasWebGl2(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    const ok =
      !!gl &&
      !!(
        gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float')
      );
    gl?.getExtension('WEBGL_lose_context')?.loseContext();
    return ok;
  } catch {
    return false;
  }
}

export function loadChoice(): DetectorChoice {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored && stored in DETECTOR_LABELS) return stored as DetectorChoice;
  return hasWebGl2() ? 'nn-refine' : 'classic';
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
  /** Why the network is not used (fallback to Classic), if it is not. */
  error: string | null;
}

let current: DetectorStatus | null = null;
const listeners = new Set<(status: DetectorStatus) => void>();

/** Subscribes to status changes of the most recently created demo detector. */
export function onDetectorStatus(listener: (status: DetectorStatus) => void): () => void {
  listeners.add(listener);
  if (current) listener(current);
  return () => listeners.delete(listener);
}

/** URL of a file in the demo's public folder (works under `/scanoficus/` and in dev). */
function publicUrl(path: string): string {
  return new URL(`${import.meta.env.BASE_URL}${path}`, location.href).href;
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
    error: null,
  };
  current = status;
  const publish = () => {
    if (current !== status) return;
    for (const listener of listeners) listener({ ...status });
  };
  const times: number[] = [];
  let disposed = false;
  let detector: DocumentDetector | null = null;

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
    const options = {
      graphUrl: publicUrl('models/docquadnet/docquadnet.json'),
      weightsUrl: publicUrl('models/docquadnet/docquadnet.bin'),
      refine: choice === 'nn-refine',
      backend: 'webgl2' as const, // never the slow CPU path in a live loop
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
      network = nn.createHybridDetector({ nn: network, classic: createDetectorWorker() });
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

  return {
    async detect(frame: DetectorFrame, options?: DetectOptions) {
      const d = detector ?? (await building);
      if (disposed) throw new Error('Detector disposed');
      const t = performance.now();
      try {
        const result = await d.detect(frame, options);
        times.push(performance.now() - t);
        if (times.length > 60) times.shift();
        status.p50Ms = [...times].sort((a, b) => a - b)[Math.floor(times.length / 2)];
        publish();
        return result;
      } catch (error) {
        if (choice === 'classic' || status.error || disposed) throw error;
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
