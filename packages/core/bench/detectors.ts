/**
 * Detector implementations the benchmark runner can run by name (`pnpm bench --detector <name>
 * --options '<json>'`). Each entry turns the JSON options into a detector instance for one
 * processing size; the runner creates one instance per clip and size, so instances may keep
 * per-stream state (buffers, temporal priors).
 */
import { pathToFileURL } from 'node:url';
import type { Detection } from '../src/detect/index.ts';
import type { GrayImage, RgbaImage } from '../src/image.ts';

export interface BenchDetector {
  /**
   * Detects the document in `gray`, which is already downscaled to the processing long side.
   * `rgba` is the same frame in colour (same size), passed if the detector asked for it with
   * `color`. Returns the ranked candidates (best first, including the winner) if it has any.
   */
  detect(gray: GrayImage, rgba?: RgbaImage): Detection | null;
  /** True if the detector needs the colour frame. */
  color?: boolean;
  /** Effective settings to record with the results (e.g. the resolved default thresholds). */
  effective?: Record<string, unknown>;
}

export type BenchDetectorFactory = (
  size: number,
  options: Record<string, unknown>,
) => Promise<BenchDetector>;

type DetectModule = typeof import('../src/detect/index.ts');
type Params = Record<string, number>;
const modules = new Map<string, Promise<DetectModule>>();

/** The tunable constants (`src/detect/params.ts`) of the checkout or `source`, if it has them. */
async function paramsModule(source: unknown): Promise<Params | null> {
  const base = typeof source === 'string' ? pathToFileURL(`${source}/`).href : '../src/';
  try {
    return (await import(`${base}detect/params.ts`)).PARAMS as Params;
  } catch {
    return null;
  }
}

/**
 * The detection module of this checkout, or of the core `src` directory `source` (e.g. an
 * exported older commit, to reproduce its numbers with the current benchmark).
 */
function detectModule(source: unknown): Promise<DetectModule> {
  const key = typeof source === 'string' ? source : '';
  let module = modules.get(key);
  if (!module) {
    module = key
      ? (import(pathToFileURL(`${key}/detect/index.ts`).href) as Promise<DetectModule>)
      : import('../src/detect/index.ts');
    modules.set(key, module);
  }
  return module;
}

export const BENCH_DETECTORS: Record<string, BenchDetectorFactory> = {
  /**
   * The library's `detectDocument`. `options` are `DetectOptions` (except `maxSide`), plus
   * `source` (core `src` directory to load the detector from, default this checkout), `color`
   * (pass the RGBA frame instead of the gray one) and `params` (overrides of the tunable
   * constants in `src/detect/params.ts`, for experiments; each clip runs in its own worker).
   */
  async classic(size, options) {
    const { source, color, params, ...rest } = options;
    const { DetectorContext, detectDocument, DEFAULT_MIN_SCORE } = await detectModule(source);
    const tunable = await paramsModule(source);
    if (params) {
      if (!tunable) throw new Error('params given, but the detector has no params.ts');
      for (const [key, value] of Object.entries(params as Params)) {
        if (!(key in tunable)) throw new Error(`unknown detector param ${key}`);
        tunable[key] = value;
      }
    }
    const effective = {
      minScore: typeof rest.minScore === 'number' ? rest.minScore : (DEFAULT_MIN_SCORE ?? 0.5),
      minArea: typeof rest.minArea === 'number' ? rest.minArea : (tunable?.minArea ?? null),
      ...(tunable ? { params: { ...tunable } } : {}),
    };
    const context = new DetectorContext();
    const detectOptions = { ...rest, maxSide: size, candidates: true };
    const detect = detectDocument as (
      image: GrayImage | RgbaImage,
      options: object,
      context: InstanceType<typeof DetectorContext>,
    ) => Detection | null;
    return color === true
      ? {
          color: true,
          effective,
          detect: (_gray, rgba) => detect(rgba as RgbaImage, detectOptions, context),
        }
      : { effective, detect: (gray) => detect(gray, detectOptions, context) };
  },
};
