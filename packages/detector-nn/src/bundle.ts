import { loadModel, type ModelSource, type NnModel } from './model.ts';

/**
 * Model bundles: a main network plus an optional corner refiner, loaded and checked together.
 *
 * @module
 */

/** Settings of the corner refiner (from `refiner.json` and `pipeline.json` of the export). */
export interface RefinerSettings {
  /** Crop side per pass, as a fraction of √(quad area); one pass per scale. */
  scales: number[];
  /** Crop size in pixels (64). */
  crop: number;
  /** Long side of the work image the crops are taken from (1600). */
  workSide: number;
}

/** Where to fetch a bundle from; every graph names its weights (length, SHA-256). */
export interface BundleSource {
  /** Short name, e.g. `docquadnet2`. */
  name?: string;
  /** Presence threshold chosen for this main network on validation. */
  threshold: number;
  main: ModelSource;
  refiner?: ModelSource & RefinerSettings;
}

/** A loaded corner refiner. */
export interface RefinerModel extends RefinerSettings {
  model: NnModel;
}

/** A loaded bundle. */
export interface NnBundle {
  name: string;
  threshold: number;
  main: NnModel;
  refiner: RefinerModel | null;
}

/** Format of `bundle.json` (written by `scripts/package_model.py`). */
export const BUNDLE_FORMAT = 'scanoficus-nn-bundle/1';

/**
 * Loads a bundle: from a {@link BundleSource} (the package's `DOCQUADNET` / `DOCQUADNET2`, or
 * your own URLs), or from the URL of a `bundle.json` manifest for self-hosted files (fetched
 * with `cache: 'no-cache'`; the graphs it names are resolved relative to it). Each graph's
 * weights are checked as in `loadModel`.
 */
export async function loadBundle(
  source: BundleSource | string | URL,
  init?: RequestInit,
): Promise<NnBundle> {
  const resolved = typeof source === 'object' && 'main' in source ? source : null;
  const spec = resolved ?? (await fetchManifest(source as string | URL, init));
  const [main, refiner] = await Promise.all([
    loadModel(spec.main, init),
    spec.refiner ? loadModel(spec.refiner, init) : null,
  ]);
  if (main.outputShape[0] < 9) throw new Error('The main network must output 8 corners + presence');
  if (refiner && spec.refiner) {
    const { batch, channels, height, width, outputShape } = refiner;
    const crop = spec.refiner.crop;
    if (channels !== 3 || height !== crop || width !== crop || outputShape[0] < 2) {
      throw new Error(`The refiner must map 3×${crop}×${crop} crops to 2 offsets`);
    }
    if (batch !== 4 && batch !== 1) throw new Error('The refiner batch must be 4 (or 1)');
  }
  return {
    name: spec.name ?? 'custom',
    threshold: spec.threshold,
    main,
    refiner:
      refiner && spec.refiner
        ? {
            model: refiner,
            scales: spec.refiner.scales,
            crop: spec.refiner.crop,
            workSide: spec.refiner.workSide,
          }
        : null,
  };
}

async function fetchManifest(url: string | URL, init?: RequestInit): Promise<BundleSource> {
  const base = typeof location !== 'undefined' ? location.href : undefined;
  const href = new URL(url, base);
  const response = await fetch(href, { ...init, cache: 'no-cache' });
  if (!response.ok) throw new Error(`Failed to load ${href.href}: HTTP ${response.status}`);
  const m = (await response.json()) as {
    format: string;
    name?: string;
    threshold: number;
    main: { graph: string };
    refiner?: { graph: string } & RefinerSettings;
  };
  if (m.format !== BUNDLE_FORMAT) throw new Error(`Unsupported bundle format ${m.format}`);
  return {
    name: m.name,
    threshold: m.threshold,
    main: { graphUrl: new URL(m.main.graph, href) },
    refiner: m.refiner && {
      graphUrl: new URL(m.refiner.graph, href),
      scales: m.refiner.scales,
      crop: m.refiner.crop,
      workSide: m.refiner.workSide,
    },
  };
}

/** A bundle source with every URL as an absolute string (for `postMessage` to a worker). */
export function serializableBundle(source: BundleSource): BundleSource {
  const base = typeof location !== 'undefined' ? location.href : undefined;
  const abs = (u: string | URL | undefined) =>
    u === undefined ? undefined : new URL(u, base).href;
  const model = <T extends ModelSource>(m: T): T => ({
    ...m,
    graphUrl: abs(m.graphUrl),
    weightsUrl: abs(m.weightsUrl),
  });
  return {
    ...source,
    main: model(source.main),
    refiner: source.refiner && model(source.refiner),
  };
}
