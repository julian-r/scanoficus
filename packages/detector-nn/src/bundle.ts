import { createDecision, type Decision, type DecisionSpec } from './decision.ts';
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
  /**
   * How the network's quad enters the work image: `'frame'` (round 2, `pipeline.py`: frame
   * pixels times work width / frame width) or `'normalized'` (round 3, `verify.py`: normalized
   * corners times the work-image size). Default `'frame'`.
   */
  coordinates?: 'frame' | 'normalized';
}

/** Where to fetch a bundle from; every graph names its weights (length, SHA-256). */
export interface BundleSource {
  /** Short name, e.g. `docquadnet2`. */
  name?: string;
  /**
   * Presence threshold chosen for this main network on validation. With a `decisionUrl`, the
   * decision's threshold is used instead (as a probability: the sigmoid of its score threshold).
   */
  threshold?: number;
  main: ModelSource;
  refiner?: ModelSource & RefinerSettings;
  /**
   * The bundle's built-in presence rule (round 3): the runtime fields of its `decision.json`,
   * inline (the packaged bundles), or `decisionUrl` to fetch the file (self-hosted `bundle.json`).
   */
  decision?: DecisionSpec;
  decisionUrl?: string | URL;
}

/** A loaded corner refiner. */
export interface RefinerModel extends RefinerSettings {
  model: NnModel;
}

/** A loaded bundle. */
export interface NnBundle {
  name: string;
  /** Presence threshold (a probability; for a decision bundle the sigmoid of its threshold). */
  threshold: number;
  main: NnModel;
  refiner: RefinerModel | null;
  /** The built-in presence decision, if the bundle has one. */
  decision: Decision | null;
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
  const [main, refiner, decisionSpec] = await Promise.all([
    loadModel(spec.main, init),
    spec.refiner ? loadModel(spec.refiner, init) : null,
    spec.decision ?? (spec.decisionUrl ? fetchJson(spec.decisionUrl, init) : null),
  ]);
  const decision = decisionSpec ? createDecision(decisionSpec as DecisionSpec) : null;
  if (decision) {
    // The decision was fitted on these exact weights: other weights would give other scores.
    const pinned = [
      ['main', decision.spec.mainSha256, main.graph.weights?.sha256],
      ['refiner', decision.spec.refinerSha256, refiner?.graph.weights?.sha256],
    ] as const;
    for (const [what, expected, actual] of pinned) {
      if (!expected || expected !== actual) {
        throw new Error(
          `The decision does not belong to these weights: ${what} weights SHA-256 ${actual}, ` +
            `decision fitted on ${expected ?? '(not pinned: mainSha256 / refinerSha256 missing)'}`,
        );
      }
    }
    const refinerOutputs = refiner?.outputShape[0] ?? 0;
    if (!spec.refiner || refinerOutputs < 3) {
      throw new Error('The decision needs a refiner with corner-confidence outputs');
    }
    if (decision.spec.scales.join() !== spec.refiner.scales.join()) {
      throw new Error('The decision was fitted with other refiner scales');
    }
  }
  const threshold = decision?.threshold ?? spec.threshold;
  if (threshold === undefined) throw new Error('The bundle has no threshold');
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
    threshold,
    decision,
    main,
    refiner:
      refiner && spec.refiner
        ? {
            model: refiner,
            scales: spec.refiner.scales,
            crop: spec.refiner.crop,
            workSide: spec.refiner.workSide,
            coordinates: spec.refiner.coordinates ?? 'frame',
          }
        : null,
  };
}

async function fetchJson(url: string | URL, init?: RequestInit): Promise<unknown> {
  const base = typeof location !== 'undefined' ? location.href : undefined;
  const href = new URL(url, base);
  const response = await fetch(href, { ...init, cache: 'no-cache' });
  if (!response.ok) throw new Error(`Failed to load ${href.href}: HTTP ${response.status}`);
  return response.json();
}

async function fetchManifest(url: string | URL, init?: RequestInit): Promise<BundleSource> {
  const base = typeof location !== 'undefined' ? location.href : undefined;
  const href = new URL(url, base);
  const response = await fetch(href, { ...init, cache: 'no-cache' });
  if (!response.ok) throw new Error(`Failed to load ${href.href}: HTTP ${response.status}`);
  const m = (await response.json()) as {
    format: string;
    name?: string;
    threshold: number | null;
    main: { graph: string };
    refiner?: { graph: string } & RefinerSettings;
    decision?: string;
  };
  if (m.format !== BUNDLE_FORMAT) throw new Error(`Unsupported bundle format ${m.format}`);
  return {
    name: m.name,
    threshold: m.threshold ?? undefined,
    decisionUrl: m.decision ? new URL(m.decision, href) : undefined,
    main: { graphUrl: new URL(m.main.graph, href) },
    refiner: m.refiner && {
      graphUrl: new URL(m.refiner.graph, href),
      scales: m.refiner.scales,
      crop: m.refiner.crop,
      workSide: m.refiner.workSide,
      coordinates: m.refiner.coordinates,
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
    decision: source.decision,
    decisionUrl: abs(source.decisionUrl),
  };
}
