/**
 * Side-by-side summary of one or more benchmark runs (one run directory per detector or
 * variant), split into the tuning and held-out parts of the protocol in `splits.ts`, plus the
 * false-positive rate on the no-document negatives.
 */
import type { DatasetName } from './datasets.ts';
import {
  aggregate,
  clipMetrics,
  type FrameMetrics,
  falsePositiveRate,
  METRIC_HEADER,
  metricCells,
  type NegativeMetrics,
  negativeMetrics,
  num,
  pct,
  table,
} from './report-data.ts';
import { type ClipResult, loadRun } from './results.ts';
import { splitOf } from './splits.ts';

/** A run and the processing long side to evaluate (`name@size` on the command line). */
export interface RunSpec {
  name: string;
  /** Explicit size, or `undefined` for the preferred size if the run has it, else its first. */
  size?: number;
}

export function parseRunSpec(text: string): RunSpec {
  const [name, size] = text.split('@');
  return size ? { name, size: Number(size) } : { name };
}

interface LoadedRun {
  label: string;
  /** Size used per dataset (runs may use different sizes per dataset). */
  sizes: Partial<Record<DatasetName, number>>;
  positives: Partial<Record<DatasetName, { clip: ClipResult; metrics: FrameMetrics[] }[]>>;
  negatives: NegativeMetrics[] | null;
}

function loadSpec(spec: RunSpec, preferred: number): LoadedRun {
  const loaded: LoadedRun = { label: spec.name, sizes: {}, positives: {}, negatives: null };
  for (const dataset of ['smartdoc', 'midv500', 'midv2019', 'negatives'] as const) {
    const clips = loadRun(spec.name, dataset);
    if (!clips.length) continue;
    const available = clips[0].sizes;
    const size = spec.size ?? (available.includes(preferred) ? preferred : available[0]);
    if (!clips.every((c) => c.sizes.includes(size))) {
      throw new Error(`run ${spec.name}: not every ${dataset} clip has size ${size}`);
    }
    loaded.sizes[dataset] = size;
    if (dataset === 'negatives') {
      loaded.negatives = clips.flatMap((clip) => negativeMetrics(clip, size));
    } else {
      loaded.positives[dataset] = clips.map((clip) => ({
        clip,
        metrics: clipMetrics(clip, size, 'frame'),
      }));
    }
  }
  const sizes = [...new Set(Object.values(loaded.sizes))];
  if (sizes.length) loaded.label = `${spec.name} @${sizes.join('/')}`;
  return loaded;
}

type Subset = {
  title: string;
  dataset: DatasetName;
  keep: (clip: ClipResult) => boolean;
};

const BG5 = 'background05';
const SUBSETS: Subset[] = [
  {
    title: 'SmartDoc tuning split',
    dataset: 'smartdoc',
    keep: (c) => splitOf(c) === 'tune',
  },
  {
    title: 'SmartDoc held-out split',
    dataset: 'smartdoc',
    keep: (c) => splitOf(c) === 'heldout',
  },
  {
    title: 'SmartDoc held-out, backgrounds 1–4',
    dataset: 'smartdoc',
    keep: (c) => splitOf(c) === 'heldout' && c.background !== BG5,
  },
  { title: 'SmartDoc all', dataset: 'smartdoc', keep: () => true },
  { title: 'MIDV-500 (held out)', dataset: 'midv500', keep: () => true },
  { title: 'MIDV-2019 (held out)', dataset: 'midv2019', keep: () => true },
];

function subsetMetrics(run: LoadedRun, subset: Subset): FrameMetrics[] | null {
  const clips = run.positives[subset.dataset];
  if (!clips) return null;
  return clips.filter((c) => subset.keep(c.clip)).flatMap((c) => c.metrics);
}

/**
 * Markdown sections comparing `specs`: a summary (JI per split, FP rate), one metric table per
 * split, SmartDoc JI per background and split, and the negatives at several score thresholds.
 */
export function summarySections(
  specs: readonly RunSpec[],
  preferredSize: number,
  thresholds: readonly number[],
): string[] {
  const runs = specs.map((spec) => loadSpec(spec, preferredSize));
  const out: string[] = [];
  const ji = (metrics: FrameMetrics[] | null) => (metrics ? num(aggregate(metrics).ji) : '–');

  out.push('## Summary by split (SmartDoc JI; held-out numbers are the ones to compare)');
  out.push(
    table(
      ['run', ...SUBSETS.map((s) => s.title), 'negatives: FP rate', 'ms (SmartDoc)'],
      runs.map((run) => [
        run.label,
        ...SUBSETS.map((s) => ji(subsetMetrics(run, s))),
        run.negatives ? pct(falsePositiveRate(run.negatives)) : '–',
        msCell(subsetMetrics(run, SUBSETS[3])),
      ]),
    ),
  );

  for (const subset of SUBSETS) {
    const rows = runs
      .map((run) => [run, subsetMetrics(run, subset)] as const)
      .filter((entry): entry is readonly [LoadedRun, FrameMetrics[]] => entry[1] !== null)
      .map(([run, metrics]) => [run.label, ...metricCells(aggregate(metrics))]);
    if (!rows.length) continue;
    out.push(`### ${subset.title}`);
    out.push(table(['run', ...METRIC_HEADER], rows));
  }

  const backgrounds = [
    ...new Set(
      runs.flatMap((run) => (run.positives.smartdoc ?? []).map(({ clip }) => clip.background)),
    ),
  ].sort();
  if (backgrounds.length) {
    out.push('### SmartDoc JI per background and split');
    const rows: string[][] = [];
    for (const split of ['tune', 'heldout'] as const) {
      for (const run of runs) {
        const clips = (run.positives.smartdoc ?? []).filter(({ clip }) => splitOf(clip) === split);
        if (!clips.length) continue;
        const of = (keep: (c: ClipResult) => boolean) =>
          ji(clips.filter(({ clip }) => keep(clip)).flatMap((c) => c.metrics));
        rows.push([
          `${split === 'tune' ? 'tuning' : 'held-out'}: ${run.label}`,
          ...backgrounds.map((bg) => of((c) => c.background === bg)),
          of((c) => c.background !== BG5),
          of(() => true),
        ]);
      }
    }
    out.push(table(['split: run', ...backgrounds, 'backgrounds 1–4', 'all'], rows));
  }

  const withNegatives = runs.filter((run) => run.negatives);
  if (withNegatives.length) {
    out.push('### No-document negatives (share of images with a returned quad)');
    out.push(
      table(
        ['run', 'images', 'any quad', ...thresholds.map((t) => `score ≥ ${t}`), 'ms'],
        withNegatives.map((run) => {
          const negatives = run.negatives as NegativeMetrics[];
          return [
            run.label,
            String(negatives.length),
            pct(falsePositiveRate(negatives)),
            ...thresholds.map((t) => pct(falsePositiveRate(negatives, t))),
            num(negatives.reduce((s, m) => s + m.ms, 0) / negatives.length, 1),
          ];
        }),
      ),
    );
  }
  return out;
}

function msCell(metrics: FrameMetrics[] | null): string {
  return metrics ? num(aggregate(metrics).msMean, 1) : '–';
}
