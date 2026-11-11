/**
 * Side-by-side summary of one or more benchmark runs (one run directory per detector or
 * variant), split into the tuning and held-out parts of the protocol in `splits.ts`, plus the
 * false-positive rate on the no-document negatives and per-flag results on the photo sets.
 */
import { DATASETS, type DatasetName } from './datasets.ts';
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
  rateWithInterval,
  table,
  thresholdAtFalsePositiveRate,
} from './report-data.ts';
import { type ClipResult, loadRun } from './results.ts';
import { isMidvTuningType, negativeSplitOf, splitOf } from './splits.ts';

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
  clips: Partial<Record<DatasetName, ClipResult[]>>;
}

function loadSpec(spec: RunSpec, preferred: number): LoadedRun {
  const loaded: LoadedRun = { label: spec.name, sizes: {}, clips: {} };
  for (const dataset of DATASETS) {
    const clips = loadRun(spec.name, dataset);
    if (!clips.length) continue;
    const available = clips[0].sizes;
    const size = spec.size ?? (available.includes(preferred) ? preferred : available[0]);
    if (!clips.every((c) => c.sizes.includes(size))) {
      throw new Error(`run ${spec.name}: not every ${dataset} clip has size ${size}`);
    }
    loaded.sizes[dataset] = size;
    loaded.clips[dataset] = clips;
  }
  const sizes = [...new Set(Object.values(loaded.sizes))];
  if (sizes.length) loaded.label = `${spec.name} @${sizes.join('/')}`;
  return loaded;
}

/** Run metrics with an optional post-hoc score threshold (see `clipMetrics`). */
interface Evaluated {
  label: string;
  positives: Partial<Record<DatasetName, { clip: ClipResult; metrics: FrameMetrics[] }[]>>;
  negatives: NegativeMetrics[] | null;
}

function evaluate(run: LoadedRun, minScore: number | undefined): Evaluated {
  const out: Evaluated = { label: run.label, positives: {}, negatives: null };
  for (const dataset of DATASETS) {
    const clips = run.clips[dataset];
    const size = run.sizes[dataset];
    if (!clips || size === undefined) continue;
    if (dataset === 'negatives') {
      out.negatives = clips
        .flatMap((clip) => negativeMetrics(clip, size))
        .map((m) =>
          minScore !== undefined && m.score < minScore ? { ...m, found: false, score: 0 } : m,
        );
    } else {
      out.positives[dataset] = clips.map((clip) => ({
        clip,
        metrics: clipMetrics(clip, size, 'frame', minScore),
      }));
    }
  }
  return out;
}

type Subset = {
  title: string;
  /** Short column title for the summary. */
  short: string;
  role: 'tuning' | 'held out' | 'held out, seen in round 1';
  dataset: DatasetName;
  keep: (clip: ClipResult) => boolean;
};

const BG5 = 'background05';
export const SUBSETS: Subset[] = [
  {
    title: 'SmartDoc tuning split',
    short: 'SmartDoc tune',
    role: 'tuning',
    dataset: 'smartdoc',
    keep: (c) => splitOf(c) === 'tune',
  },
  {
    title: 'MIDV-500 tuning half',
    short: 'MIDV-500 tune',
    role: 'tuning',
    dataset: 'midv500',
    keep: (c) => splitOf(c) === 'tune',
  },
  {
    title: 'SmartDoc held-out split',
    short: 'SmartDoc held-out',
    role: 'held out',
    dataset: 'smartdoc',
    keep: (c) => splitOf(c) === 'heldout',
  },
  {
    title: 'SmartDoc held-out, backgrounds 1–4',
    short: 'SmartDoc held-out bg 1–4',
    role: 'held out',
    dataset: 'smartdoc',
    keep: (c) => splitOf(c) === 'heldout' && c.background !== BG5,
  },
  {
    title: 'MIDV-500 held-out half (aggregates seen in round 1)',
    short: 'MIDV-500 held-out*',
    role: 'held out, seen in round 1',
    dataset: 'midv500',
    keep: (c) => splitOf(c) === 'heldout',
  },
  {
    title: 'MIDV-2019 (held out)',
    short: 'MIDV-2019',
    role: 'held out',
    dataset: 'midv2019',
    keep: () => true,
  },
  {
    title: 'MIDV-2019, MIDV-500 held-out types only',
    short: 'MIDV-2019 held-out types',
    role: 'held out',
    dataset: 'midv2019',
    keep: (c) => !isMidvTuningType(c.docType),
  },
  { title: 'Dizaj (photos)', short: 'Dizaj', role: 'held out', dataset: 'dizaj', keep: () => true },
  { title: 'RWMD (photos)', short: 'RWMD', role: 'held out', dataset: 'rwmd', keep: () => true },
];

/** Flag groups of the photo sets (`gt.jsonl` flags, see `$SCANOFICUS_DATASETS/FORMAT.md`). */
const PHOTO_GROUPS: [string, (flags: string[]) => boolean][] = [
  ['all', () => true],
  ['no flags', (f) => f.length === 0],
  ['partial (a corner outside)', (f) => f.includes('partial')],
  ['multi-document', (f) => f.includes('multi-document')],
  ['occluded or curved (mask-quad-mismatch)', (f) => f.includes('mask-quad-mismatch')],
  ['weak-edge-support (label likely off)', (f) => f.includes('weak-edge-support')],
];

function subsetMetrics(run: Evaluated, subset: Subset): FrameMetrics[] | null {
  const clips = run.positives[subset.dataset];
  if (!clips) return null;
  const metrics = clips.filter((c) => subset.keep(c.clip)).flatMap((c) => c.metrics);
  return metrics.length ? metrics : null;
}

export interface SummaryOptions {
  /** Score thresholds for the negatives table. */
  thresholds: readonly number[];
  /**
   * Post-hoc score thresholds: one summary per entry (`undefined` = the runs as recorded). Valid
   * for runs made with a lower `minScore` (e.g. 0).
   */
  at: readonly (number | undefined)[];
  /**
   * False-positive rates for the equal-FP comparison: per run, the threshold at which this share
   * of the calibration half of the negatives yields a quad (runs made with `minScore` 0).
   */
  fp: readonly number[];
}

const half = (negatives: NegativeMetrics[], which: 'calibration' | 'report') =>
  negatives.filter((m) => negativeSplitOf(m.index) === which);

/**
 * Markdown sections comparing `specs`: per threshold in `at`, a summary (JI per tuning and
 * held-out subset, FP rate) and the share of frames without a quad (escalated in a hybrid), then
 * at the first threshold one metric table per subset, SmartDoc JI per background and MIDV JI per
 * condition and split, the photo sets per flag group, and the negatives at several thresholds.
 */
export function summarySections(
  specs: readonly RunSpec[],
  preferredSize: number,
  options: SummaryOptions,
): string[] {
  const loaded = specs.map((spec) => loadSpec(spec, preferredSize));
  const out: string[] = [];
  const ji = (metrics: FrameMetrics[] | null) => (metrics ? num(aggregate(metrics).ji) : '–');
  const atLabel = (t: number | undefined) =>
    t === undefined ? 'as recorded' : `minScore ${t} applied`;

  const present = SUBSETS.filter((s) => loaded.some((run) => run.clips[s.dataset]?.length));
  for (const t of options.at) {
    const runs = loaded.map((run) => evaluate(run, t));
    out.push(`## Summary (JI, ${atLabel(t)})`);
    out.push(
      'Tuning columns are the ones detector changes are chosen on. Held-out columns are only ' +
        'reported. * = aggregates of the whole MIDV-500 subset were seen in round 1.',
    );
    out.push(
      table(
        [
          'run',
          ...present.map((s) => `${s.short} (${s.role === 'tuning' ? 'tune' : 'held out'})`),
          'negatives, report half: FP rate (95 % CI)',
          'ms (SmartDoc)',
        ],
        runs.map((run) => [
          run.label,
          ...present.map((s) => ji(subsetMetrics(run, s))),
          run.negatives ? rateWithInterval(half(run.negatives, 'report')) : '–',
          msCell(subsetMetrics(run, present.find((s) => s.dataset === 'smartdoc') ?? SUBSETS[0])),
        ]),
      ),
    );
    out.push(
      `### Share of frames without a returned quad (escalated to a learned detector in a hybrid), ${atLabel(t)}`,
    );
    out.push(
      table(
        ['run', ...present.map((s) => s.short), 'negatives'],
        runs.map((run) => [
          run.label,
          ...present.map((s) => {
            const m = subsetMetrics(run, s);
            return m ? pct(1 - aggregate(m).found) : '–';
          }),
          run.negatives ? pct(1 - falsePositiveRate(run.negatives)) : '–',
        ]),
      ),
    );
  }

  for (const rate of options.fp) {
    const rows: string[][] = [];
    for (const run of loaded) {
      const negatives = evaluate(run, undefined).negatives;
      const relaxed = Object.values(run.clips).every((clips) =>
        (clips ?? []).every((c) => c.minScore === 0),
      );
      if (!negatives || !relaxed) continue;
      const threshold = thresholdAtFalsePositiveRate(half(negatives, 'calibration'), rate);
      const at = evaluate(run, threshold);
      rows.push([
        run.label,
        num(threshold),
        ...present.map((s) => ji(subsetMetrics(at, s))),
        rateWithInterval(half(negatives, 'report'), threshold),
      ]);
    }
    if (!rows.length) continue;
    out.push(
      `## JI at an equal false-positive rate (${pct(rate, 0)} of the calibration negatives)`,
    );
    out.push(
      'Per run, the score threshold is set so that this share of the calibration half of the ' +
        'negatives (even COCO ids) yields a quad; the last column is the rate on the report half ' +
        '(odd ids). Only runs made with `minScore` 0.',
    );
    out.push(
      table(
        [
          'run',
          'threshold',
          ...present.map((s) => s.short),
          'negatives, report half: FP rate (95 % CI)',
        ],
        rows,
      ),
    );
  }

  const runs = loaded.map((run) => evaluate(run, options.at[0]));
  out.push(`## Details (${atLabel(options.at[0])})`);
  for (const subset of present) {
    const rows = runs
      .map((run) => [run, subsetMetrics(run, subset)] as const)
      .filter((entry): entry is readonly [Evaluated, FrameMetrics[]] => entry[1] !== null)
      .map(([run, metrics]) => [run.label, ...metricCells(aggregate(metrics))]);
    if (!rows.length) continue;
    out.push(`### ${subset.title}`);
    out.push(table(['run', ...METRIC_HEADER], rows));
  }

  for (const dataset of ['smartdoc', 'midv500', 'midv2019'] as const) {
    const groups = [
      ...new Set(
        runs.flatMap((run) => (run.positives[dataset] ?? []).map(({ clip }) => clip.background)),
      ),
    ].sort();
    if (!groups.length) continue;
    const smartdoc = dataset === 'smartdoc';
    out.push(
      `### ${smartdoc ? 'SmartDoc JI per background' : `${dataset === 'midv500' ? 'MIDV-500' : 'MIDV-2019'} JI per condition`} and split`,
    );
    const rows: string[][] = [];
    const parts: [string, (c: ClipResult) => boolean][] =
      dataset === 'midv2019'
        ? [
            ['MIDV-500 tuning types', (c) => isMidvTuningType(c.docType)],
            ['MIDV-500 held-out types', (c) => !isMidvTuningType(c.docType)],
          ]
        : [
            ['tuning', (c) => splitOf(c) === 'tune'],
            ['held-out', (c) => splitOf(c) === 'heldout'],
          ];
    for (const [name, keepPart] of parts) {
      for (const run of runs) {
        const clips = (run.positives[dataset] ?? []).filter(({ clip }) => keepPart(clip));
        if (!clips.length) continue;
        const of = (keep: (c: ClipResult) => boolean) =>
          ji(clips.filter(({ clip }) => keep(clip)).flatMap((c) => c.metrics));
        rows.push([
          `${name}: ${run.label}`,
          ...groups.map((g) => of((c) => c.background === g)),
          ...(smartdoc ? [of((c) => c.background !== BG5)] : []),
          of(() => true),
        ]);
      }
    }
    out.push(
      table(
        [
          'split: run',
          ...groups.map((g) => g.replace('background', 'bg ')),
          ...(smartdoc ? ['bg 1–4'] : []),
          'all',
        ],
        rows,
      ),
    );
  }

  for (const dataset of ['dizaj', 'rwmd'] as const) {
    const rows: string[][] = [];
    for (const [group, keep] of PHOTO_GROUPS) {
      for (const run of runs) {
        const clips = run.positives[dataset];
        if (!clips) continue;
        const metrics = clips.flatMap((c) => c.metrics).filter((m) => keep(m.flags ?? []));
        if (!metrics.length) continue;
        rows.push([`${group}: ${run.label}`, ...metricCells(aggregate(metrics))]);
      }
    }
    if (!rows.length) continue;
    out.push(`### ${dataset === 'dizaj' ? 'Dizaj' : 'RWMD'} per flag group (held out)`);
    out.push(
      'Photo metrics: JI with a per-photo reference rectangle, IoU of the full quads (not ' +
        'clipped to the image), as the Python photo scorer.',
    );
    out.push(table(['group: run', ...METRIC_HEADER], rows));
  }

  const withNegatives = runs.filter((run) => run.negatives);
  if (withNegatives.length) {
    out.push(
      '### No-document negatives (share of images with a returned quad; calibration half = even COCO ids, report half = odd ids)',
    );
    out.push(
      table(
        ['run: half', 'images', 'any quad', ...options.thresholds.map((t) => `score ≥ ${t}`), 'ms'],
        withNegatives.flatMap((run) => {
          const all = run.negatives as NegativeMetrics[];
          return (['report', 'calibration'] as const).map((which) => {
            const negatives = half(all, which);
            return [
              `${run.label}: ${which}`,
              String(negatives.length),
              rateWithInterval(negatives),
              ...options.thresholds.map((t) => pct(falsePositiveRate(negatives, t))),
              num(negatives.reduce((s, m) => s + m.ms, 0) / negatives.length, 1),
            ];
          });
        }),
      ),
    );
  }
  return out;
}

function msCell(metrics: FrameMetrics[] | null): string {
  return metrics ? num(aggregate(metrics).msMean, 1) : '–';
}
