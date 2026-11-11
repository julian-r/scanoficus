/**
 * Markdown tables from the raw results of `pnpm bench`.
 *
 * Usage: `pnpm bench:report [--run full] [--datasets smartdoc,midv500,midv2019] [--detail 384]
 *   [--diagnostic <run>] [--thresholds 0.6,0.7,0.8]`
 *   or `pnpm bench:report --runs full,relaxed,nn-x@256 [--out <file>]`
 *
 * With `--run`, writes `eval-out/bench/<run>/report.md` (summary by split, negatives, then the
 * detailed tables per dataset) and prints it. With `--runs`, compares several runs side by side
 * (summary by split and negatives only) and writes `eval-out/bench/compare.md` or `--out`. A run
 * is evaluated at `--detail` px if it has that size, else at its first size; `name@size` picks
 * the size explicitly. Metrics (see docs/benchmarks.md):
 * SmartDoc JI (official Challenge 1 Jaccard index, mean over frames, misses score 0), visible-
 * region image IoU and the shares of frames with IoU ≥ 0.9 / 0.95 (misses count as failures),
 * mean / p90 corner error in % of the image diagonal over frames with a detection, detection
 * rate and the time of the `detectDocument` call. "tracked" runs the library's QuadTracker
 * (default options) over the per-frame detections of each clip.
 *
 * `--diagnostic` names a run of the same frames at the detail size with relaxed detector options
 * (e.g. `--min-area 0.02 --min-score 0`); failures are then split into causes.
 */
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { DATASETS, type DatasetName } from './datasets.ts';
import {
  aggregate,
  clipMetrics,
  type FrameMetrics,
  METRIC_HEADER,
  metricCells,
  num,
  pct,
  table,
} from './report-data.ts';
import { parseRunSpec, summarySections } from './report-summary.ts';
import { type ClipResult, loadRun, runDir } from './results.ts';

const { values } = parseArgs({
  options: {
    run: { type: 'string', default: 'full' },
    datasets: { type: 'string', default: DATASETS.join(',') },
    detail: { type: 'string', default: '384' },
    diagnostic: { type: 'string' },
    runs: { type: 'string' },
    out: { type: 'string' },
    thresholds: { type: 'string', default: '0.6,0.7,0.8' },
  },
});
const detail = Number(values.detail);
const thresholds = values.thresholds.split(',').map(Number);
if (values.runs) {
  const specs = values.runs.split(',').map(parseRunSpec);
  const markdown = `${['# Detector comparison', ...summarySections(specs, detail, thresholds)].join('\n\n')}\n`;
  const out = values.out
    ? resolve(process.env.INIT_CWD ?? process.cwd(), values.out)
    : new URL('../eval-out/bench/compare.md', import.meta.url);
  writeFileSync(out, markdown);
  console.log(markdown);
  process.exit(0);
}
/** `detectDocument`'s default `minArea` (fraction of the frame). */
const DEFAULT_MIN_AREA = 0.04;
const TITLES: Record<DatasetName, string> = {
  smartdoc: 'SmartDoc 2015 Challenge 1',
  midv500: 'MIDV-500',
  midv2019: 'MIDV-2019',
  negatives: 'No-document negatives',
};

interface Labeled {
  clip: ClipResult;
  metrics: FrameMetrics[];
}

function groupBy(items: Labeled[], key: (clip: ClipResult) => string): [string, Labeled[]][] {
  const groups = new Map<string, Labeled[]>();
  for (const item of items) {
    const k = key(item.clip);
    groups.set(k, [...(groups.get(k) ?? []), item]);
  }
  return [...groups].sort(([a], [b]) => a.localeCompare(b));
}

const frames = (items: Labeled[]) => items.flatMap((l) => l.metrics);
const success = (m: FrameMetrics, smartdoc: boolean) => (smartdoc ? m.ji : m.iou) >= 0.9;
const candidateSuccess = (m: FrameMetrics, smartdoc: boolean) =>
  (smartdoc ? m.oracleJi : m.oracleIou) >= 0.9;

/**
 * Outcome of each frame: success, no detection, a wrong quad while one of the runner-up
 * candidates was right (ranking error), or a wrong quad without any right candidate.
 */
function outcomes(metrics: FrameMetrics[], smartdoc: boolean): string[] {
  let ok = 0;
  let missed = 0;
  let ranking = 0;
  let wrong = 0;
  for (const m of metrics) {
    if (success(m, smartdoc)) ok++;
    else if (!m.found) missed++;
    else if (candidateSuccess(m, smartdoc)) ranking++;
    else wrong++;
  }
  const n = metrics.length;
  return [pct(ok / n), pct(missed / n), pct(ranking / n), pct(wrong / n)];
}

/**
 * Splits the failures of the default run into causes, in this order: document smaller than the
 * default `minArea`; a wrong quad won although the right one was a runner-up candidate; the
 * relaxed diagnostic run's best quad is right (so only the score threshold rejected it); the
 * right quad is among the relaxed run's candidates but not first; no right candidate at all.
 */
function causes(main: FrameMetrics[], relaxed: FrameMetrics[], smartdoc: boolean): string[] {
  const counts = [0, 0, 0, 0, 0];
  let failures = 0;
  for (let i = 0; i < main.length; i++) {
    const m = main[i];
    if (success(m, smartdoc)) continue;
    failures++;
    if (m.area < DEFAULT_MIN_AREA) counts[0]++;
    else if (m.found && candidateSuccess(m, smartdoc)) counts[1]++;
    else if (success(relaxed[i], smartdoc)) counts[2]++;
    else if (candidateSuccess(relaxed[i], smartdoc)) counts[3]++;
    else counts[4]++;
  }
  return [String(failures), ...counts.map((c) => pct(failures ? c / failures : Number.NaN))];
}

const sections: string[] = [
  `# Detector benchmark — run \`${values.run}\``,
  ...summarySections([{ name: values.run }], detail, thresholds),
];
for (const dataset of values.datasets.split(',') as DatasetName[]) {
  if (dataset === 'negatives') continue; // in the summary
  const clips = loadRun(values.run, dataset);
  if (!clips.length) continue;
  const smartdoc = dataset === 'smartdoc';
  const sizes = clips[0].sizes;
  const steps = [...new Set(clips.map((c) => c.step))];
  const out: string[] = [];
  const cache = new Map<string, Labeled[]>();
  const all = (size: number, variant: 'frame' | 'tracked'): Labeled[] => {
    const key = `${size} ${variant}`;
    let labeled = cache.get(key);
    if (!labeled) {
      labeled = clips.map((clip) => ({ clip, metrics: clipMetrics(clip, size, variant) }));
      cache.set(key, labeled);
    }
    return labeled;
  };
  const frameCount = clips.reduce((s, c) => s + c.frames.length, 0);
  out.push(
    `## ${TITLES[dataset]}\n\n${clips.length} clips, ${frameCount} frames (every ${steps.join('/')}. frame).`,
  );

  out.push('### Overall');
  const overallRows: string[][] = [];
  for (const size of sizes) {
    for (const variant of ['frame', 'tracked'] as const) {
      overallRows.push([
        `${size} ${variant}`,
        ...metricCells(aggregate(frames(all(size, variant)))),
      ]);
    }
  }
  out.push(table(['long side', ...METRIC_HEADER], overallRows));

  const label = smartdoc ? 'background' : 'condition';
  out.push(`### JI per ${label}`);
  const jiRow = (name: string, pick: (items: Labeled[]) => Labeled[]) => [
    name,
    String(frames(pick(all(sizes[0], 'frame'))).length),
    ...sizes.map((size) => num(aggregate(frames(pick(all(size, 'frame')))).ji)),
    num(aggregate(frames(pick(all(detail, 'tracked')))).ji),
  ];
  const jiRows = groupBy(all(sizes[0], 'frame'), (c) => c.background).map(([k]) =>
    jiRow(k, (items) => items.filter((l) => l.clip.background === k)),
  );
  if (smartdoc) {
    // Some published methods report backgrounds 1–4 only.
    jiRows.push(
      jiRow('backgrounds 1–4', (items) =>
        items.filter((l) => l.clip.background !== 'background05'),
      ),
    );
  }
  jiRows.push(jiRow('all', (items) => items));
  out.push(table([label, 'frames', ...sizes.map(String), `${detail} tracked`], jiRows));

  const detailed = all(detail, 'frame');
  out.push(`### Per ${label} at ${detail}`);
  out.push(
    table(
      [label, ...METRIC_HEADER],
      groupBy(detailed, (c) => c.background).map(([k, items]) => [
        k,
        ...metricCells(aggregate(frames(items))),
      ]),
    ),
  );
  out.push(`### Per document type at ${detail}`);
  out.push(
    table(
      ['document type', ...METRIC_HEADER],
      groupBy(detailed, (c) => c.docType).map(([k, items]) => [
        k,
        ...metricCells(aggregate(frames(items))),
      ]),
    ),
  );
  if (!smartdoc) {
    // The subsets used by Tropin et al. (2020, 2021) for MIDV-500.
    out.push(`### By ground-truth corners inside the frame at ${detail}`);
    const metrics = frames(detailed);
    out.push(
      table(
        ['subset', ...METRIC_HEADER],
        [
          ['all frames', ...metricCells(aggregate(metrics))],
          [
            'at least 3 corners inside',
            ...metricCells(aggregate(metrics.filter((m) => m.cornersInside >= 3))),
          ],
          [
            'all 4 corners inside',
            ...metricCells(aggregate(metrics.filter((m) => m.cornersInside === 4))),
          ],
        ],
      ),
    );
  }
  const criterion = smartdoc ? 'JI ≥ 0.9' : 'IoU ≥ 0.9';
  out.push(`### Outcomes per ${label} at ${detail} (success: ${criterion})`);
  out.push(
    table(
      [
        label,
        'success',
        'no detection',
        'wrong quad, right one among candidates',
        'wrong quad, no right candidate',
      ],
      [
        ...groupBy(detailed, (c) => c.background).map(([k, items]) => [
          k,
          ...outcomes(frames(items), smartdoc),
        ]),
        ['all', ...outcomes(frames(detailed), smartdoc)],
      ],
    ),
  );

  const relaxedClips = values.diagnostic
    ? new Map(loadRun(values.diagnostic, dataset).map((c) => [c.id, c]))
    : new Map<string, ClipResult>();
  if (relaxedClips.size) {
    const relaxedOf = (items: Labeled[]) =>
      items.flatMap(({ clip }) => {
        const other = relaxedClips.get(clip.id);
        if (!other || other.frames.length !== clip.frames.length || !other.sizes.includes(detail)) {
          throw new Error(`diagnostic run ${values.diagnostic} lacks the frames of ${clip.id}`);
        }
        return clipMetrics(other, detail, 'frame');
      });
    const options = [...relaxedClips.values()][0];
    out.push(
      `### Failure causes at ${detail} (share of failing frames; diagnostic run \`${values.diagnostic}\`: minArea ${options.minArea ?? 'default'}, minScore ${options.minScore ?? 'default'})`,
    );
    out.push(
      table(
        [
          label,
          'failures',
          `document < ${DEFAULT_MIN_AREA * 100} % of frame`,
          'right quad ranked below a wrong one',
          'right quad rejected by score threshold',
          'right quad among relaxed candidates only',
          'no right candidate',
        ],
        [
          ...groupBy(detailed, (c) => c.background).map(([k, items]) => [
            k,
            ...causes(frames(items), relaxedOf(items), smartdoc),
          ]),
          ['all', ...causes(frames(detailed), relaxedOf(detailed), smartdoc)],
        ],
      ),
    );
  }
  sections.push(out.join('\n\n'));
}

const markdown = `${sections.join('\n\n')}\n`;
writeFileSync(new URL('report.md', runDir(values.run)), markdown);
console.log(markdown);
