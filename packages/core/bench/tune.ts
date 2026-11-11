/**
 * Tuning-split scores of one or more diagnostic runs (made with `--min-score 0`), for the
 * experiment log in docs/benchmarks.md. Reads only the tuning subsets (SmartDoc tuning split,
 * MIDV-500 tuning half) and the negatives, never held-out data.
 *
 * Usage: `node bench/tune.ts --runs a,b [--size 384] [--fp 0.10,0.05,0.02]`
 *
 * Per run and tuning subset it prints the JI of the best candidate without a threshold ("any"),
 * and the JI at the score thresholds where 10 %, 5 % and 2 % of the negatives yield a quad
 * ("FP10 / FP5 / FP2", on the calibration half of the negatives; frames whose best candidate scores below the threshold count as misses).
 * The objective is the mean of the SmartDoc and MIDV-500 tuning JI, averaged over the thresholds
 * for FP rates of 4, 5, 6, 7 and 8 %. With `--at`, it also
 * prints the JI, precision (JI ≥ 0.9 among returned quads) and the negatives rate at fixed
 * thresholds, for calibrating `DEFAULT_MIN_SCORE` and `CONFIDENT_SCORE`.
 */
import { parseArgs } from 'node:util';
import type { DatasetName } from './datasets.ts';
import { clipMetrics, type FrameMetrics, negativeMetrics } from './report-data.ts';
import { loadRun } from './results.ts';
import { negativeSplitOf, splitOf } from './splits.ts';

const { values } = parseArgs({
  options: {
    runs: { type: 'string' },
    size: { type: 'string', default: '384' },
    fp: { type: 'string', default: '0.10,0.05,0.02' },
    at: { type: 'string' },
    by: { type: 'string' },
  },
});
if (!values.runs) throw new Error('--runs is required');
const size = Number(values.size);
const fpRates = values.fp.split(',').map(Number);

/** Lowest threshold at which at most `rate` of the negatives keep a quad. */
function thresholdFor(scores: number[], rate: number): number {
  const sorted = [...scores].sort((a, b) => b - a);
  const allowed = Math.floor(rate * sorted.length);
  // Scores strictly above sorted[allowed] are allowed; quads need score >= threshold.
  return allowed < sorted.length ? sorted[allowed] + 1e-6 : 0;
}

function jiAt(metrics: FrameMetrics[], threshold: number): number {
  let sum = 0;
  for (const m of metrics) if (m.bestScore >= threshold && m.bestScore > 0) sum += m.ji;
  return sum / metrics.length;
}

interface Tuning {
  smartdoc: FrameMetrics[];
  midv500: FrameMetrics[];
  negatives: number[];
}

function load(run: string): Tuning {
  const subset = (dataset: DatasetName) =>
    loadRun(run, dataset)
      .filter((c) => splitOf(c) === 'tune')
      .flatMap((clip) => {
        if (clip.minScore !== 0) throw new Error(`${run}: ${clip.id} not run with --min-score 0`);
        return clipMetrics(clip, size, 'frame').map((m, i) => ({
          ...m,
          background: clip.background,
          index: i,
        }));
      });
  // Only the calibration half of the negatives (the report half is for reporting).
  const negatives = loadRun(run, 'negatives').flatMap((clip) =>
    negativeMetrics(clip, size)
      .filter((m) => negativeSplitOf(m.index) === 'calibration')
      .map((m) => (m.found ? m.score : 0)),
  );
  return { smartdoc: subset('smartdoc'), midv500: subset('midv500'), negatives };
}

const f3 = (v: number) => v.toFixed(3);
const rows: string[] = [];
for (const run of values.runs.split(',')) {
  const t = load(run);
  const thresholds = fpRates.map((r) => thresholdFor(t.negatives, r));
  const line = (metrics: FrameMetrics[]) =>
    `${f3(jiAt(metrics, 0))} / ${thresholds.map((th) => f3(jiAt(metrics, th))).join(' / ')}`;
  // Mean of both subsets, averaged over FP rates 4–8 % (a single FP rate depends on a handful of
  // calibration images).
  const band = [0.04, 0.05, 0.06, 0.07, 0.08].map((r) => thresholdFor(t.negatives, r));
  const objective =
    band.reduce((sum, th) => sum + (jiAt(t.smartdoc, th) + jiAt(t.midv500, th)) / 2, 0) /
      band.length || Number.NaN;
  rows.push(
    `| ${run} | ${line(t.smartdoc)} | ${line(t.midv500)} | ${f3(objective)} | ${thresholds.map(f3).join(' / ')} |`,
  );
  if (values.by) {
    // Per background (SmartDoc) and condition (MIDV-500), "any" and FP5.
    for (const [name, metrics] of [
      ['smartdoc', t.smartdoc],
      ['midv500', t.midv500],
    ] as const) {
      const groups = new Map<string, FrameMetrics[]>();
      for (const m of metrics) {
        const key = (m as FrameMetrics & { background: string }).background;
        groups.set(key, [...(groups.get(key) ?? []), m]);
      }
      const cells = [...groups]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([g, ms]) => `${g} ${f3(jiAt(ms, 0))}/${f3(jiAt(ms, thresholds[1]))}`);
      rows.push(`|   ${name} | ${cells.join(' · ')} |`);
    }
  }
  if (values.at) {
    for (const th of values.at.split(',').map(Number)) {
      const neg = t.negatives.filter((s) => s >= th && s > 0).length / t.negatives.length;
      const cell = (metrics: FrameMetrics[]) => {
        const returned = metrics.filter((m) => m.bestScore >= th && m.bestScore > 0);
        const right = returned.filter((m) => m.ji >= 0.9).length;
        return `JI ${f3(jiAt(metrics, th))}, right ${((100 * right) / returned.length).toFixed(1)} % of returned, recall ${((100 * right) / metrics.length).toFixed(1)} %, no quad ${((100 * (metrics.length - returned.length)) / metrics.length).toFixed(1)} %`;
      };
      rows.push(
        `|   at ${th} | negatives ${(100 * neg).toFixed(1)} % | SmartDoc ${cell(t.smartdoc)} | MIDV-500 ${cell(t.midv500)} |`,
      );
    }
  }
}
console.log(
  '| run | SmartDoc tune: any / FP10 / FP5 / FP2 | MIDV-500 tune: any / FP10 / FP5 / FP2 | objective (mean JI, FP 4–8 %) | thresholds FP10 / FP5 / FP2 |',
);
console.log('| --- | --- | --- | ---: | --- |');
for (const row of rows) console.log(row);
