/**
 * Offline auto-capture simulator: replays per-frame detector output of real hand-held video
 * clips through an auto-capture policy and scores the frame it would capture.
 *
 * Usage: `pnpm bench:autocapture [--bench-dir <eval-out/bench>] [--detector nn|classic]
 *   [--rate 17] [--latency 60] [--policies 'legacy;library;lib:{…};cand:{…}'] [--fast 0.25]
 *   [--json <file>]` (run `pnpm bench:autocapture:features` once before).
 *
 * Policies (`;`-separated): `legacy` (before the rework), `library` (the shipped defaults),
 * `lib:{"tracker":{…},"trigger":{…},"bestFrameMs":…}` (the library with options) and
 * `cand:{…}` (the design-study policy with {@link CandidateOptions}).
 *
 * Inputs (all read-only):
 * - detections of the benchmark runs in {@link AUTOCAPTURE_RUNS} (`--bench-dir`, default
 *   `$SCANOFICUS_BENCH_DIR` or this package's `eval-out/bench/`), with the detector's presence
 *   threshold applied;
 * - per-frame sharpness from `autocapture-features.ts` (run it first);
 * - ground truth from the datasets (`$SCANOFICUS_DATASETS`).
 *
 * Timing. SmartDoc 2015 Ch.1 clips have a variable frame rate of about 17–20 fps and no
 * timestamps; frames are taken as 50 ms apart (20 fps, as in docs/benchmarks.md). MIDV-500 and
 * MIDV-2019 frames were extracted at 10 fps (`-r 10`, dataset readme): 100 ms apart, 3 s per
 * clip. The live loop is simulated at `--rate` detections per second (default 17, as measured
 * for "Neural v4" on an iPhone): at every tick the newest frame is analyzed, so SmartDoc is
 * subsampled; MIDV is slower than the live rate and every frame is analyzed once (10 Hz).
 * `--rate 100` analyzes every frame of both.
 *
 * Capture. When a policy fires on the frame analyzed at time t, the controller grabs the
 * current video frame as the still: the frame nearest to t + `--latency` (default 60 ms, one
 * detection). A policy can instead name an analyzed frame (a frame buffer). The page quad is
 * the detection on the captured frame, else the tracked quad. Only the first capture of a clip
 * is scored (the controller then waits for a different document).
 *
 * Metrics per clip group: share of clips with a capture, median time to the first capture,
 * mean JI of the captured page against the ground truth (SmartDoc Jaccard index), share of bad
 * captures (JI < 0.9), median sharpness of the captured frame relative to the clip's sharpest
 * frame (ground-truth region at 768 px), and the share of captures during fast motion (ground
 * truth corners moving faster than `--fast` diagonals per second).
 *
 * The datasets were filmed while moving the camera (the median ground-truth corner speed is
 * 0.2–0.3 diagonals per second; SmartDoc has no 0.5-s period below 0.05/s), so few clips offer
 * a steady moment. A second, semi-synthetic replay ({@link handClip}) therefore keeps each
 * frame's real detection error, misses and sharpness but replaces the camera motion by a
 * synthetic hand: steady, slowly drifting, panning or walking.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { distance, type Quad } from '../src/geometry/quad.ts';
import { Random } from '../test/support/random.ts';
import {
  AUTOCAPTURE_RUNS,
  type AutocaptureDetector,
  type ClipFeatures,
  defaultBenchDir,
  featuresPath,
  loadClipRun,
  VIDEO_DATASETS,
} from './autocapture-features.ts';
import {
  type CandidateOptions,
  CandidatePolicy,
  LegacyPolicy,
  LibraryPolicy,
  type Policy,
  percentile,
} from './autocapture-policies.ts';
import { loadClips } from './datasets.ts';
import { smartDocJaccard } from './metrics.ts';
import { unpackQuad } from './results.ts';
import { splitOf } from './splits.ts';

export interface SimFrame {
  t: number;
  truth: Quad;
  quad: Quad | null;
  sharpness: number;
  truthSharpness: number;
  /** Ground-truth corner speed in diagonals per second. */
  speed: number;
}

export interface SimClip {
  dataset: string;
  id: string;
  group: string;
  tuning: boolean;
  reference: { width: number; height: number };
  frames: SimFrame[];
  bestSharpness: number;
}

/** Ground-truth corner speed of every frame (central difference), in diagonals per second. */
function setSpeeds(frames: SimFrame[]): void {
  for (let i = 0; i < frames.length; i++) {
    const a = frames[Math.max(0, i - 1)];
    const b = frames[Math.min(frames.length - 1, i + 1)];
    const d = Math.max(...a.truth.map((p, k) => distance(p, b.truth[k])));
    frames[i].speed = b.t > a.t ? d / diagonal(frames[i].truth) / ((b.t - a.t) / 1000) : 0;
  }
}

function diagonal(q: Quad): number {
  return Math.max(distance(q[0], q[2]), distance(q[1], q[3]));
}

/**
 * Hand motion of the semi-synthetic replay, as offsets in diagonals over time (s):
 * - `steady`: a hand that tries to hold still: slow sway (amplitude 1 %, 0.3–0.6 Hz, peak speed
 *   ≈ 0.02–0.04 diagonals/s) and tremor (0.1 %, 8–10 Hz);
 * - `drift`: steady plus a slow constant drift of 0.05 diagonals/s;
 * - `pan`: steady plus panning at 0.3 diagonals/s (moving to the next page);
 * - `walk`: steady plus walking bob (amplitude 4 %, 1.5–2 Hz, peak ≈ 0.4–0.5 diagonals/s).
 */
export type HandScenario = 'steady' | 'drift' | 'pan' | 'walk';
export const HAND_SCENARIOS: readonly HandScenario[] = ['steady', 'drift', 'pan', 'walk'];

function handMotion(
  scenario: HandScenario,
  random: Random,
): (t: number) => { x: number; y: number } {
  const sway = [0, 1].map(() => ({ f: random.range(0.3, 0.6), p: random.range(0, 2 * Math.PI) }));
  const tremor = [0, 1].map(() => ({ f: random.range(8, 10), p: random.range(0, 2 * Math.PI) }));
  const bob = { f: random.range(1.5, 2), p: random.range(0, 2 * Math.PI) };
  const angle = random.range(0, 2 * Math.PI);
  const linear = scenario === 'drift' ? 0.05 : scenario === 'pan' ? 0.3 : 0;
  return (t) => {
    const axis = (k: number) =>
      0.01 * Math.sin(2 * Math.PI * sway[k].f * t + sway[k].p) +
      0.001 * Math.sin(2 * Math.PI * tremor[k].f * t + tremor[k].p);
    const walk = scenario === 'walk' ? 0.04 * Math.sin(2 * Math.PI * bob.f * t + bob.p) : 0;
    return {
      x: axis(0) + linear * t * Math.cos(angle),
      y: axis(1) + linear * t * Math.sin(angle) + walk,
    };
  };
}

/**
 * Semi-synthetic replay: the document stays where it is in the clip's middle frame and moves
 * only by a synthetic hand motion; the detections keep their real per-frame errors (detection
 * minus ground truth, corners aligned), and the frames keep their real sharpness and missed
 * detections. This shows how a policy copes with the real detector noise when the hand is as
 * steady (or as fast) as the scenario says.
 */
export function handClip(clip: SimClip, scenario: HandScenario): SimClip {
  const random = new Random(hashString(clip.id + scenario));
  const motion = handMotion(scenario, random);
  const base = clip.frames[Math.floor(clip.frames.length / 2)].truth;
  const diag = diagonal(base);
  const frames: SimFrame[] = clip.frames.map((f) => {
    const offset = motion(f.t / 1000);
    const truth = base.map((p) => ({
      x: p.x + offset.x * diag,
      y: p.y + offset.y * diag,
    })) as unknown as Quad;
    let quad: Quad | null = null;
    if (f.quad) {
      const aligned = alignCorners(f.quad, f.truth);
      quad = truth.map((p, i) => ({
        x: p.x + aligned[i].x - f.truth[i].x,
        y: p.y + aligned[i].y - f.truth[i].y,
      })) as unknown as Quad;
    }
    return { ...f, truth, quad };
  });
  setSpeeds(frames);
  return { ...clip, frames };
}

/** The clip with an extra share `p` of detections removed at random (missed detections). */
export function withMisses(clip: SimClip, p: number): SimClip {
  const random = new Random(hashString(`${clip.id}misses`));
  return { ...clip, frames: clip.frames.map((f) => (random.chance(p) ? { ...f, quad: null } : f)) };
}

function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

function alignCorners(quad: Quad, reference: Quad): Quad {
  let best = quad;
  let bestCost = Number.POSITIVE_INFINITY;
  for (let r = 0; r < 4; r++) {
    const rotated: Quad = [quad[r], quad[(r + 1) % 4], quad[(r + 2) % 4], quad[(r + 3) % 4]];
    let cost = 0;
    for (let i = 0; i < 4; i++) cost += distance(rotated[i], reference[i]);
    if (cost < bestCost) {
      bestCost = cost;
      best = rotated;
    }
  }
  return best;
}

export interface ClipOutcome {
  captured: boolean;
  time: number;
  ji: number;
  relSharpness: number;
  speed: number;
}

export const GROUPS = [
  'SmartDoc tune',
  'MIDV-500 tune',
  'SmartDoc held-out',
  'MIDV-500 held-out*',
  'MIDV-2019 held-out',
] as const;

export function loadSimClips(detector: AutocaptureDetector, benchDir: string): SimClip[] {
  const config = AUTOCAPTURE_RUNS[detector];
  const clips: SimClip[] = [];
  for (const dataset of VIDEO_DATASETS) {
    const interval = dataset === 'smartdoc' ? 50 : 100;
    const name: string = (
      { smartdoc: 'SmartDoc', midv500: 'MIDV-500', midv2019: 'MIDV-2019' } as Record<string, string>
    )[dataset];
    for (const clip of loadClips(dataset)) {
      const run = loadClipRun(benchDir, config.run, dataset, clip.id);
      const features = JSON.parse(
        readFileSync(featuresPath(dataset, clip.id), 'utf8'),
      ) as ClipFeatures;
      const split = splitOf(clip);
      const frames: SimFrame[] = clip.frames.map((f, i) => {
        const r = run.frames[i].runs[config.size];
        return {
          t: i * interval,
          truth: f.quad as Quad,
          quad: r?.quad && r.score >= config.minScore ? unpackQuad(r.quad) : null,
          sharpness: features[detector][i],
          truthSharpness: features.truth[i],
          speed: 0,
        };
      });
      setSpeeds(frames);
      clips.push({
        dataset,
        id: clip.id,
        group: `${name} ${split === 'tune' ? 'tune' : dataset === 'midv500' ? 'held-out*' : 'held-out'}`,
        tuning: split === 'tune',
        reference: clip.reference,
        frames,
        bestSharpness: Math.max(...frames.map((f) => f.truthSharpness)),
      });
    }
  }
  return clips;
}

/** Indices of the frames the live loop analyzes at `rate` detections per second. */
export function analyzedFrames(clip: SimClip, rate: number): number[] {
  const frames = clip.frames;
  const interval = frames.length > 1 ? frames[1].t - frames[0].t : 1;
  const tickMs = Math.max(1000 / rate, interval);
  const end = frames[frames.length - 1].t;
  const indices: number[] = [];
  for (let tick = 0; tick <= end + 1e-6; tick += tickMs) {
    const index = Math.min(frames.length - 1, Math.floor(tick / interval + 1e-9));
    if (index !== indices.at(-1)) indices.push(index);
  }
  return indices;
}

export function simulate(
  clip: SimClip,
  policy: Policy,
  rate: number,
  latency: number,
): ClipOutcome {
  const frames = clip.frames;
  const interval = frames.length > 1 ? frames[1].t - frames[0].t : 1;
  for (const index of analyzedFrames(clip, rate)) {
    const f = frames[index];
    const decision = policy.step({ t: f.t, frame: index, quad: f.quad, sharpness: f.sharpness });
    if (!decision.fire) continue;
    const captured =
      decision.captureFrame ?? Math.min(frames.length - 1, Math.round((f.t + latency) / interval));
    const c = frames[captured];
    const quad = c.quad ?? decision.quad;
    return {
      captured: true,
      time: f.t,
      ji: quad ? smartDocJaccard(c.truth, quad, clip.reference) : 0,
      relSharpness: clip.bestSharpness > 0 ? c.truthSharpness / clip.bestSharpness : 0,
      speed: c.speed,
    };
  }
  return { captured: false, time: Number.NaN, ji: 0, relSharpness: 0, speed: 0 };
}

export interface GroupStats {
  clips: number;
  captured: number;
  medianTime: number;
  meanJi: number;
  bad: number;
  medianSharpness: number;
  fast: number;
}

export function aggregate(outcomes: ClipOutcome[], fastThreshold: number): GroupStats {
  const hits = outcomes.filter((o) => o.captured);
  const n = hits.length || 1;
  return {
    clips: outcomes.length,
    captured: hits.length / outcomes.length,
    medianTime: percentile(
      hits.map((o) => o.time / 1000),
      0.5,
    ),
    meanJi: hits.reduce((s, o) => s + o.ji, 0) / n,
    bad: hits.filter((o) => o.ji < 0.9).length / n,
    medianSharpness: percentile(
      hits.map((o) => o.relSharpness),
      0.5,
    ),
    fast: hits.filter((o) => o.speed > fastThreshold).length / n,
  };
}

/** Named policies; `cand:<json>` builds a {@link CandidatePolicy} from JSON options. */
export function policyFactory(name: string): () => Policy {
  if (name === 'legacy') return () => new LegacyPolicy();
  if (name === 'library') return () => new LibraryPolicy();
  if (name.startsWith('lib:')) {
    const { tracker, trigger, bestFrameMs } = JSON.parse(name.slice(4));
    return () => new LibraryPolicy(tracker, trigger, bestFrameMs);
  }
  if (name.startsWith('cand:')) {
    const options = JSON.parse(name.slice(5)) as Partial<CandidateOptions>;
    return () => new CandidatePolicy(options);
  }
  throw new Error(`Unknown policy ${name}`);
}

export function formatStats(s: GroupStats): string[] {
  const pct = (v: number) => `${(v * 100).toFixed(0)} %`;
  const any = s.captured > 0;
  return [
    String(s.clips),
    pct(s.captured),
    any ? `${s.medianTime.toFixed(1)} s` : '–',
    any ? s.meanJi.toFixed(3) : '–',
    any ? pct(s.bad) : '–',
    any ? s.medianSharpness.toFixed(2) : '–',
    any ? pct(s.fast) : '–',
  ];
}

/** Clip groups of the semi-synthetic hand replay: tuning and held-out clips of all datasets. */
const HAND_GROUPS = ['tune', 'held-out'] as const;

if (process.argv[1] && import.meta.filename === process.argv[1]) {
  const { values } = parseArgs({
    options: {
      'bench-dir': { type: 'string', default: defaultBenchDir() },
      detector: { type: 'string', default: 'nn' },
      rate: { type: 'string', default: '17' },
      latency: { type: 'string', default: '60' },
      policies: { type: 'string', default: 'legacy;library' },
      fast: { type: 'string', default: '0.25' },
      json: { type: 'string' },
      sweep: { type: 'string' },
    },
  });
  const clips = loadSimClips(values.detector as AutocaptureDetector, values['bench-dir']);
  if (values.sweep) {
    rateSweep(
      clips,
      values.sweep.split(',').map(Number),
      values.policies.split(';'),
      Number(values.latency),
    );
    process.exit(0);
  }
  const rate = Number(values.rate);
  const latency = Number(values.latency);
  const fast = Number(values.fast);
  const policies = values.policies.split(';');
  const results: Record<string, Record<string, GroupStats>> = {};
  const run = (name: string, group: string, set: SimClip[]) => {
    const make = policyFactory(name);
    const stats = aggregate(
      set.map((c) => simulate(c, make(), rate, latency)),
      fast,
    );
    results[name] ??= {};
    results[name][group] = stats;
    return stats;
  };
  console.log(
    `detector ${values.detector} (${AUTOCAPTURE_RUNS[values.detector as AutocaptureDetector].run}), ` +
      `${rate} detections/s, still latency ${latency} ms, fast motion > ${fast} diagonals/s\n`,
  );
  console.log('Real replay (first capture per clip)\n');
  console.log(
    '| policy | group | clips | captured | median time | mean JI | bad (JI < 0.9) | sharpness / best | fast motion |',
  );
  console.log('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
  for (const name of policies) {
    for (const group of GROUPS) {
      const [n, ...rest] = formatStats(
        run(
          name,
          group,
          clips.filter((c) => c.group === group),
        ),
      );
      console.log(`| ${name} | ${group} | ${n} | ${rest.join(' | ')} |`);
    }
  }
  console.log(
    '\nSemi-synthetic hand replay (real detection errors and sharpness, synthetic hand)\n',
  );
  console.log(
    '| policy | clips | steady: captured / time / sharpness | drift 0.05/s | pan 0.3/s: captured | walk: captured |',
  );
  console.log('| --- | --- | --- | --- | ---: | ---: |');
  for (const name of policies) {
    for (const split of HAND_GROUPS) {
      const set = clips.filter((c) => c.tuning === (split === 'tune'));
      const cells = HAND_SCENARIOS.map((scenario) => {
        const s = run(
          name,
          `hand ${scenario} ${split}`,
          set.map((c) => handClip(c, scenario)),
        );
        const pct = `${(s.captured * 100).toFixed(0)} %`;
        if (scenario === 'pan' || scenario === 'walk') return pct;
        return s.captured > 0
          ? `${pct} / ${s.medianTime.toFixed(1)} s / ${s.medianSharpness.toFixed(2)}`
          : pct;
      });
      console.log(`| ${name} | ${split} (${set.length}) | ${cells.join(' | ')} |`);
    }
  }
  console.log(
    `\nWhy the library policy is not ready (share of analyzed frames until the first capture, ${values.detector}, tuning clips)\n`,
  );
  const tuning = clips.filter((c) => c.tuning);
  for (const [label, set] of [
    ['real replay', tuning],
    ['steady hand', tuning.map((c) => handClip(c, 'steady'))],
  ] as const) {
    for (const key of Object.keys(LibraryPolicy.blockers)) delete LibraryPolicy.blockers[key];
    for (const c of set) simulate(c, new LibraryPolicy(), rate, latency);
    const counts = LibraryPolicy.blockers;
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const parts = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k} ${((v / total) * 100).toFixed(0)} %`);
    console.log(`- ${label}: ${parts.join(', ')}`);
  }
  if (values.json) writeFileSync(values.json, JSON.stringify(results, null, 2));
}

/**
 * Detection-rate sweep: capture share (real replay; steady, pan and walk hand replays; tuning /
 * held-out) per policy and rate, and the share of analyzed frames without a detection.
 */
function rateSweep(clips: SimClip[], rates: number[], policies: string[], latency: number): void {
  console.log(
    '| rate | policy | real: tune / held-out | steady: tune / held-out (median time) | steady, +20 % missed: tune / held-out | pan / walk | missed detections (as recorded / +20 %) |',
  );
  console.log('| ---: | --- | --- | --- | --- | --- | ---: |');
  const splits = [clips.filter((c) => c.tuning), clips.filter((c) => !c.tuning)];
  const hands = Object.fromEntries(
    HAND_SCENARIOS.map((h) => [h, splits.map((set) => set.map((c) => handClip(c, h)))]),
  ) as Record<HandScenario, SimClip[][]>;
  const lossy = hands.steady.map((set) => set.map((c) => withMisses(c, 0.2)));
  const pct = (v: number) => `${(v * 100).toFixed(0)} %`;
  for (const rate of rates) {
    const missedShare = (set: SimClip[]) => {
      let analyzed = 0;
      let missed = 0;
      for (const c of set) {
        for (const i of analyzedFrames(c, rate)) {
          analyzed++;
          if (!c.frames[i].quad) missed++;
        }
      }
      return missed / analyzed;
    };
    const missed = `${pct(missedShare(clips))} / ${pct(missedShare(lossy.flat()))}`;
    for (const name of policies) {
      const make = policyFactory(name);
      const stats = (set: SimClip[]) =>
        aggregate(
          set.map((c) => simulate(c, make(), rate, latency)),
          0.25,
        );
      const real = splits.map(stats);
      const steady = hands.steady.map(stats);
      const steadyLossy = lossy.map(stats);
      const moving = [...hands.pan, ...hands.walk].flat();
      console.log(
        `| ${rate} | ${name} | ${real.map((s) => pct(s.captured)).join(' / ')} | ${steady
          .map((s) => `${pct(s.captured)} (${s.captured ? s.medianTime.toFixed(1) : '–'} s)`)
          .join(' / ')} | ${steadyLossy
          .map((s) => `${pct(s.captured)} (${s.captured ? s.medianTime.toFixed(1) : '–'} s)`)
          .join(' / ')} | ${pct(stats(moving).captured)} | ${missed} |`,
      );
    }
  }
}
