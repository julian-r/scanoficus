/**
 * Headless-Chrome checks of the built package (`pnpm build` first), loaded as unbundled ES
 * modules: WebGL2 parity with the NumPy reference (float32 and float16 activations, network
 * only and end to end with GPU preprocessing from RGBA and ImageBitmap frames), the detector on
 * the main thread and in a worker (OffscreenCanvas), context loss and restore, and that
 * dispose() deletes every GPU object. With `--bench`, also cold-start and warm per-frame timings.
 *
 * Uses the installed Google Chrome (`channel: 'chrome'`, override with CHROME_PATH).
 * Usage: `pnpm --filter @scanoficus/detector-nn check:browser [-- --bench] [--json out.json]`.
 */
import { readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { renderScene } from '../test/scene.ts';

const pkg = fileURLToPath(new URL('..', import.meta.url));
const root = join(pkg, '../..');
const models = join(pkg, 'models');
const args = process.argv.slice(2);
const benchMode = args.includes('--bench');
const jsonOut = args.includes('--json') ? args[args.indexOf('--json') + 1] : null;
const fixtures = JSON.parse(await readFile(join(pkg, 'test/fixtures/cases.json'), 'utf8'));
const refinerFixtures = JSON.parse(await readFile(join(pkg, 'test/fixtures/refiner.json'), 'utf8'));
// round3.json / round4.json add near-threshold blends and a white-on-white page.
const decisionFixtures = await Promise.all(
  [3, 4].map(async (n) =>
    JSON.parse(await readFile(join(pkg, `test/fixtures/round${n}.json`), 'utf8')),
  ),
);
const scenes = [
  ...fixtures.cases,
  ...refinerFixtures.cases,
  ...decisionFixtures.flatMap((f) => f.cases),
];

const failures = [];
const check = (name, ok, details = '') => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${details ? ` — ${details}` : ''}`);
  if (!ok) failures.push(name);
};

// Workers do not support import maps: rewrite bare package specifiers in served modules.
const rewrite = (code) =>
  code
    .replace(/(["'])@scanoficus\/core\/([\w-]+)\1/g, '"/packages/core/dist/$2.js"')
    .replace(/(["'])@scanoficus\/core\1/g, '"/packages/core/dist/index.js"')
    .replace(/(["'])@scanoficus\/detector-nn\1/g, '"/packages/detector-nn/dist/index.js"')
    .replace(
      /(["'])@scanoficus\/detector-nn\/(docquadnet\d)\1/g,
      '"/packages/detector-nn/dist/$2.js"',
    );

const types = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json' };
const server = createServer(async (req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
  try {
    if (path === '/') {
      res.setHeader('content-type', 'text/html');
      res.end('<!doctype html><title>detector-nn check</title>');
      return;
    }
    if (path.startsWith('/frame/')) {
      const scene = scenes.find((c) => c.name === path.slice(7));
      res.end(Buffer.from(renderScene(scene).data.buffer));
      return;
    }
    const file = path.startsWith('/models/')
      ? join(models, path.slice(8))
      : path.startsWith('/fixtures/')
        ? join(pkg, 'test', path)
        : path === '/check-page.mjs'
          ? join(pkg, 'scripts/check-page.mjs')
          : join(root, path);
    const type = types[extname(file)] ?? 'application/octet-stream';
    res.setHeader('content-type', type);
    const body = await readFile(file);
    res.end(type === 'text/javascript' ? rewrite(body.toString()) : body);
  } catch {
    res.statusCode = 404;
    res.end();
  }
});
await new Promise((resolve) => server.listen(0, resolve));
const origin = `http://localhost:${server.address().port}`;

const browser = await chromium.launch({
  ...(process.env.CHROME_PATH
    ? { executablePath: process.env.CHROME_PATH }
    : { channel: 'chrome' }),
  headless: true,
  // Without a GPU (CI), Chrome only offers WebGL through SwiftShader with this flag. Add e.g.
  // SWIFTSHADER=1 forces the software renderer CI uses (ubuntu without a GPU) on a machine with
  // a GPU; CHROME_ARGS adds any other flags.
  args: [
    '--enable-unsafe-swiftshader',
    ...(process.env.SWIFTSHADER === '1' ? ['--use-angle=swiftshader'] : []),
    ...(process.env.CHROME_ARGS?.split(' ') ?? []),
  ],
});

/** Runs an export of check-page.mjs in a fresh page (fresh caches, contexts and workers). */
async function inPage(name, arg = {}) {
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(`${origin}/`);
  try {
    return await page.evaluate(
      async ([n, a]) => (await import('/check-page.mjs'))[n](a),
      [name, arg],
    );
  } finally {
    if (errors.length) console.log(`  page errors: ${errors.join(' | ')}`);
    await context.close();
  }
}

const report = { renderer: await inPage('renderer') };
console.log(`WebGL2 renderer: ${report.renderer}`);

// ---- parity -----------------------------------------------------------------------------------
report.parity = {};
for (const precision of ['float32', 'float16']) {
  const result = await inPage('parity', { precision });
  report.parity[precision] = result;
  // Corners (normalized coordinates) and the presence logit have separate tolerances.
  const part = (key, field) => Math.max(...result.cases.map((c) => c[key][field]));
  const tol =
    precision === 'float32'
      ? { network: { corners: 1e-5, logit: 1e-4 }, e2e: { corners: 1e-4, logit: 5e-3 } }
      : { network: { corners: 2e-3, logit: 1e-1 }, e2e: { corners: 2e-3, logit: 1e-1 } };
  for (const [key, label] of [
    ['networkSplit', 'network (CPU-preprocessed input)'],
    ['e2eSplit', 'end to end (GPU preprocessing)'],
  ]) {
    const t = key === 'networkSplit' ? tol.network : tol.e2e;
    check(
      `${precision}: ${label} vs NumPy`,
      part(key, 'corners') < t.corners && part(key, 'logit') < t.logit,
      `corners ${part(key, 'corners').toExponential(2)} (< ${t.corners}), logit ` +
        `${part(key, 'logit').toExponential(2)} (< ${t.logit}), presence ` +
        part(key, 'presence').toExponential(2),
    );
  }
  const worstBitmap = Math.max(...result.cases.map((c) => c.e2eBitmap));
  check(
    `${precision}: ImageBitmap frames give the RGBA results`,
    worstBitmap <= Math.max(...result.cases.map((c) => c.e2e)) + 1e-6,
    `max abs error ${worstBitmap.toExponential(2)}`,
  );
  const mismatched = (key) => result.cases.reduce((n, c) => n + c[key].mismatched, 0);
  const maxLevel = (key) => Math.max(...result.cases.map((c) => c[key].max));
  const total = result.cases.reduce((n, c) => n + c.input.total, 0);
  check(
    `${precision}: GPU preprocessing vs Python (8-bit levels)`,
    maxLevel('input') <= 1 && maxLevel('inputBitmap') <= 1,
    `RGBA: ${mismatched('input')} of ${total} values differ (max ${maxLevel('input')}); ` +
      `ImageBitmap: ${mismatched('inputBitmap')} (max ${maxLevel('inputBitmap')})`,
  );
}

// ---- synthetic graphs ---------------------------------------------------------------------------
for (const precision of ['float32', 'float16']) {
  const results = await inPage('synthetic', { precision });
  report[`synthetic ${precision}`] = results;
  const tolerance = precision === 'float32' ? 1e-4 : 1e-2;
  check(
    `${precision}: synthetic graphs (odd/non-square sizes, C % 4 ≠ 0, unfused add, hswish, sigmoid) vs NumPy`,
    results.every((r) => r.error < tolerance && r.length === r.expected),
    results.map((r) => `${r.name} ${r.error.toExponential(2)}`).join(', ') +
      ` (relative, < ${tolerance})`,
  );
}

// ---- detector API on both threads -----------------------------------------------------------
for (const [label, arg] of [
  ['main thread', { worker: false }],
  ['worker (OffscreenCanvas)', { worker: true }],
  ['main thread + refine', { worker: false, refine: true }],
]) {
  const results = await inPage('detectors', arg);
  report[`detect ${label}`] = results;
  const wrongPresence = results.filter((r) => r.detected !== r.expected);
  const worstCorner = Math.max(...results.map((r) => r.cornerError ?? 0));
  const worstScore = Math.max(...results.map((r) => r.scoreError ?? 0));
  check(
    `detect, ${label}: same decisions, quads and scores as the reference`,
    wrongPresence.length === 0 && worstCorner < (arg.refine ? 0.5 : 0.05) && worstScore < 1e-3,
    `${results.length} frames, worst corner error ${worstCorner.toFixed(4)} px, ` +
      `presence ${worstScore.toExponential(1)}`,
  );
}

// ---- round 2: corner refiner ------------------------------------------------------------------
for (const precision of ['float32', 'float16']) {
  const results = await inPage('refinerParity', { precision });
  report[`refiner ${precision}`] = results;
  const docs = results.filter((r) => r.document);
  const max = (key, list = results) => Math.max(...list.map((r) => r[key]));
  const mismatched = results.reduce((n, r) => n + r.cropMismatched, 0);
  const total = results.reduce((n, r) => n + r.crops, 0);
  const t =
    precision === 'float32'
      ? { main: 1e-4, offsets: 1e-3, refined: 0.05 }
      : { main: 2e-3, offsets: 2e-2, refined: 0.5 };
  check(
    `${precision}: round-2 main network end to end vs NumPy (documents)`,
    max('mainCorners', docs) < t.main,
    `corners ${max('mainCorners', docs).toExponential(2)} (all frames ` +
      `${max('mainCorners').toExponential(2)}), logit ${max('mainLogit').toExponential(2)}`,
  );
  check(
    `${precision}: refiner crops (GPU) vs the bit-exact CPU crops`,
    max('cropMax') <= 1 && mismatched < 0.001 * total,
    `${mismatched} of ${total} values differ by one level`,
  );
  check(
    `${precision}: refiner offsets from Python's pass quads vs NumPy (documents)`,
    max('offsets', docs) < t.offsets,
    `${max('offsets', docs).toExponential(2)} half-crop units (CPU vs GPU ` +
      `${max('offsetsVsCpu', docs).toExponential(2)})`,
  );
  check(
    `${precision}: full refinement from the raw quad vs pipeline.py (documents)`,
    max('refined', docs) < t.refined,
    `${max('refined', docs).toFixed(4)} px vs the NumPy refiner, ` +
      `${max('refinedVsOnnx', docs).toFixed(4)} px vs the ONNX pipeline`,
  );
}
// ---- round 3: 3-output refiner and the built-in decision -------------------------------------
/**
 * Refiner output tolerances vs NumPy (offsets in half-crop units; corner logit), per output:
 * - `net*`: the GPU network on the CPU's crops, which are bit-exact with Python, i.e. the
 *   network's own float32 arithmetic (measured: 1e-5 on Apple M5 / Metal and SwiftShader);
 * - end to end with the GPU's own crops: GLSL ES 3.00 has no fma, so a GPU can round some crop
 *   pixels differently from OpenCV (0 of 2 M values on Apple M5 / Metal, 22–24 on SwiftShader
 *   as on CI), and the corner logit reacts to single pixels. SwiftShader measured offsets
 *   7e-6 / 6e-5 and corner logit 1.1e-3 / 5.9e-3 (round 3 / round 4, whose refiner is more
 *   sensitive); Metal at most 2e-7 and 1.3e-5. The logit tolerance 1.5e-2 keeps 2.5× margin
 *   over round 4 on SwiftShader; the |weights| of the logit features sum to 1.33 / 0.95 in the
 *   round-3 / round-4 decision, so the score moves by < 2e-2, below its own tolerance of 0.05.
 *   Offsets may differ by 1e-3 half-crop units (0.03 px of a 64-px crop).
 * float16 activations: corner logit up to 0.057 (Metal, round 4); 0.1 keeps a margin across GPUs.
 */
const REFINER_TOLERANCE = {
  float32: { netOffsets: 1e-4, netLogit: 1e-4, offsets: 1e-3, logit: 1.5e-2 },
  float16: { netOffsets: 0.05, netLogit: 0.1, offsets: 0.05, logit: 0.1 },
};
// Scenes: the 12 round-2 scenes, 8 faint blends with NumPy margins of ±0.05 … ±0.5 around the
// threshold, and a white page on a light table.
for (const round of [3, 4]) {
  const byPrecision = {};
  for (const precision of ['float32', 'float16']) {
    const results = await inPage('decisionParity', { precision, round });
    byPrecision[precision] = results;
    report[`round${round} ${precision}`] = results;
    const docs = results.filter((r) => r.document && !r.faint);
    const max = (key, list = results) => Math.max(...list.map((r) => r[key] ?? 0));
    const mismatched = results.reduce((n, r) => n + r.cropMismatched, 0);
    const total = results.reduce((n, r) => n + r.crops, 0);
    // Score tolerance vs NumPy: float32 0.05; float16 0.3 on clear pages and 0.7 on faint
    // near-threshold pages (measured up to 0.66 before the float16 offset).
    // Score tolerance vs NumPy, clear pages / faint near-threshold pages. float32: 0.05 / 0.1
    // (faint pages: the CPU path itself differs by up to 0.04, SwiftShader measured 0.061 with its
    // GPU-rounded crops, Metal 0.042). float16: 0.3 / 0.7 (faint: up to 0.66 on Metal).
    const t =
      precision === 'float32'
        ? { score: 0.05, faint: 0.1, ...REFINER_TOLERANCE.float32 }
        : { score: 0.3, faint: 0.7, ...REFINER_TOLERANCE.float16 };
    const tolerance = (r) => (r.faint ? t.faint : t.score);
    // Decisions must agree with NumPy wherever the NumPy margin exceeds the score tolerance.
    const decided = results.filter((r) => Math.abs(r.margin) > tolerance(r));
    const wrong = decided.filter((r) => r.accept !== r.expected);
    const flips = results
      .filter((r) => r.accept !== r.expected)
      .map((r) => `${r.name} (${r.margin.toFixed(3)})`);
    check(
      `${precision}: round-${round} refiner network on the CPU's (bit-exact) crops vs NumPy`,
      max('netOffsets', docs) < t.netOffsets && max('netLogit', docs) < t.netLogit,
      `offsets ${max('netOffsets', docs).toExponential(2)} (< ${t.netOffsets}), corner logit ` +
        `${max('netLogit', docs).toExponential(2)} (< ${t.netLogit}) (clear pages)`,
    );
    check(
      `${precision}: round-${round} refiner end to end (GPU crops) from Python's pass inputs vs NumPy`,
      max('offsets', docs) < t.offsets &&
        max('logit', docs) < t.logit &&
        mismatched < 0.001 * total,
      `offsets ${max('offsets', docs).toExponential(2)} (< ${t.offsets}), corner logit ` +
        `${max('logit', docs).toExponential(2)} (< ${t.logit}) (clear pages); crops: ${mismatched} ` +
        `of ${total} values differ by one level from the CPU crops`,
    );
    check(
      `${precision}: round-${round} decision score within ${t.score} (faint pages ${t.faint}) of NumPy; decisions equal beyond it`,
      wrong.length === 0 && results.every((r) => r.score < tolerance(r)),
      `score vs NumPy ${max('score').toFixed(4)} (clear pages ${max('score', docs).toFixed(4)}, faint ` +
        `${max(
          'score',
          results.filter((r) => r.faint),
        ).toFixed(4)}), vs ONNX ` +
        `${max('scoreOnnx', docs).toFixed(4)} (clear pages); quad vs NumPy ${max('quad', docs).toFixed(4)} px ` +
        `(clear pages); ${decided.length - wrong.length}/${decided.length} decisions equal where ` +
        `|margin| exceeds the tolerance; flips within it: ${flips.join(', ') || 'none'}`,
    );
  }
  {
    // Reported, not checked: the float16 − float32 shift of the decision score depends on the GPU
    // (Apple M5 / Metal: mean −0.14; SwiftShader: about −0.02), so no constant correction fits.
    const diffs = byPrecision.float16.map((r, i) => r.value - byPrecision.float32[i].value);
    const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    const sd = Math.sqrt(diffs.reduce((a, b) => a + (b - mean) ** 2, 0) / diffs.length);
    report[`float16Bias round${round}`] = { renderer: report.renderer, mean, sd, diffs };
    console.log(
      `info round ${round}: float16 − float32 decision score on ${report.renderer}: mean ${mean.toFixed(4)}, sd ` +
        `${sd.toFixed(4)}, max |diff| ${Math.max(...diffs.map(Math.abs)).toFixed(4)} over ${diffs.length} scenes`,
    );
  }
}

for (const [label, arg] of [
  ['main thread', { worker: false }],
  ['worker', { worker: true }],
]) {
  const results = await inPage('stillDetectors', arg);
  report[`still ${label}`] = results;
  const wrong = results.filter((r) => r.detected !== r.expected);
  const worstStill = Math.max(...results.map((r) => r.still ?? 0));
  const worstLive = Math.max(...results.map((r) => r.live ?? 0));
  check(
    `round 2, ${label}: detectStill refines, detect() stays raw, threshold 0.98`,
    wrong.length === 0 && worstStill < 0.05 && worstLive < 0.05,
    `${results.length} frames; still vs pipeline.py ${worstStill.toFixed(4)} px, live vs raw ` +
      `${worstLive.toFixed(4)} px${wrong.length ? `; wrong decisions: ${wrong.map((r) => r.name)}` : ''}`,
  );
}

{
  const r = await inPage('batchSelfTest');
  check(
    "the refiner's self-test has one constant per crop (swapped batch items are caught)",
    r.passed && /self-test failed/.test(r.swapped ?? ''),
    r.swapped ?? 'swapped items were not detected',
  );
}
{
  const r = await inPage('concurrency');
  report.concurrency = r;
  const worst = Math.max(...r.errors.flatMap((e) => [e.a, e.b]));
  check(
    'concurrent detectStill/detect calls on one detector refine their own frames',
    worst < 0.05 && r.stale === 'FrameChangedError',
    `${r.errors.map((e) => `${e.round}: ${e.a.toFixed(4)} / ${e.b.toFixed(4)} px`).join(', ')}; ` +
      `stale token → ${r.stale}`,
  );
}

{
  const r = await inPage('cpuStill');
  report.cpuStill = r;
  check(
    'CPU path: a 12 MP bitmap still is read once, at 3200 px, and refined onto the page',
    r.reads.length === 1 && r.reads[0] === '3200×2400' && r.error < 2,
    `reads ${r.reads.join(', ')}; corner error ${r.error.toFixed(2)} px (640×480 scale)`,
  );
}

// ---- robustness -------------------------------------------------------------------------------
{
  const r = await inPage('contextLoss');
  report.contextLoss = r;
  check(
    'context loss: auto backend answers from the CPU while lost, WebGL2 after restore',
    r.before < 0.05 &&
      r.during < 0.05 &&
      r.backendDuring === 'cpu' &&
      r.after < 0.05 &&
      r.backendAfter === 'webgl2',
    JSON.stringify(r),
  );
  check(
    'context loss: webgl2 backend rejects while lost and recovers',
    /context lost/i.test(r.strictError ?? '') && r.strictAfter < 0.05,
    r.strictError ?? 'resolved',
  );
}
{
  const r = await inPage('selfTestFailure');
  report.selfTestFailure = r;
  check(
    'a failing WebGL2 self-test throws, leaks nothing, and auto falls back to the CPU',
    /self-test failed/.test(r.error ?? '') &&
      Object.values(r.leaked).every((n) => n === 0) &&
      r.backend === 'cpu' &&
      /self-test failed/.test(r.glError ?? '') &&
      r.detected,
    `${r.error}; objects left ${JSON.stringify(r.leaked)}; backend ${r.backend}`,
  );
}
for (const v2 of [false, true]) {
  const r = await inPage('disposeCheck', { v2 });
  report[v2 ? 'dispose v2' : 'dispose'] = r;
  check(
    `dispose deletes every texture, framebuffer and program${v2 ? ' (round 2 + refiner)' : ''}`,
    Object.values(r.leaked).every((n) => n === 0) && r.rejected,
    `live before dispose ${JSON.stringify(r.live)}, after ${JSON.stringify(r.leaked)}`,
  );
}

// ---- speed --------------------------------------------------------------------------------------
if (benchMode) {
  const configs = [
    { thread: 'main', backend: 'webgl2', precision: 'float32', input: 'rgba' },
    { thread: 'main', backend: 'webgl2', precision: 'float32', input: 'bitmap' },
    { thread: 'main', backend: 'webgl2', precision: 'float16', input: 'rgba' },
    { thread: 'main', backend: 'webgl2', precision: 'float32', input: 'rgba', refine: true },
    { thread: 'main', backend: 'webgl2', precision: 'float32', input: 'rgba384' },
    { thread: 'main', backend: 'webgl2', precision: 'float32', input: 'rgba384', refine: true },
    { thread: 'main', backend: 'cpu', input: 'rgba', runs: 20 },
    { thread: 'worker', backend: 'webgl2', precision: 'float32', input: 'bitmap' },
    { thread: 'worker', backend: 'webgl2', precision: 'float32', input: 'rgba' },
    { thread: 'worker', backend: 'webgl2', precision: 'float16', input: 'bitmap' },
    { thread: 'worker', backend: 'webgl2', precision: 'float32', input: 'rgba384' },
    { thread: 'worker', backend: 'cpu', input: 'bitmap', runs: 20 },
  ];
  configs.push(
    { thread: 'main', backend: 'webgl2', precision: 'float32', input: 'rgba', v2: true },
    {
      thread: 'main',
      backend: 'webgl2',
      precision: 'float32',
      input: 'rgba',
      v2: true,
      refinerLive: true,
      label: 'v2+ref',
    },
    {
      thread: 'main',
      backend: 'webgl2',
      precision: 'float32',
      input: 'rgba384',
      v2: true,
      refinerLive: true,
      label: 'v2+ref',
    },
    {
      thread: 'worker',
      backend: 'webgl2',
      precision: 'float32',
      input: 'rgba',
      v2: true,
      refinerLive: true,
      label: 'v2+ref',
    },
    {
      thread: 'main',
      backend: 'webgl2',
      precision: 'float32',
      input: 'still12',
      v2: true,
      still: true,
      runs: 20,
      label: 'still',
    },
    {
      thread: 'worker',
      backend: 'webgl2',
      precision: 'float32',
      input: 'still12',
      v2: true,
      still: true,
      runs: 20,
      label: 'still',
    },
    {
      thread: 'main',
      backend: 'cpu',
      input: 'still12',
      v2: true,
      still: true,
      runs: 3,
      label: 'still',
    },
  );
  // Round 4 (the demo default): the decision needs the refiner on every live frame.
  for (const thread of ['main', 'worker']) {
    for (const input of ['rgba384', 'rgba768', 'rgba']) {
      configs.push({
        thread,
        backend: 'webgl2',
        precision: 'float32',
        input,
        v4: true,
        label: 'v4',
      });
    }
  }
  report.bench = [];
  console.log(
    '\nthread  backend precision input   model  | cold total (load / init / 1st frame) | warm p50 / p95 ms',
  );
  for (const config of configs) {
    const arg = { precision: 'auto', refine: false, runs: 100, warmup: 10, ...config };
    if (arg.backend === 'cpu') arg.warmup = 3;
    const r = await inPage('bench', arg);
    report.bench.push({ ...arg, ...r });
    const c = r.cold;
    const parts = [c.loadModel, c.init, c.firstFrame].map((v) =>
      v === undefined ? '–' : v.toFixed(1),
    );
    console.log(
      `${arg.thread.padEnd(7)} ${r.backend.padEnd(7)} ${(arg.backend === 'cpu' ? '–' : arg.precision).padEnd(9)} ` +
        `${arg.input.padEnd(7)} ${(arg.label ?? (arg.v2 ? 'v2' : arg.refine ? 'edge' : 'v1')).padEnd(6)} | ${c.total.toFixed(1)} (${parts.join(' / ')}) | ` +
        `${r.p50.toFixed(2)} / ${r.p95.toFixed(2)}`,
    );
  }
}

await browser.close();
server.close();
if (jsonOut) await writeFile(jsonOut, `${JSON.stringify(report, null, 1)}\n`);
if (failures.length) {
  console.error(`\n${failures.length} browser check(s) failed`);
  process.exit(1);
}
console.log('\nall detector-nn browser checks passed');
