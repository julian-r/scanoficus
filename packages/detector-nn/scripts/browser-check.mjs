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
const models = join(root, 'apps/demo/public/models/docquadnet');
const args = process.argv.slice(2);
const benchMode = args.includes('--bench');
const jsonOut = args.includes('--json') ? args[args.indexOf('--json') + 1] : null;
const fixtures = JSON.parse(await readFile(join(pkg, 'test/fixtures/cases.json'), 'utf8'));

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
    .replace(/(["'])@scanoficus\/detector-nn\1/g, '"/packages/detector-nn/dist/index.js"');

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
      const scene = fixtures.cases.find((c) => c.name === path.slice(7));
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
  const worst = (key) => Math.max(...result.cases.map((c) => c[key]));
  const tolerance = precision === 'float32' ? 1e-3 : 2e-2;
  check(
    `${precision}: network (CPU-preprocessed input) vs NumPy`,
    worst('network') < tolerance,
    `max abs error ${worst('network').toExponential(2)}`,
  );
  check(
    `${precision}: end to end (GPU preprocessing, RGBA / ImageBitmap) vs NumPy`,
    worst('e2e') < 10 * tolerance && worst('e2eBitmap') < 10 * tolerance,
    `max abs error ${worst('e2e').toExponential(2)} / ${worst('e2eBitmap').toExponential(2)}`,
  );
  const part = (key, field) => Math.max(...result.cases.map((c) => c[key][field]));
  console.log(
    `     corners / presence probability / logit: network ${['corners', 'presence', 'logit']
      .map((f) => part('networkSplit', f).toExponential(2))
      .join(' / ')}; end to end ${['corners', 'presence', 'logit']
      .map((f) => part('e2eSplit', f).toExponential(2))
      .join(' / ')}`,
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

// ---- robustness -------------------------------------------------------------------------------
{
  const r = await inPage('contextLoss');
  report.contextLoss = r;
  check(
    'context loss: auto backend answers from the CPU while lost, WebGL2 after restore',
    r.before < 0.05 && r.during < 0.05 && r.after < 0.05 && r.backendAfter === 'webgl2',
    JSON.stringify(r),
  );
  check(
    'context loss: webgl2 backend rejects while lost and recovers',
    /context lost/i.test(r.strictError ?? '') && r.strictAfter < 0.05,
    r.strictError ?? 'resolved',
  );
}
{
  const r = await inPage('disposeCheck');
  report.dispose = r;
  check(
    'dispose deletes every texture, framebuffer and program',
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
  report.bench = [];
  console.log(
    '\nthread  backend precision input   refine | cold total (load / init / 1st frame) | warm p50 / p95 ms',
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
        `${arg.input.padEnd(7)} ${String(!!arg.refine).padEnd(6)} | ${c.total.toFixed(1)} (${parts.join(' / ')}) | ` +
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
