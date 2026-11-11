/**
 * Headless-browser checks of the parts that unit tests cannot cover:
 *
 * 1. Built package: `createDetectorWorker()` with its default worker URL works when
 *    `packages/core/dist` is served as-is (unbundled ES modules, import map), and a missing
 *    worker script fails fast with a message naming the URL.
 * 2. Demo under Vite dev and 3. under a Vite production build: the library's default worker URL
 *    (`new Worker(new URL('./worker-entry.js', import.meta.url))`, resolved from the package
 *    source) loads; opening and closing the live scanner twice terminates every detector worker
 *    it created; no main-thread fallback happens.
 *
 * Uses the installed Google Chrome (`channel: 'chrome'`, override with CHROME_PATH) with a fake
 * camera. Run after `pnpm build`: `pnpm check:browser`.
 */
import { readFile } from 'node:fs/promises';
import { createServer as createHttpServer } from 'node:http';
import { createRequire } from 'node:module';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const root = fileURLToPath(new URL('..', import.meta.url));
const demo = join(root, 'apps/demo');
const require = createRequire(join(demo, 'package.json'));
const vite = await import(require.resolve('vite'));

const failures = [];
const check = (name, ok, details = '') => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${details ? ` — ${details}` : ''}`);
  if (!ok) failures.push(name);
};

const browser = await chromium.launch({
  ...(process.env.CHROME_PATH
    ? { executablePath: process.env.CHROME_PATH }
    : { channel: 'chrome' }),
  headless: true,
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
});

/**
 * Counts Worker construction/termination and worker error events inside the page, and records
 * the scanner's `scan-error` events (which carry the controller's non-fatal "Falling back to
 * main-thread detection" notice). With `silent`, detector workers never receive requests,
 * simulating a worker that loads but never answers.
 */
const WORKER_PROBE = (silent) => {
  const stats = { created: [], terminated: 0, errors: [], scanErrors: [] };
  window.__workers = stats;
  document.addEventListener('scan-error', (e) => stats.scanErrors.push(e.detail.error.message));
  const NativeWorker = window.Worker;
  window.Worker = class extends NativeWorker {
    constructor(url, options) {
      super(url, options);
      stats.created.push(String(url));
      this.addEventListener('error', (e) => stats.errors.push(e.message || 'error'));
      if (silent && String(url).includes('worker-entry')) this.postMessage = () => {};
    }
    terminate() {
      stats.terminated++;
      super.terminate();
    }
  };
};

/** In-page detection of a bright quad on a dark frame through a detector module. */
const DETECT_IN_PAGE = async (moduleUrl) => {
  const { createDetectorWorker } = await import(moduleUrl);
  const canvas = new OffscreenCanvas(384, 288);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#222';
  ctx.fillRect(0, 0, 384, 288);
  ctx.fillStyle = '#eee';
  ctx.beginPath();
  ctx.moveTo(90, 40);
  ctx.lineTo(300, 60);
  ctx.lineTo(280, 250);
  ctx.lineTo(70, 230);
  ctx.closePath();
  ctx.fill();
  const frame = ctx.getImageData(0, 0, 384, 288);
  const detector = createDetectorWorker();
  const result = { detection: null, error: null, missing: null };
  try {
    result.detection = await detector.detect(frame);
  } catch (error) {
    result.error = String(error);
  }
  detector.dispose();
  const missing = createDetectorWorker({ workerUrl: '/does-not-exist/worker-entry.js' });
  result.missing = await missing.detect(frame).then(
    () => 'resolved',
    (error) => String(error),
  );
  return result;
};

// ---- 1. built package, unbundled -------------------------------------------------------------
{
  const types = { '.js': 'text/javascript', '.html': 'text/html', '.map': 'application/json' };
  const server = createHttpServer(async (req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
    if (path === '/') {
      res.setHeader('content-type', 'text/html');
      res.end(
        '<!doctype html><script type="importmap">{"imports":{"@scanoficus/core/worker":"/packages/core/dist/worker.js"}}</script>',
      );
      return;
    }
    try {
      const body = await readFile(join(root, path));
      res.setHeader('content-type', types[extname(path)] ?? 'application/octet-stream');
      res.end(body);
    } catch {
      res.statusCode = 404;
      res.end();
    }
  });
  await new Promise((resolve) => server.listen(0, resolve));
  const page = await browser.newPage();
  await page.goto(`http://localhost:${server.address().port}/`);
  const result = await page.evaluate(DETECT_IN_PAGE, '@scanoficus/core/worker');
  check('dist: default worker URL detects a document', !!result.detection, result.error ?? '');
  check(
    'dist: missing worker fails with the URL in the message',
    /failed to load.*does-not-exist\/worker-entry\.js/.test(result.missing),
    result.missing,
  );
  await page.close();
  server.close();
}

/**
 * Opens and closes the demo's live scanner `rounds` times and returns what happened: worker
 * URLs, terminations, worker error events and the scanner's fallback notices.
 */
async function runDemo(url, { rounds = 2, silent = false } = {}) {
  const page = await browser.newPage();
  await page.addInitScript(WORKER_PROBE, silent);
  await page.goto(`${url}#/live`);
  for (let round = 0; round < rounds; round++) {
    await page.getByText('Start scanning').click();
    // A silent worker only times out after 10 s; detection then continues on the main thread.
    await page.waitForFunction(
      () => document.querySelector('scan-document')?.controller?.getSnapshot().detection,
      null,
      { timeout: 25000 },
    );
    await page.evaluate(() => {
      const camera = document
        .querySelector('scan-document')
        .shadowRoot.querySelector('scan-camera');
      camera.shadowRoot.querySelector('.cancel').click();
    });
    await page.waitForFunction(() => !document.querySelector('scan-document'));
  }
  const stats = await page.evaluate(() => window.__workers);
  await page.close();
  return {
    ...stats,
    detectorWorkers: stats.created.filter((u) => u.includes('worker-entry')),
    fallback: stats.scanErrors.filter((m) => m.startsWith('Falling back')),
  };
}

/** Checks a demo build: default worker URL, worker cleanup, no fallback; and that fallback is caught. */
async function checkDemo(label, url) {
  const result = await runDemo(url);
  check(
    `${label}: live scanner uses the default worker URL`,
    result.detectorWorkers.length === 2,
    result.created.join(', '),
  );
  check(
    `${label}: every detector worker is terminated on close`,
    result.terminated >= result.detectorWorkers.length,
    `created ${result.created.length}, terminated ${result.terminated}`,
  );
  check(
    `${label}: no main-thread fallback or worker errors`,
    result.fallback.length === 0 && result.errors.length === 0,
    [...result.scanErrors, ...result.errors].join(' | '),
  );
  // Control: a worker that loads but never answers must be reported as a fallback.
  const silent = await runDemo(url, { rounds: 1, silent: true });
  check(
    `${label}: a silent worker is detected as a main-thread fallback`,
    silent.fallback.length === 1 && /did not answer/.test(silent.fallback[0]),
    silent.scanErrors.join(' | '),
  );
}

// ---- 2. demo under Vite dev ------------------------------------------------------------------
{
  const server = await vite.createServer({
    root: demo,
    configFile: join(demo, 'vite.config.ts'),
    logLevel: 'error',
    server: { port: 0 },
  });
  await server.listen();
  const url = server.resolvedUrls.local[0];
  const page = await browser.newPage();
  await page.goto(url);
  const coreWorker = `/@fs${join(root, 'packages/core/src/worker/index.ts')}`;
  const result = await page.evaluate(DETECT_IN_PAGE, coreWorker);
  check(
    'vite dev: default worker URL (from source) detects a document',
    !!result.detection,
    result.error ?? '',
  );
  await page.close();
  await checkDemo('vite dev', url);
  await server.close();
}

// ---- 3. demo production build ----------------------------------------------------------------
{
  const outDir = join(root, 'node_modules/.cache/browser-check-demo');
  await vite.build({
    root: demo,
    configFile: join(demo, 'vite.config.ts'),
    logLevel: 'error',
    build: { outDir, emptyOutDir: true },
  });
  const server = await vite.preview({
    root: demo,
    configFile: join(demo, 'vite.config.ts'),
    logLevel: 'error',
    build: { outDir },
    preview: { port: 0 },
  });
  await checkDemo('vite build', server.resolvedUrls.local[0]);
  await new Promise((resolve) => server.httpServer.close(resolve));
}

await browser.close();
if (failures.length) {
  console.error(`\n${failures.length} browser check(s) failed`);
  process.exit(1);
}
console.log('\nall browser checks passed');
