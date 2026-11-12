/**
 * Headless-browser checks of the parts that unit tests cannot cover:
 *
 * 1. Built package: `createDetectorWorker()` with its default worker URL works when
 *    `packages/core/dist` is served as-is (unbundled ES modules, import map), and a missing
 *    worker script fails fast with a message naming the URL.
 * 2. Demo under Vite dev and 3. under a Vite production build: the library's default worker URL
 *    (`new Worker(new URL('./worker-entry.js', import.meta.url))`, resolved from the package
 *    source) loads; opening and closing the live scanner twice terminates every detector worker
 *    it created; no main-thread fallback happens. With each detector choice of the demo
 *    (Classic, Neural network, Neural + refine, Hybrid), the live scanner and the photo import
 *    find the synthetic document; the networks run in their worker on WebGL2, without fallback.
 *    The production build is also checked under the GitHub Pages base path `/scanoficus/`.
 *
 * Uses the installed Google Chrome (`channel: 'chrome'`, override with CHROME_PATH) with a fake
 * camera that plays a synthetic page (the `page-640x480` scene of the detector-nn fixtures).
 * Run after `pnpm build`: `pnpm check:browser`.
 */
import { mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createServer as createHttpServer } from 'node:http';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { renderScene } from '../packages/detector-nn/test/scene.ts';

const root = fileURLToPath(new URL('..', import.meta.url));
const demo = join(root, 'apps/demo');
const require = createRequire(join(demo, 'package.json'));
const vite = await import(require.resolve('vite'));

const failures = [];
const check = (name, ok, details = '') => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${details ? ` — ${details}` : ''}`);
  if (!ok) failures.push(name);
};

// The synthetic document: a page with text lines on a noisy desk (640×480).
const SCENE = JSON.parse(
  await readFile(join(root, 'packages/detector-nn/test/fixtures/cases.json'), 'utf8'),
).cases.find((c) => c.name === 'page-640x480');
const sceneRgba = renderScene(SCENE);
const tmp = mkdtempSync(join(tmpdir(), 'scanoficus-browser-check-'));
const videoFile = join(tmp, 'page.y4m');
writeFileSync(videoFile, toY4m(sceneRgba, 2));
const photoBmp = toBmp(sceneRgba);

/** Y4M (4:2:0, full-range BT.601) video of `frames` copies of an RGBA image. */
function toY4m({ width, height, data }, frames) {
  const y = Buffer.alloc(width * height);
  const u = Buffer.alloc((width / 2) * (height / 2));
  const v = Buffer.alloc(u.length);
  for (let i = 0; i < width * height; i++) {
    y[i] = Math.round(0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2]);
  }
  for (let j = 0; j < height / 2; j++) {
    for (let i = 0; i < width / 2; i++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (const [dx, dy] of [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ]) {
        const p = 4 * ((2 * j + dy) * width + 2 * i + dx);
        r += data[p] / 4;
        g += data[p + 1] / 4;
        b += data[p + 2] / 4;
      }
      u[j * (width / 2) + i] = Math.round(128 - 0.168736 * r - 0.331264 * g + 0.5 * b);
      v[j * (width / 2) + i] = Math.round(128 + 0.5 * r - 0.418688 * g - 0.081312 * b);
    }
  }
  const frame = Buffer.concat([Buffer.from('FRAME\n'), y, u, v]);
  return Buffer.concat([
    Buffer.from(`YUV4MPEG2 W${width} H${height} F30:1 Ip A1:1 C420jpeg\n`),
    ...Array(frames).fill(frame),
  ]);
}

/** Uncompressed 24-bit BMP of an RGBA image (rows are 4-byte aligned for widths ÷ 4). */
function toBmp({ width, height, data }) {
  const row = width * 3;
  const out = Buffer.alloc(54 + row * height);
  out.write('BM', 0);
  out.writeUInt32LE(out.length, 2);
  out.writeUInt32LE(54, 10);
  out.writeUInt32LE(40, 14);
  out.writeInt32LE(width, 18);
  out.writeInt32LE(height, 22);
  out.writeUInt16LE(1, 26);
  out.writeUInt16LE(24, 28);
  out.writeUInt32LE(row * height, 34);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = 4 * ((height - 1 - y) * width + x);
      const o = 54 + y * row + 3 * x;
      out[o] = data[p + 2];
      out[o + 1] = data[p + 1];
      out[o + 2] = data[p];
    }
  }
  return out;
}

const browser = await chromium.launch({
  ...(process.env.CHROME_PATH
    ? { executablePath: process.env.CHROME_PATH }
    : { channel: 'chrome' }),
  headless: true,
  args: [
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    // Without a GPU (CI), WebGL (the demo's neural detectors) needs SwiftShader.
    '--enable-unsafe-swiftshader',
    `--use-file-for-fake-video-capture=${videoFile}`,
    // SWIFTSHADER=1 forces the software renderer CI uses (ubuntu without a GPU).
    ...(process.env.SWIFTSHADER === '1' ? ['--use-angle=swiftshader'] : []),
  ],
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
  // These checks are about the core's detector worker: use the Classic detector choice.
  await page.addInitScript(() => localStorage.setItem('scanoficus-detector', 'classic'));
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

const CHOICES = ['classic', 'nn', 'nn-refine', 'hybrid', 'nn2-refiner', 'nn3', 'nn4'];

/**
 * Without a stored choice, the demo defaults to Neural v4 (round 4 with float32 activations on
 * 768-px frames) where WebGL2 renders to float32 textures (headless Chrome does), else to
 * Classic; a stored choice is kept.
 */
async function checkDefaultChoice(label, url) {
  const page = await browser.newPage();
  await page.addInitScript(() => {
    localStorage.setItem('scanoficus-debug', '1');
  });
  await page.goto(`${url}#/live`);
  const selected = await page.evaluate(() => document.getElementById('detector').value);
  await page.getByText('Start scanning').click();
  const status = await page
    .waitForFunction(
      () =>
        document.querySelector('scan-document')?.controller?.getSnapshot().detection?.quad &&
        /worker ·/.test(window.scanoficusDetector?.running ?? '') &&
        window.scanoficusDetector,
      null,
      { timeout: 40000 },
    )
    .then((h) => h.jsonValue())
    .catch((e) => ({ failed: String(e).split('\n')[0] }));
  const maxSide = await page.evaluate(
    () => document.querySelector('scan-document')?.options?.detection?.maxSide,
  );
  await page.close();
  check(
    `${label}: the default detector is Neural v4 (float32, 768-px frames)`,
    selected === 'nn4' &&
      status.choice === 'nn4' &&
      /Neural v4 · worker · WebGL2 float32/.test(status.running) &&
      maxSide === 768 &&
      !status.error,
    `${selected}; ${status.failed ?? status.running}; detection maxSide ${maxSide}`,
  );
  // Without float32 render targets (float16-only GPUs) the default is Classic.
  const half = await browser.newPage();
  await half.addInitScript(() => {
    const get = WebGL2RenderingContext.prototype.getExtension;
    WebGL2RenderingContext.prototype.getExtension = function (name) {
      return name === 'EXT_color_buffer_float' ? null : get.call(this, name);
    };
  });
  await half.goto(`${url}#/live`);
  const halfChoice = await half.evaluate(() => document.getElementById('detector').value);
  // A stored Neural v4 on such a GPU falls back to Classic (it needs float32). The override
  // above does not reach workers, so the network runs on the main thread here.
  await half.evaluate(() => localStorage.setItem('scanoficus-detector', 'nn4'));
  await half.addInitScript(() => {
    globalThis.OffscreenCanvas = undefined;
  });
  await half.reload();
  await half.getByText('Start scanning').click();
  const halfStatus = await half
    .waitForFunction(() => window.scanoficusDetector?.error && window.scanoficusDetector, null, {
      timeout: 40000,
    })
    .then((h) => h.jsonValue())
    .catch((e) => ({ failed: String(e).split('\n')[0] }));
  await half.close();
  check(
    `${label}: without float32 render targets the default is Classic, and Neural v4 falls back`,
    halfChoice === 'classic' &&
      /Classic · worker \(fallback\)/.test(halfStatus.running ?? '') &&
      /float32 textures \(needs EXT_color_buffer_float\)/.test(halfStatus.error ?? ''),
    `${halfChoice}; ${halfStatus.failed ?? `${halfStatus.running}: ${halfStatus.error}`}`,
  );
  // A stored choice wins over the default.
  const stored = await browser.newPage();
  await stored.addInitScript(() => localStorage.setItem('scanoficus-detector', 'nn'));
  await stored.goto(`${url}#/live`);
  const kept = await stored.evaluate(() => document.getElementById('detector').value);
  await stored.close();
  check(`${label}: a stored detector choice is kept`, kept === 'nn', kept);
}

/** Detection rate from which Neural v4 must auto-capture (override: NN_AUTOCAPTURE_MIN_RATE). */
const NN_AUTOCAPTURE_MIN_RATE = Number(process.env.NN_AUTOCAPTURE_MIN_RATE ?? 6);

/** Auto-capture lines of the demo's debug overlay, joined for a log line. */
const overlayLines = (panel) =>
  panel
    .split('\n')
    .filter((l) => /auto-capture|rate|frames|speed|jitter|sharpness|time/.test(l))
    .map((l) => l.replace(/ +/g, ' ').trim())
    .join(' | ');

/** Waits until the controller has measured its detection rate; returns it and the overlay. */
async function measureRate(page) {
  return page
    .waitForFunction(
      () => {
        const d = document.querySelector('scan-document')?.controller?.getSnapshot().detection;
        const rate = d?.autoCapture?.detectionRate;
        if (rate == null) return null;
        return {
          rate,
          min: d.autoCapture.minDetectionRate,
          panel: document.querySelector('.debug-panel pre')?.textContent ?? '',
        };
      },
      null,
      { timeout: 15000, polling: 200 },
    )
    .then((h) => h.jsonValue())
    .catch(async () => ({
      rate: null,
      min: null,
      panel: await page.evaluate(
        () => document.querySelector('.debug-panel pre')?.textContent ?? '',
      ),
    }));
}

/**
 * The fake camera shows a still page, so auto-capture must fire (with the best-frame buffer
 * grabbing full-resolution video frames). Required with Classic (fast everywhere); with Neural v4
 * only if this machine detects at ≥ 6/s (CI without a GPU runs it on SwiftShader at a few
 * frames per second), else skipped with a note. Classic then exports the page as a PDF (the
 * controller's lazy chunk).
 */
async function checkAutoCapture(label, choice, page, errors) {
  const measured = await measureRate(page);
  const rate = measured.rate === null ? 'not measured' : `${measured.rate.toFixed(1)} detections/s`;
  if (choice !== 'classic' && !(measured.rate >= NN_AUTOCAPTURE_MIN_RATE)) {
    console.log(
      `skip ${label}: auto-capture with "${choice}" not required: ${rate} on this machine ` +
        `(< ${NN_AUTOCAPTURE_MIN_RATE}/s, e.g. SwiftShader without a GPU) — ${overlayLines(measured.panel)}`,
    );
    if (measured.rate !== null && measured.rate < measured.min) {
      check(
        `${label}: a too slow detector says so in the overlay ("${choice}")`,
        /detection rate too low for auto-capture: [\d.]+\/s/.test(measured.panel),
        overlayLines(measured.panel),
      );
    }
    return;
  }
  const started = Date.now();
  const captured = await page
    .waitForFunction(
      () => (document.querySelector('scan-document')?.controller?.pages.length ?? 0) > 0,
      null,
      { timeout: 15000, polling: 100 },
    )
    .then(() => true)
    .catch(() => false);
  const panel = await page.evaluate(
    () => document.querySelector('.debug-panel pre')?.textContent ?? '',
  );
  check(
    `${label}: auto-capture fires on a steady page ("${choice}")`,
    captured && errors.length === 0,
    `${captured ? `captured after ≤ ${Date.now() - started} ms more` : 'no capture'} at ${rate}; ${
      errors.join('; ') || overlayLines(captured ? measured.panel : panel)
    }`,
  );
  if (choice !== 'classic') return;
  const pdf = captured
    ? await page
        .evaluate(async () => {
          const blob = await document.querySelector('scan-document').controller.exportPdf();
          const head = new TextDecoder().decode(await blob.slice(0, 5).arrayBuffer());
          return { size: blob.size, head };
        })
        .catch((e) => ({ failed: String(e) }))
    : { failed: 'no page (auto-capture did not fire)' };
  check(
    `${label}: PDF export loads its lazy chunk ("${choice}")`,
    pdf.head === '%PDF-' && pdf.size > 1000,
    pdf.failed ?? `${pdf.size} bytes`,
  );
}

/**
 * A slow device (Classic slowed down to about 2.5 detections/s through the demo's delay hook):
 * auto-capture cannot fill its stability window, and the overlay must say why.
 */
async function checkSlowDetector(label, url) {
  const page = await browser.newPage();
  await page.addInitScript(() => {
    localStorage.setItem('scanoficus-detector', 'classic');
    localStorage.setItem('scanoficus-debug', '1');
    window.__scanoficusDemoDelayMs = 400;
  });
  await page.goto(`${url}#/live`);
  await page.getByText('Start scanning').click();
  const measured = await measureRate(page);
  const shown = await page
    .waitForFunction(
      () =>
        /detection rate too low for auto-capture: [\d.]+\/s/.test(
          document.querySelector('.debug-panel pre')?.textContent ?? '',
        ),
      null,
      { timeout: 10000, polling: 200 },
    )
    .then(() => true)
    .catch(() => false);
  const panel = await page.evaluate(
    () => document.querySelector('.debug-panel pre')?.textContent ?? '',
  );
  const pages = await page.evaluate(
    () => document.querySelector('scan-document')?.controller?.pages.length ?? -1,
  );
  check(
    `${label}: a too slow detector shows "detection rate too low" and does not capture`,
    shown && pages === 0 && measured.rate !== null && measured.rate < 4,
    `${measured.rate?.toFixed(1)} detections/s, pages ${pages}; ${overlayLines(panel)}`,
  );
  await page.close();
}

/** Live scanner and photo import with every detector choice find the synthetic page. */
async function checkDetectorChoices(label, url) {
  for (const choice of CHOICES) {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.addInitScript((c) => {
      localStorage.setItem('scanoficus-detector', c);
      localStorage.setItem('scanoficus-debug', '1');
      // Record every status the demo publishes.
      let value;
      window.__statusLog = [];
      Object.defineProperty(window, 'scanoficusDetector', {
        get: () => value,
        set: (v) => {
          value = v;
          window.__statusLog.push(v.running);
        },
      });
    }, choice);
    await page.goto(`${url}#/live`);
    await page.getByText('Start scanning').click();
    const live = await page
      .waitForFunction(
        ([page, choice]) => {
          const d = document.querySelector('scan-document')?.controller?.getSnapshot().detection;
          if (!d?.quad) return null;
          // Largest corner distance to the page (640×480 scene) as a fraction of the diagonal.
          const scale = d.frameWidth / 640;
          const error =
            Math.max(
              ...d.quad.map((p) =>
                Math.min(...page.map(([x, y]) => Math.hypot(p.x / scale - x, p.y / scale - y))),
              ),
            ) / 800;
          if (error > 0.02) return null;
          const panel = document.querySelector('.debug-panel pre')?.textContent ?? '';
          const status = window.scanoficusDetector;
          // Wait for the chosen detector itself (Hybrid answers with the classic worker until
          // the network is ready) and for the overlay to show it.
          if (choice !== 'classic' && !status?.error && !/· worker ·/.test(status?.running ?? '')) {
            return null;
          }
          if (!panel.includes(status?.running ?? '\u0000')) return null;
          return { error, status, panel };
        },
        [SCENE.page, choice],
        { timeout: 40000, polling: 100 },
      )
      .then((h) => h.jsonValue())
      .catch(async (e) => ({
        failed: String(e).split('\n')[0],
        status: await page.evaluate(() => window.scanoficusDetector),
      }));
    const status = live.status ?? {};
    const network = choice !== 'classic';
    const runsAsChosen =
      !status.error &&
      (!network ||
        (/· worker ·/.test(status.running) && /WebGL2 float(32|16)/.test(status.running)));
    check(
      `${label}: live scan with "${choice}" finds the page`,
      !live.failed && runsAsChosen,
      live.failed
        ? `${live.failed}; ${JSON.stringify(status)}`
        : `corner error ${(live.error * 100).toFixed(2)} % of the diagonal; ${status.running}` +
            (network ? `, cold start ${Math.round(status.coldStartMs)} ms` : ''),
    );
    if (!live.failed && choice === 'hybrid') {
      const log = await page.evaluate(() => window.__statusLog);
      check(
        `${label}: Hybrid answers with Classic until round 1 + refine is ready`,
        log.some((r) => /classic until the network is ready/.test(r)) &&
          /round 1 \+ refine/.test(status.running) &&
          (await page.evaluate(
            () => document.querySelector('scan-document')?.options?.detection?.maxSide,
          )) === 768,
        log.join(' → '),
      );
    }
    if (!live.failed && choice === 'nn') {
      // A context loss (simulated through the demo's fault hook) skips frames but keeps the
      // network; any other failure falls back to Classic with the reason.
      const after = async (fault, until) => {
        await page.evaluate((f) => {
          window.__scanoficusDemoFault = f;
        }, fault);
        return page
          .waitForFunction(until, null, { timeout: 15000 })
          .then(() => page.evaluate(() => window.scanoficusDetector))
          .catch(async () => ({
            failed: true,
            ...(await page.evaluate(() => window.scanoficusDetector)),
          }));
      };
      const lost = await after('WebGL context lost', () =>
        /context lost/.test(window.scanoficusDetector?.notice ?? ''),
      );
      check(
        `${label}: a context loss keeps the network ("${choice}")`,
        !lost.failed && !lost.error && /Neural network · worker/.test(lost.running),
        `${lost.running}; ${lost.notice}`,
      );
      const broken = await after('shader exploded', () =>
        /shader exploded/.test(window.scanoficusDetector?.error ?? ''),
      );
      check(
        `${label}: another network failure falls back to Classic ("${choice}")`,
        !broken.failed && /Classic/.test(broken.running),
        `${broken.running}: ${broken.error}`,
      );
    }
    if (!live.failed) {
      check(
        `${label}: debug overlay shows the detector ("${choice}")`,
        live.panel.includes(status.running) &&
          /cold start/.test(live.panel) &&
          /detect p50/.test(live.panel) &&
          // Decision bundles name their decision file (shipped decision.json or a swapped one).
          (choice === 'nn4'
            ? /decision +export\/docquadnet4\/decision\.json/.test(live.panel)
            : choice === 'nn3'
              ? /decision +export\/docquadnet3\/decision\.json/.test(live.panel)
              : !/decision /.test(live.panel)),
        live.panel
          .split('\n')
          .filter((l) => /detector|cold|p50|fallback|decision/.test(l))
          .join(' | '),
      );
      check(
        `${label}: debug overlay explains auto-capture ("${choice}")`,
        /auto-capture (READY|waiting: )/.test(live.panel) &&
          /[✓✗·] speed .*diag\/s/.test(live.panel) &&
          /[✓✗·] jitter /.test(live.panel) &&
          /[✓✗] sharpness .*of ref/.test(live.panel) &&
          /[✓✗] time .* ms ready/.test(live.panel),
        live.panel
          .split('\n')
          .filter((l) => /auto-capture|speed|jitter|frames|sharpness|gyro|time/.test(l))
          .join(' | '),
      );
      if (choice === 'classic' || choice === 'nn4')
        await checkAutoCapture(label, choice, page, errors);
    }

    // Photo import: the detected quad is shown in the crop editor.
    await page.goto(`${url}#/photo`);
    await page.setInputFiles('input[type=file]', {
      name: 'page.bmp',
      mimeType: 'image/bmp',
      buffer: photoBmp,
    });
    const photo = await page
      .waitForFunction(
        (page) => {
          const editor = document.querySelector('scan-crop-editor');
          if (!editor?.quad) return null;
          const error =
            Math.max(
              ...editor.quad.map((p) =>
                Math.min(...page.map(([x, y]) => Math.hypot(p.x - x, p.y - y))),
              ),
            ) / 800;
          return { error, status: window.scanoficusDetector };
        },
        SCENE.page,
        { timeout: 40000 },
      )
      .then((h) => h.jsonValue())
      .catch((e) => ({ failed: String(e).split('\n')[0] }));
    // Rounds 2–4 re-detect the full-resolution still with the corner refiner (detectStill).
    const usedStill = photo.status?.stillSize === '640×480';
    check(
      `${label}: photo import with "${choice}" finds the page`,
      !photo.failed &&
        photo.error < 0.02 &&
        usedStill === (choice === 'nn2-refiner' || choice === 'nn3' || choice === 'nn4'),
      photo.failed ??
        `corner error ${(photo.error * 100).toFixed(2)} % of the diagonal` +
          (usedStill
            ? `; detectStill on the 640×480 still in ${Math.round(photo.status.stillMs)} ms`
            : ''),
    );
    if (errors.length) check(`${label}: no page errors ("${choice}")`, false, errors.join(' | '));
    await page.close();
  }
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
  await checkDefaultChoice('vite dev', url);
  await checkDetectorChoices('vite dev', url);
  await checkSlowDetector('vite dev', url);
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

// ---- 4. demo production build under the GitHub Pages base path --------------------------------
{
  const outDir = join(root, 'node_modules/.cache/browser-check-demo-pages');
  const options = {
    root: demo,
    configFile: join(demo, 'vite.config.ts'),
    logLevel: 'error',
    base: '/scanoficus/',
  };
  await vite.build({ ...options, build: { outDir, emptyOutDir: true } });
  const server = await vite.preview({ ...options, build: { outDir }, preview: { port: 0 } });
  const url = server.resolvedUrls.local[0];
  // The model files are emitted with content hashes (safe for the cache-first service worker).
  const assets = readdirSync(join(outDir, 'assets'));
  const hashed = (re) => assets.filter((f) => re.test(f));
  check(
    'vite build /scanoficus/: model graphs, weights, licenses and credits are content-hashed assets',
    // Rounds 1 and 4 from the package, rounds 2 and 3 from apps/demo/models; two LICENSE.md
    // (package and demo); the coco-credits.csv of rounds 3 and 4 are equal (one asset).
    ['docquadnet', 'docquadnet2', 'docquadnet4'].every(
      (m) =>
        hashed(new RegExp(`^${m}-[\\w-]+\\.json$`)).length >= 1 &&
        hashed(new RegExp(`^${m}-[0-9a-f]{12}-[\\w-]+\\.bin$`)).length >= 1,
    ) &&
      hashed(/^docquadnet3-refiner-[0-9a-f]{12}-[\w-]+\.bin$/).length === 1 &&
      hashed(/^LICENSE-[\w-]+\.md$/).length === 2 &&
      hashed(/^coco-credits-[\w-]+\.csv$/).length === 1,
    [...hashed(/^docquadnet/), ...hashed(/^LICENSE/), ...hashed(/^coco-credits/)].join(', '),
  );
  await checkDetectorChoices('vite build /scanoficus/', url);
  await checkSlowDetector('vite build /scanoficus/', url);
  // Fallback: without the weights, the network choice falls back to Classic and says why.
  {
    const page = await browser.newPage();
    await page.route('**/docquadnet-*.bin', (route) => route.fulfill({ status: 404, body: '' }));
    await page.addInitScript(() => localStorage.setItem('scanoficus-detector', 'nn-refine'));
    await page.goto(`${url}#/live`);
    await page.getByText('Start scanning').click();
    const status = await page
      .waitForFunction(
        () =>
          document.querySelector('scan-document')?.controller?.getSnapshot().detection?.quad &&
          window.scanoficusDetector,
        null,
        { timeout: 40000 },
      )
      .then((h) => h.jsonValue())
      .catch((e) => ({ failed: String(e).split('\n')[0] }));
    check(
      'vite build /scanoficus/: missing weights fall back to Classic with the reason',
      /Classic/.test(status.running ?? '') && /404/.test(status.error ?? ''),
      status.failed ?? `${status.running}: ${status.error}`,
    );
    await page.close();
  }
  await new Promise((resolve) => server.httpServer.close(resolve));
}

await browser.close();
rmSync(tmp, { recursive: true, force: true });
if (failures.length) {
  console.error(`\n${failures.length} browser check(s) failed`);
  process.exit(1);
}
console.log('\nall browser checks passed');
