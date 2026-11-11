// In-page half of scripts/browser-check.mjs: parity, robustness and timing checks of the
// built package, loaded unbundled. Every export takes plain JSON arguments and returns JSON.
import {
  bytesToInput,
  createNnDetector,
  createNnDetectorWorker,
  GlRunner,
  loadModel,
  preprocessRgba,
} from '@scanoficus/detector-nn';

const MODEL = { graphUrl: '/models/docquadnet.json', weightsUrl: '/models/docquadnet.bin' };

async function fixtures() {
  return (await fetch('/fixtures/cases.json')).json();
}

async function frame(c) {
  const bytes = new Uint8ClampedArray(await (await fetch(`/frame/${c.name}`)).arrayBuffer());
  return new ImageData(bytes, c.width, c.height);
}

const maxAbs = (a, b) => Math.max(...Array.from(b, (v, i) => Math.abs(v - a[i])));

function cornerError(quad, expected) {
  let best = Number.POSITIVE_INFINITY;
  for (let shift = 0; shift < 4; shift++) {
    let worst = 0;
    for (let i = 0; i < 4; i++) {
      const p = quad[(i + shift) % 4];
      worst = Math.max(worst, Math.hypot(p.x - expected[i][0], p.y - expected[i][1]));
    }
    best = Math.min(best, worst);
  }
  return best;
}

function compareInput(gpu, cpu) {
  let mismatched = 0;
  let max = 0;
  for (let i = 0; i < cpu.length; i++) {
    const d = Math.abs(gpu[i] - cpu[i]);
    if (d) mismatched++;
    max = Math.max(max, d);
  }
  return { mismatched, max, total: cpu.length };
}

/** Renderer string of the WebGL2 implementation used. */
export function renderer() {
  const gl = new OffscreenCanvas(1, 1).getContext('webgl2');
  const info = gl.getExtension('WEBGL_debug_renderer_info');
  const name = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  gl.getExtension('WEBGL_lose_context')?.loseContext();
  return name;
}

/** GlRunner vs the NumPy reference: network only, and end to end with GPU preprocessing. */
export async function parity({ precision }) {
  const split = (heads, c) => ({
    corners: maxAbs(heads.subarray(0, 8), c.heads.slice(0, 8)),
    presence: Math.abs(1 / (1 + Math.exp(-heads[8])) - c.presence),
    logit: Math.abs(heads[8] - c.heads[8]),
  });
  const meta = await fixtures();
  const model = await loadModel(MODEL);
  const runner = new GlRunner(model, { precision });
  const cases = [];
  for (const c of meta.cases) {
    const image = await frame(c);
    const bytes = preprocessRgba(image, model.size);
    const networkHeads = runner.runTensor(bytesToInput(bytes));
    const network = maxAbs(networkHeads, c.heads);
    const networkSplit = split(networkHeads, c);
    const e2eHeads = runner.run(image);
    const e2e = maxAbs(e2eHeads, c.heads);
    const e2eSplit = split(e2eHeads, c);
    const input = compareInput(runner.readInput(), bytes);
    const bitmap = await createImageBitmap(image);
    const e2eBitmap = maxAbs(runner.run(bitmap), c.heads);
    const inputBitmap = compareInput(runner.readInput(), bytes);
    bitmap.close();
    cases.push({
      name: c.name,
      network,
      networkSplit,
      e2e,
      e2eSplit,
      e2eBitmap,
      input,
      inputBitmap,
    });
  }
  runner.dispose();
  return { precision: runner.precision, cases };
}

/** The detector API on this thread or in a worker, against the fixture quads. */
export async function detectors({ worker, refine }) {
  const meta = await fixtures();
  const detector = worker
    ? createNnDetectorWorker({ ...MODEL, backend: 'webgl2', refine })
    : createNnDetector({ ...MODEL, backend: 'webgl2', refine });
  const results = [];
  for (const c of meta.cases) {
    const image = await frame(c);
    // Bitmaps are consumed (transferred to the worker); RGBA frames are copied.
    for (const kind of ['rgba', 'bitmap']) {
      const input = kind === 'rgba' ? image : await createImageBitmap(image);
      const detection = await detector.detect(input);
      const expected = refine ? c.refined : c.quad;
      results.push({
        name: c.name,
        kind,
        detected: !!detection,
        expected: c.detected,
        cornerError: detection ? cornerError(detection.quad, expected) : null,
        scoreError: detection ? Math.abs(detection.score - c.presence) : null,
      });
    }
  }
  detector.dispose();
  return results;
}

/** Context loss: the 'auto' backend answers from the CPU while lost and recovers on restore. */
export async function contextLoss() {
  const meta = await fixtures();
  const c = meta.cases[0];
  const image = await frame(c);
  const canvas = new OffscreenCanvas(1, 1);
  const detector = createNnDetector({ ...MODEL, canvas });
  const strictCanvas = new OffscreenCanvas(1, 1);
  const strict = createNnDetector({ ...MODEL, canvas: strictCanvas, backend: 'webgl2' });
  const before = await detector.detect(image);
  await strict.detect(image);
  const lose = (cv) => cv.getContext('webgl2').getExtension('WEBGL_lose_context');
  const ext = lose(canvas);
  const strictExt = lose(strictCanvas);
  const event = (cv, type) =>
    new Promise((resolve, reject) => {
      cv.addEventListener(type, resolve, { once: true });
      setTimeout(() => reject(new Error(`no ${type} within 5 s`)), 5000);
    });
  const lost = Promise.all([
    event(canvas, 'webglcontextlost'),
    event(strictCanvas, 'webglcontextlost'),
  ]);
  ext.loseContext();
  strictExt.loseContext();
  const during = await detector.detect(image);
  const backendDuring = detector.backend;
  const strictError = await strict.detect(image).then(
    () => null,
    (e) => String(e),
  );
  // The browser restores only after the (prevented) lost event was dispatched; leave the
  // dispatch (our listener resumes inside it) before restoring.
  await lost;
  await new Promise((r) => setTimeout(r, 0));
  const both = Promise.all([
    event(canvas, 'webglcontextrestored'),
    event(strictCanvas, 'webglcontextrestored'),
  ]);
  ext.restoreContext();
  strictExt.restoreContext();
  await both;
  const after = await detector.detect(image);
  const backendAfter = detector.backend;
  const strictAfter = await strict.detect(image);
  detector.dispose();
  strict.dispose();
  const err = (d) => (d ? cornerError(d.quad, c.quad) : null);
  return {
    before: err(before),
    during: err(during),
    backendDuring,
    after: err(after),
    backendAfter,
    strictError,
    strictAfter: err(strictAfter),
  };
}

/** Every texture, framebuffer and program created is deleted by dispose(). */
export async function disposeCheck() {
  const meta = await fixtures();
  const canvas = new OffscreenCanvas(1, 1);
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
  });
  const counts = {};
  for (const kind of ['Texture', 'Framebuffer', 'Program', 'Buffer', 'VertexArray']) {
    const create = gl[`create${kind}`].bind(gl);
    const remove = gl[`delete${kind}`].bind(gl);
    counts[kind] = 0;
    gl[`create${kind}`] = () => {
      counts[kind]++;
      return create();
    };
    gl[`delete${kind}`] = (o) => {
      if (o) counts[kind]--;
      return remove(o);
    };
  }
  const detector = createNnDetector({ ...MODEL, canvas, backend: 'webgl2' });
  for (const c of meta.cases.slice(0, 3)) await detector.detect(await frame(c));
  const live = { ...counts };
  detector.dispose();
  const rejected = await detector.detect(await frame(meta.cases[0])).then(
    () => false,
    () => true,
  );
  return { live, leaked: counts, rejected };
}

const percentile = (sorted, p) =>
  sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];

/** Cold start and warm per-frame timings of one configuration. */
export async function bench({ backend, precision, thread, refine, input, runs, warmup }) {
  const meta = await fixtures();
  const c = meta.cases.find((x) => x.name.startsWith('tilted'));
  let image = await frame(c);
  if (input === 'rgba384') {
    // What the ScannerController hands to detectors by default: the camera frame drawn at a
    // long side of 384 px (here 384×216) by the browser.
    const source = new OffscreenCanvas(image.width, image.height);
    source.getContext('2d').putImageData(image, 0, 0);
    const ctx = new OffscreenCanvas(384, 216).getContext('2d');
    ctx.drawImage(source, 0, 0, 384, 216);
    image = ctx.getImageData(0, 0, 384, 216);
  }
  const next = async () => (input === 'bitmap' ? createImageBitmap(image) : image);
  const cold = {};
  let detector;
  const t0 = performance.now();
  if (thread === 'worker') {
    detector = createNnDetectorWorker({ ...MODEL, backend, precision, refine, timeoutMs: 120000 });
  } else {
    const model = await loadModel(MODEL);
    cold.loadModel = performance.now() - t0;
    const t1 = performance.now();
    detector = createNnDetector({ model, backend, precision, refine });
    await detector.ready();
    cold.init = performance.now() - t1;
  }
  const f0 = await next();
  const t2 = performance.now();
  const first = await detector.detect(f0);
  cold.firstFrame = performance.now() - t2;
  cold.total = performance.now() - t0;
  const times = [];
  for (let i = 0; i < warmup + runs; i++) {
    const f = await next();
    const t = performance.now();
    await detector.detect(f);
    if (i >= warmup) times.push(performance.now() - t);
  }
  const used = detector.backend ?? backend;
  detector.dispose();
  times.sort((a, b) => a - b);
  return {
    backend: used,
    detected: !!first,
    cold,
    p50: percentile(times, 0.5),
    p95: percentile(times, 0.95),
    runs,
  };
}
