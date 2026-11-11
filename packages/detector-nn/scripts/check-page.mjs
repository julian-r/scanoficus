// In-page half of scripts/browser-check.mjs: parity, robustness and timing checks of the
// built package, loaded unbundled. Every export takes plain JSON arguments and returns JSON.

import {
  bytesToInput,
  CpuRunner,
  createNnDetector,
  createNnDetectorWorker,
  GlRunner,
  loadBundle,
  loadModel,
  parseModel,
  preprocessRgba,
  refineCornersCpu,
  rgbaToRgb,
} from '@scanoficus/detector-nn';
import { DOCQUADNET2 } from '@scanoficus/detector-nn/docquadnet2';

// The weights are resolved from the graph's `weights.file` (and checked against it). The
// detector checks below use the package's default URLs instead.
const MODEL = { graphUrl: '/models/docquadnet/docquadnet.json' };

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
    // GPU preprocessing first, so a failed preprocessing draw cannot reuse this frame's input.
    const bitmap = await createImageBitmap(image);
    const e2eBitmap = maxAbs(runner.run(bitmap), c.heads);
    const inputBitmap = compareInput(runner.readInput(), bytes);
    bitmap.close();
    const e2eHeads = runner.run(image);
    const e2e = maxAbs(e2eHeads, c.heads);
    const e2eSplit = split(e2eHeads, c);
    const input = compareInput(runner.readInput(), bytes);
    const networkHeads = runner.runTensor(bytesToInput(bytes));
    const network = maxAbs(networkHeads, c.heads);
    const networkSplit = split(networkHeads, c);
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
  // Default model URLs (the files next to dist/), raw quads (no clipping) for the comparison.
  const options = { backend: 'webgl2', refine, clipToFrame: false };
  const detector = worker ? createNnDetectorWorker(options) : createNnDetector(options);
  const results = [];
  // With refinement, compare where refine.py converged (see test/refine.test.ts).
  const converged = (c) =>
    Math.max(
      ...c.refined.map(([x, y]) => Math.min(...c.page.map(([u, v]) => Math.hypot(x - u, y - v)))),
    ) < 1;
  for (const c of meta.cases.filter((x) => !refine || !x.detected || converged(x))) {
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
  const detector = createNnDetector({ ...MODEL, canvas, clipToFrame: false });
  const strictCanvas = new OffscreenCanvas(1, 1);
  const strict = createNnDetector({
    ...MODEL,
    canvas: strictCanvas,
    backend: 'webgl2',
    clipToFrame: false,
  });
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
export async function disposeCheck({ v2 } = {}) {
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
  const counts = countObjects(gl);
  const detector = createNnDetector({
    ...(v2 ? { bundle: DOCQUADNET2 } : MODEL),
    canvas,
    backend: 'webgl2',
  });
  for (const c of meta.cases.slice(0, 3)) {
    await detector.detect(await frame(c));
    if (v2) await detector.detectStill(await frame(c)); // refiner, pyramid of a 640×480 frame
  }
  const live = { ...counts };
  detector.dispose();
  const rejected = await detector.detect(await frame(meta.cases[0])).then(
    () => false,
    () => true,
  );
  return { live, leaked: counts, rejected };
}

/** Wraps create/delete calls of a context to count live GPU objects. */
function countObjects(gl) {
  const counts = {};
  for (const kind of ['Texture', 'Framebuffer', 'Program', 'Shader', 'Buffer', 'VertexArray']) {
    const create = gl[`create${kind}`].bind(gl);
    const remove = gl[`delete${kind}`].bind(gl);
    counts[kind] = 0;
    gl[`create${kind}`] = (...args) => {
      counts[kind]++;
      return create(...args);
    };
    gl[`delete${kind}`] = (o) => {
      if (o) counts[kind]--;
      return remove(o);
    };
  }
  return counts;
}

/**
 * A broken GPU path (simulated by a wrong self-test reference) makes the constructor throw and
 * release everything; the 'auto' detector then answers from the CPU and reports why.
 */
export async function selfTestFailure() {
  const graph = await (await fetch(MODEL.graphUrl)).json();
  const weights = await (await fetch(`/models/docquadnet/${graph.weights.file}`)).arrayBuffer();
  graph.selfTest.heads[8] += 1;
  const broken = parseModel(graph, weights);
  const canvas = new OffscreenCanvas(1, 1);
  const counts = countObjects(
    canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
    }),
  );
  let error = null;
  try {
    new GlRunner(broken, { canvas });
  } catch (e) {
    error = String(e);
  }
  const detector = createNnDetector({ model: broken });
  const meta = await fixtures();
  const detection = await detector.detect(await frame(meta.cases[0]));
  const result = {
    error,
    leaked: counts,
    backend: detector.backend,
    glError: detector.glError,
    detected: !!detection,
  };
  detector.dispose();
  return result;
}

/** The synthetic random-weight graphs (test/fixtures/synthetic) on the GPU vs NumPy. */
export async function synthetic({ precision }) {
  const results = [];
  for (const name of ['odd-mix', 'two-channels']) {
    const graph = await (await fetch(`/fixtures/synthetic/${name}.json`)).json();
    const weights = await (await fetch(`/fixtures/synthetic/${name}.bin`)).arrayBuffer();
    const runner = new GlRunner(parseModel(graph, weights), { precision });
    const out = runner.runTensor(Float32Array.from(graph.test.input));
    runner.dispose();
    const error = Math.max(
      ...graph.test.output.map((v, i) => Math.abs(out[i] - v) / Math.max(1, Math.abs(v))),
    );
    results.push({ name, error, length: out.length, expected: graph.test.output.length });
  }
  return results;
}

async function refinerFixtures() {
  return (await fetch('/fixtures/refiner.json')).json();
}

const quadError = (quad, expected) => cornerError(quad, expected);

/**
 * Round 2 on the GPU vs the CPU path (whose crops are bit-exact with refiner.py, see
 * test/refiner.test.ts) and vs the Python fixtures: main heads, per-pass crops and offsets from
 * Python's pass quads, and the whole refinement from Python's raw quad.
 */
export async function refinerParity({ precision }) {
  const meta = await refinerFixtures();
  const bundle = await loadBundle(DOCQUADNET2);
  const runner = new GlRunner(bundle.main, { precision, refiner: bundle.refiner.model });
  const cpu = new CpuRunner(bundle.refiner.model);
  const results = [];
  for (const c of meta.cases) {
    const image = await frame(c);
    const rgb = { width: c.width, height: c.height, data: rgbaToRgb(image) };
    const heads = runner.run(image);
    const r = {
      name: c.name,
      document: c.main.presence > 0.5,
      mainCorners: maxAbs(heads.subarray(0, 8), c.main.heads.slice(0, 8)),
      mainLogit: Math.abs(heads[8] - c.main.heads[8]),
      cropMismatched: 0,
      cropMax: 0,
      offsets: 0,
      offsetsVsCpu: 0,
    };
    const s = c.work.width / c.width;
    for (const pass of c.passes) {
      const q = pass.quad.map(([x, y]) => ({ x: x / s, y: y / s }));
      const settings = { ...bundle.refiner, scales: [pass.scale] };
      const gpu = runner.refine(q, settings);
      const gpuCrops = runner.readCrops();
      const cpuCrops = [];
      const ref = refineCornersCpu(rgb, q, settings, (x, n) => cpu.run(x, n), cpuCrops);
      for (let i = 0; i < gpuCrops.length; i++) {
        const d = Math.abs(gpuCrops[i] - cpuCrops[0][i]);
        if (d) r.cropMismatched++;
        r.cropMax = Math.max(r.cropMax, d);
      }
      gpu.passes[0].offsets.forEach(([dx, dy], n) => {
        const [ex, ey] = pass.offsets[n];
        const [cx, cy] = ref.passes[0].offsets[n];
        r.offsets = Math.max(r.offsets, Math.abs(dx - ex), Math.abs(dy - ey));
        r.offsetsVsCpu = Math.max(r.offsetsVsCpu, Math.abs(dx - cx), Math.abs(dy - cy));
      });
    }
    const raw = c.main.rawQuad.map(([x, y]) => ({ x, y }));
    r.refined = quadError(runner.refine(raw, bundle.refiner).quad, c.refined);
    r.refinedVsOnnx = quadError(runner.refine(raw, bundle.refiner).quad, c.refinedOnnx);
    r.crops = c.passes.length * 4 * 64 * 64 * 3;
    results.push(r);
  }
  runner.dispose();
  return results;
}

/** The round-2 detector API with stills: refined quads vs the Python pipeline. */
export async function stillDetectors({ worker }) {
  const meta = await refinerFixtures();
  const options = { bundle: DOCQUADNET2, backend: 'webgl2', clipToFrame: false };
  const detector = worker ? createNnDetectorWorker(options) : createNnDetector(options);
  const results = [];
  for (const c of meta.cases) {
    const image = await frame(c);
    for (const kind of ['rgba', 'bitmap']) {
      const input = kind === 'rgba' ? image : await createImageBitmap(image);
      const still = await detector.detectStill(input);
      const live = await detector.detect(kind === 'rgba' ? image : await createImageBitmap(image));
      results.push({
        name: c.name,
        kind,
        detected: !!still,
        expected: c.main.presence > 0.98,
        still: still ? quadError(still.quad, c.refined) : null,
        live: live ? quadError(live.quad, c.main.rawQuad) : null,
      });
    }
  }
  detector.dispose();
  return results;
}

/** A 4000×3000 (12 MP) still: the 640×480 page scene drawn large by the browser. */
async function still12(meta) {
  const c = meta.cases.find((x) => x.name.startsWith('page'));
  const source = new OffscreenCanvas(c.width, c.height);
  source.getContext('2d').putImageData(await frame(c), 0, 0);
  const big = new OffscreenCanvas(4000, 3000);
  big.getContext('2d').drawImage(source, 0, 0, 4000, 3000);
  return big;
}

const percentile = (sorted, p) =>
  sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];

/** Cold start and warm per-frame timings of one configuration. */
export async function bench({
  backend,
  precision,
  thread,
  refine,
  input,
  runs,
  warmup,
  v2,
  refinerLive,
  still,
}) {
  const meta = await fixtures();
  const c = meta.cases.find((x) => x.name.startsWith('tilted'));
  let image = await frame(c);
  let big = null;
  if (input === 'still12') big = await still12(meta);
  if (input === 'rgba384') {
    // What the ScannerController hands to detectors by default: the camera frame drawn at a
    // long side of 384 px (here 384×216) by the browser.
    const source = new OffscreenCanvas(image.width, image.height);
    source.getContext('2d').putImageData(image, 0, 0);
    const ctx = new OffscreenCanvas(384, 216).getContext('2d');
    ctx.drawImage(source, 0, 0, 384, 216);
    image = ctx.getImageData(0, 0, 384, 216);
  }
  const next = async () =>
    big ? createImageBitmap(big) : input === 'bitmap' ? createImageBitmap(image) : image;
  const extra = v2 ? { bundle: DOCQUADNET2, refiner: { live: !!refinerLive, still: true } } : {};
  const call = (d, f) => (still ? d.detectStill(f) : d.detect(f));
  const cold = {};
  let detector;
  const t0 = performance.now();
  if (thread === 'worker') {
    detector = createNnDetectorWorker({
      ...(v2 ? {} : MODEL),
      ...extra,
      backend,
      precision,
      refine,
      timeoutMs: 120000,
    });
  } else {
    const loaded = v2 ? await loadBundle(DOCQUADNET2) : { model: await loadModel(MODEL) };
    cold.loadModel = performance.now() - t0;
    const t1 = performance.now();
    detector = createNnDetector({
      ...(v2 ? { ...extra, bundle: loaded } : { model: loaded.model }),
      backend,
      precision,
      refine,
    });
    await detector.ready();
    cold.init = performance.now() - t1;
  }
  const f0 = await next();
  const t2 = performance.now();
  const first = await call(detector, f0);
  cold.firstFrame = performance.now() - t2;
  cold.total = performance.now() - t0;
  const times = [];
  for (let i = 0; i < warmup + runs; i++) {
    const f = await next();
    const t = performance.now();
    await call(detector, f);
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
