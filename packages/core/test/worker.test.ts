import { describe, expect, it } from 'vitest';
import { convexIoU } from '../src/geometry/index.ts';
import {
  createDetectorWorker,
  createMainThreadDetector,
  FrameProcessor,
  handleDetectRequest,
} from '../src/worker/index.ts';
import { FakeWorker } from './support/fake-worker.ts';
import { generateScene } from './support/synthetic.ts';

describe('handleDetectRequest', () => {
  it('answers with the detection for the request id', () => {
    const scene = generateScene('easy', 21);
    const response = handleDetectRequest(new FrameProcessor(), { id: 7, frame: scene.image });
    expect(response.id).toBe(7);
    expect('detection' in response && response.detection).toBeTruthy();
  });

  it('turns invalid frames into error responses', () => {
    const frame = { width: 10, height: 10, data: new Uint8ClampedArray(3) };
    const response = handleDetectRequest(new FrameProcessor(), { id: 1, frame });
    expect(response).toEqual({ id: 1, error: expect.stringContaining('RGBA') });
  });
});

describe('createMainThreadDetector', () => {
  it('detects documents in RGBA frames and honors dispose', async () => {
    const scene = generateScene('perspective', 3, 640, 480);
    const detector = createMainThreadDetector();
    const detection = await detector.detect(scene.image);
    expect(convexIoU(detection!.quad, scene.quad!)).toBeGreaterThan(0.95);
    detector.dispose();
    await expect(detector.detect(scene.image)).rejects.toThrow('disposed');
  });
});

describe('createDetectorWorker', () => {
  it('resolves detections from the worker and copies pixels by default', async () => {
    const worker = new FakeWorker();
    const detector = createDetectorWorker({ worker });
    const scene = generateScene('easy', 4);
    const [a, b] = await Promise.all([
      detector.detect(scene.image),
      detector.detect(generateScene('none', 4).image),
    ]);
    expect(convexIoU(a!.quad, scene.quad!)).toBeGreaterThan(0.95);
    expect(b).toBeNull();
    expect(worker.transfers[0]).toEqual([]);
    expect(scene.image.data.length).toBeGreaterThan(0);
  });

  it('transfers pixel buffers when asked to', async () => {
    const worker = new FakeWorker();
    const detector = createDetectorWorker({ worker, transferPixels: true });
    const scene = generateScene('easy', 5);
    await detector.detect(scene.image, { minScore: 0.9 });
    expect(worker.transfers[0]).toHaveLength(1);
    expect(scene.image.data.length).toBe(0); // detached
  });

  it('rejects errors and pending requests on dispose', async () => {
    const worker = new FakeWorker();
    const detector = createDetectorWorker({ worker });
    const bad = { width: 2, height: 2, data: new Uint8ClampedArray(1) };
    await expect(detector.detect(bad)).rejects.toThrow('RGBA');
    const pending = detector.detect(generateScene('easy', 6).image);
    detector.dispose();
    await expect(pending).rejects.toThrow('disposed');
    expect(worker.terminated).toBe(true);
  });

  it('fails permanently when the worker cannot load, even before the first request', async () => {
    const worker = new FakeWorker({ ready: false });
    const detector = createDetectorWorker({ worker });
    worker.error(); // e.g. 404 for the worker script: browsers give no details
    await expect(detector.detect(generateScene('easy', 1).image)).rejects.toThrow(
      /failed to load: the browser gave no details \(worker: the provided worker\)/,
    );
    expect(worker.terminated).toBe(true);
    await expect(detector.detect(generateScene('easy', 2).image)).rejects.toThrow('failed to load');
  });

  it('reports crashes between requests with their location and rejects later requests', async () => {
    const worker = new FakeWorker();
    const detector = createDetectorWorker({ worker });
    await detector.detect(generateScene('easy', 3).image);
    worker.error({ message: 'boom', filename: 'worker-entry.js', lineno: 3, colno: 7 });
    await expect(detector.detect(generateScene('easy', 3).image)).rejects.toThrow(
      'Detector worker crashed: boom at worker-entry.js:3:7',
    );
  });

  it('fails on messageerror and on a worker that stops answering', async () => {
    const broken = new FakeWorker();
    const a = createDetectorWorker({ worker: broken });
    broken.messageError();
    await expect(a.detect(generateScene('easy', 4).image)).rejects.toThrow('deserialize');

    const silent = new FakeWorker();
    silent.responsive = false;
    const b = createDetectorWorker({ worker: silent, timeoutMs: 20 });
    await expect(b.detect(generateScene('easy', 4).image)).rejects.toThrow('did not answer');
    expect(silent.terminated).toBe(true);
    await expect(b.detect(generateScene('easy', 4).image)).rejects.toThrow('did not answer');
  });
});
