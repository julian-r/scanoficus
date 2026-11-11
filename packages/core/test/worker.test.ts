import { describe, expect, it } from 'vitest';
import { convexIoU } from '../src/geometry/index.ts';
import {
  createDetectorWorker,
  createMainThreadDetector,
  type DetectorWorkerLike,
  type DetectRequest,
  type DetectResponse,
  FrameProcessor,
  handleDetectRequest,
} from '../src/worker/index.ts';
import { generateScene } from './support/synthetic.ts';

/** In-process stand-in for a Worker that runs the real request handler asynchronously. */
class FakeWorker implements DetectorWorkerLike {
  readonly processor = new FrameProcessor();
  readonly transfers: Transferable[][] = [];
  private readonly listeners: ((event: MessageEvent<DetectResponse>) => void)[] = [];
  terminated = false;

  postMessage(message: DetectRequest, transfer: Transferable[]): void {
    this.transfers.push(transfer);
    const cloned = structuredClone(message, { transfer });
    setTimeout(() => {
      const data = handleDetectRequest(this.processor, cloned);
      for (const listener of this.listeners) listener(new MessageEvent('message', { data }));
    });
  }

  addEventListener(type: string, listener: (event: never) => void): void {
    if (type === 'message')
      this.listeners.push(listener as (event: MessageEvent<DetectResponse>) => void);
  }

  terminate(): void {
    this.terminated = true;
  }
}

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
});
