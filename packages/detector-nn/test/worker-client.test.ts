import { describe, expect, it, vi } from 'vitest';
import { checkPresenceModule, createNnDetectorWorker } from '../src/index.ts';

/** A worker stand-in: records messages, lets the test answer or fail. */
class FakeWorker extends EventTarget {
  sent: unknown[] = [];
  terminated = false;
  throwOnPost: Error | null = null;
  postMessage(message: unknown) {
    if (this.throwOnPost && (message as { nnStill?: unknown }).nnStill) throw this.throwOnPost;
    this.sent.push(message);
  }
  terminate() {
    this.terminated = true;
  }
  emit(type: string, data?: unknown, message?: string) {
    this.dispatchEvent(Object.assign(new Event(type), { data, message }));
  }
}

const rgba = () => ({ width: 2, height: 2, data: new Uint8ClampedArray(16) });
const start = (timeoutMs = 60_000) => {
  const worker = new FakeWorker();
  const detector = createNnDetectorWorker({ worker: worker as unknown as Worker, timeoutMs });
  detector.ready().catch(() => {});
  worker.emit('message', { ready: true });
  worker.emit('message', { nnReady: { backend: 'webgl2', precision: 'float32', ms: 1 } });
  return { worker, detector };
};

describe('createNnDetectorWorker: still requests', () => {
  it('answers stills by id', async () => {
    const { worker, detector } = start();
    const still = detector.detectStill(rgba());
    const id = (worker.sent.at(-1) as { nnStill: { id: number } }).nnStill.id;
    worker.emit('message', { nnStillResult: { id, detection: null } });
    await expect(still).resolves.toBeNull();
  });

  it('rejects pending and later stills at once when the worker fails', async () => {
    const { worker, detector } = start();
    const pending = detector.detectStill(rgba());
    worker.emit('error', undefined, 'boom');
    await expect(pending).rejects.toThrow(/worker failed: boom/);
    await expect(detector.detectStill(rgba())).rejects.toThrow(/worker failed: boom/);
  });

  it('fails stills when the core client reports a terminal failure', async () => {
    vi.useFakeTimers();
    try {
      const { worker, detector } = start(1000);
      const detect = detector.detect(rgba());
      const still = detector.detectStill(rgba());
      const rejected = expect(still).rejects.toThrow(/did not answer/);
      const detectRejected = expect(detect).rejects.toThrow(/did not answer/);
      await vi.advanceTimersByTimeAsync(1000);
      await detectRejected;
      await rejected;
      expect(worker.terminated).toBe(true);
      await expect(detector.detectStill(rgba())).rejects.toThrow(/did not answer/);
    } finally {
      vi.useRealTimers();
    }
  });

  it('cleans up when postMessage throws (DataCloneError)', async () => {
    vi.useFakeTimers();
    try {
      const { worker, detector } = start(1000);
      worker.throwOnPost = new Error('DataCloneError');
      await expect(detector.detectStill(rgba())).rejects.toThrow('DataCloneError');
      expect(vi.getTimerCount()).toBe(0);
      worker.throwOnPost = null;
      const still = detector.detectStill(rgba());
      const id = (worker.sent.at(-1) as { nnStill: { id: number } }).nnStill.id;
      worker.emit('message', { nnStillResult: { id, detection: null } });
      await expect(still).resolves.toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('rejects stills after dispose', async () => {
    const { worker, detector } = start();
    const pending = detector.detectStill(rgba());
    detector.dispose();
    await expect(pending).rejects.toThrow('Detector disposed');
    await expect(detector.detectStill(rgba())).rejects.toThrow('Detector disposed');
    expect(worker.terminated).toBe(true);
  });
});

describe('checkPresenceModule', () => {
  const origin = 'https://app.test';
  it('accepts http(s) modules of the worker origin only', () => {
    expect(checkPresenceModule('https://app.test/rule.js', origin)).toBe(
      'https://app.test/rule.js',
    );
    for (const url of [
      'https://evil.test/rule.js',
      'data:text/javascript,1',
      'blob:https://app.test/x',
    ]) {
      expect(() => checkPresenceModule(url, origin)).toThrow(/Refusing presenceModule/);
    }
  });
  it('accepts other origins with the explicit opt-in', () => {
    expect(checkPresenceModule('https://cdn.test/rule.js', origin, true)).toBe(
      'https://cdn.test/rule.js',
    );
  });
});
