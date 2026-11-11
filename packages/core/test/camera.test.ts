// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CameraError, type CameraStartStage, openCamera } from '../src/camera/index.ts';

class FakeTrack {
  readyState = 'live';
  stopped = false;
  constraints: unknown[] = [];
  stop() {
    this.stopped = true;
    this.readyState = 'ended';
  }
  getCapabilities() {
    return { torch: true };
  }
  async applyConstraints(c: unknown) {
    this.constraints.push(c);
  }
}

function setup(play: () => Promise<void>, gum?: () => Promise<unknown>) {
  const track = new FakeTrack();
  const stream = { getVideoTracks: () => [track], getTracks: () => [track] };
  const requests: MediaStreamConstraints[] = [];
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: async (c: MediaStreamConstraints) => {
        requests.push(c);
        if (gum) return gum();
        return stream;
      },
    },
  });
  const video = document.createElement('video');
  Object.defineProperty(video, 'videoWidth', { configurable: true, value: 1080 });
  Object.defineProperty(video, 'videoHeight', { configurable: true, value: 1920 });
  // happy-dom does not implement media playback: stub what openCamera uses.
  video.play = play;
  video.requestVideoFrameCallback = (cb) => {
    setTimeout(() => cb(0, {} as VideoFrameCallbackMetadata));
    return 1;
  };
  Object.defineProperty(video, 'srcObject', { configurable: true, writable: true, value: null });
  const bitmaps: { width: number; height: number; source: unknown }[] = [];
  vi.stubGlobal('createImageBitmap', async (source: unknown) => {
    const size = source === video ? { width: 1080, height: 1920 } : { width: 720, height: 1280 };
    const bitmap = { ...size, source, close() {} };
    bitmaps.push(bitmap);
    return bitmap;
  });
  return { track, video, requests, bitmaps };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('openCamera', () => {
  it('starts in stages and reports the actual (portrait) frame size', async () => {
    const { video, requests } = setup(async () => {});
    const stages: CameraStartStage[] = [];
    const session = await openCamera(video, { onStage: (s) => stages.push(s) });
    expect(stages).toEqual(['acquiring', 'playing', 'waiting-for-frames']);
    expect([session.width, session.height]).toEqual([1080, 1920]);
    expect(requests[0].video).toMatchObject({
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      facingMode: { ideal: 'environment' },
    });
    expect(video.muted).toBe(true);
    expect(session.features.torch).toBe(true);
  });

  it('requests the high-resolution preset when asked', async () => {
    const { video, requests } = setup(async () => {});
    await openCamera(video, { resolution: 'max', deviceId: 'cam-2' });
    expect(requests[0].video).toMatchObject({
      width: { ideal: 4096 },
      height: { ideal: 4096 },
      deviceId: { exact: 'cam-2' },
    });
  });

  it('times out a preview that never plays and releases the stream', async () => {
    const { video, track } = setup(() => new Promise(() => {}));
    const error = await openCamera(video, { startTimeoutMs: 20 }).catch((e) => e);
    expect(error).toBeInstanceOf(CameraError);
    expect(error.code).toBe('start-timeout');
    expect(error.recoverable).toBe(true);
    expect(track.stopped).toBe(true);
    expect(video.srcObject).toBeNull();
  });

  it('maps an AbortError from play() to a recoverable "aborted" error, not a denial', async () => {
    const { video, track } = setup(async () => {
      throw new DOMException('The operation was aborted.', 'AbortError');
    });
    const error = await openCamera(video).catch((e) => e);
    expect(error.code).toBe('aborted');
    expect(error.recoverable).toBe(true);
    expect(track.stopped).toBe(true);
  });

  it('reports permission denial as not recoverable', async () => {
    const { video } = setup(
      async () => {},
      async () => {
        throw new DOMException('denied', 'NotAllowedError');
      },
    );
    const error = await openCamera(video).catch((e) => e);
    expect(error.code).toBe('permission-denied');
    expect(error.recoverable).toBe(false);
  });

  it('cancels a pending start through the abort signal', async () => {
    const { video, track } = setup(() => new Promise(() => {}));
    const controller = new AbortController();
    const pending = openCamera(video, { signal: controller.signal, startTimeoutMs: 10_000 });
    setTimeout(() => controller.abort(), 5);
    const error = await pending.catch((e) => e);
    expect(error.code).toBe('aborted');
    expect(track.stopped).toBe(true);
  });
});

describe('CameraSession.captureStill', () => {
  it('captures the video frame at stream resolution by default', async () => {
    const { video } = setup(async () => {});
    const session = await openCamera(video);
    const still = await session.captureStill();
    expect(still).toMatchObject({ width: 1080, height: 1920, method: 'video-frame' });
  });

  it('uses takePhoto() only when opted in and only if it is not smaller than the stream', async () => {
    const { video, bitmaps } = setup(async () => {});
    const photos: string[] = [];
    vi.stubGlobal(
      'ImageCapture',
      class {
        async takePhoto() {
          photos.push('taken');
          return new Blob(['x']);
        }
      },
    );
    const session = await openCamera(video, { stillCapture: 'photo' });
    // The decoded photo (720×1280 from the stub) is smaller than the 1080×1920 stream.
    const still = await session.captureStill();
    expect(photos).toHaveLength(1);
    expect(still).toMatchObject({ width: 1080, height: 1920, method: 'video-frame' });
    expect(bitmaps.map((b) => b.source === video)).toEqual([false, true]);
  });

  it('stops trying takePhoto() after it fails', async () => {
    const { video } = setup(async () => {});
    let calls = 0;
    vi.stubGlobal(
      'ImageCapture',
      class {
        async takePhoto(): Promise<Blob> {
          calls++;
          throw new DOMException('Internal error', 'UnknownError');
        }
      },
    );
    const session = await openCamera(video, { stillCapture: 'photo' });
    expect((await session.captureStill()).method).toBe('video-frame');
    expect((await session.captureStill()).method).toBe('video-frame');
    expect(calls).toBe(1);
  });

  it('switches the torch and turns it off on stop', async () => {
    const { video, track } = setup(async () => {});
    const session = await openCamera(video);
    expect(await session.setTorch(true)).toBe(true);
    session.stop();
    expect(track.constraints).toEqual([
      { advanced: [{ torch: true }] },
      { advanced: [{ torch: false }] },
    ]);
    expect(track.stopped).toBe(true);
  });
});
