import { type ScannerCamera, ScannerController } from '@scanoficus/core/controller';
import type { Quad } from '@scanoficus/core/geometry';
import { createRgba, type RgbaImage } from '@scanoficus/core/image';
import type { PageRenderer } from '@scanoficus/core/render';
import { createMainThreadDetector } from '@scanoficus/core/worker';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  defineScanElements,
  fitContain,
  fullQuad,
  moveCorner,
  type ScanCompleteDetail,
  type ScanCropEditorElement,
  type ScanDocumentElement,
  type ScanReviewElement,
} from '../src/index.ts';

// Smallest valid baseline JPEG header with an SOF0 segment is enough for the PDF writer.
const JPEG = new Uint8Array([
  0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x02, 0x00, 0x02, 0x03, 0x01, 0x22, 0x00, 0x02,
  0x11, 0x01, 0x03, 0x11, 0x01, 0xff, 0xd9,
]);

function fakeRenderer(calls: string[] = []): PageRenderer {
  return {
    backend: 'cpu',
    async render(_source, options) {
      calls.push(`${options.filter}:${options.rotation}`);
      return new Blob([JPEG], { type: 'image/jpeg' });
    },
    dispose() {},
  };
}

function gray(width: number, height: number): RgbaImage {
  const image = createRgba(width, height);
  image.data.fill(128);
  return image;
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeAll(() => defineScanElements());

describe('geometry helpers', () => {
  it('fits images like object-fit: contain', () => {
    expect(fitContain(200, 100, 100, 100)).toEqual({
      scale: 0.5,
      x: 0,
      y: 25,
      width: 100,
      height: 50,
    });
  });

  it('moves corners, clamps them to the image and keeps the quad convex', () => {
    const quad = fullQuad(100, 100);
    expect(moveCorner(quad, 0, { x: -10, y: 20 }, 100, 100)?.[0]).toEqual({ x: 0, y: 20 });
    // Dragging the top-left corner past the opposite diagonal would fold the quad.
    expect(moveCorner(quad, 0, { x: 90, y: 90 }, 100, 100)).toBeNull();
  });
});

describe('defineScanElements', () => {
  it('registers all elements idempotently', () => {
    defineScanElements();
    for (const name of ['scan-camera', 'scan-crop-editor', 'scan-review', 'scan-document']) {
      expect(customElements.get(name)).toBeDefined();
    }
  });
});

describe('<scan-review>', () => {
  async function setup() {
    const calls: string[] = [];
    const controller = new ScannerController({
      renderer: fakeRenderer(calls),
      detector: createMainThreadDetector(),
    });
    await controller.addImage(gray(40, 30));
    await controller.addImage(gray(30, 40));
    const review = document.createElement('scan-review') as ScanReviewElement;
    review.setAttribute('page-size', 'a4');
    review.controller = controller;
    document.body.append(review);
    return { controller, review, calls, root: review.shadowRoot as ShadowRoot };
  }

  it('shows the pages and applies filter and rotation to the current page', async () => {
    const { controller, review, root, calls } = await setup();
    expect(root.querySelectorAll('.page img')).toHaveLength(2);
    expect(root.querySelector('.title')?.textContent).toBe('1 of 2');
    (root.querySelector('[data-filter="bw"]') as HTMLButtonElement).click();
    await flush();
    expect(controller.pages[0].filter).toBe('bw');
    expect(root.querySelector('[data-filter="bw"]')?.getAttribute('aria-pressed')).toBe('true');
    (root.querySelector('.rotate') as HTMLButtonElement).click();
    await flush();
    expect(controller.pages[0].rotation).toBe(90);
    expect(calls.at(-1)).toBe('bw:90');
    review.remove();
  });

  it('deletes pages and exports a PDF with scan-complete', async () => {
    const { controller, review, root } = await setup();
    (root.querySelector('.delete') as HTMLButtonElement).click();
    expect(controller.pages).toHaveLength(1);
    const complete = new Promise<ScanCompleteDetail>((resolve) =>
      review.addEventListener('scan-complete', (e) => resolve(e.detail)),
    );
    (root.querySelector('.save') as HTMLButtonElement).click();
    const detail = await complete;
    expect(detail.pages).toHaveLength(1);
    expect(detail.pdf.type).toBe('application/pdf');
    const text = new TextDecoder('latin1').decode(await detail.pdf.arrayBuffer());
    expect(text).toContain('/MediaBox [0 0 595.28 841.89]');
    review.remove();
  });
});

describe('<scan-crop-editor>', () => {
  it('moves the focused corner with the arrow keys and confirms the quad', () => {
    const editor = document.createElement('scan-crop-editor') as ScanCropEditorElement;
    editor.source = gray(200, 100);
    const quad: Quad = [
      { x: 20, y: 20 },
      { x: 180, y: 20 },
      { x: 180, y: 80 },
      { x: 20, y: 80 },
    ];
    editor.quad = quad;
    document.body.append(editor);
    const changes: Quad[] = [];
    editor.addEventListener('quad-change', (e) => changes.push(e.detail.quad));
    const root = editor.shadowRoot as ShadowRoot;
    const handle = root.querySelector('.handle[data-index="1"]') as HTMLElement;
    handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    expect(changes[0][1]).toEqual({ x: 182, y: 20 });
    let confirmed: Quad | null = null;
    editor.addEventListener('crop-confirm', (e) => {
      confirmed = e.detail.quad;
    });
    (root.querySelector('.confirm') as HTMLButtonElement).click();
    expect(confirmed).toEqual(changes[0]);
    (root.querySelector('.reset') as HTMLButtonElement).click();
    expect(editor.quad).toEqual(quad);
    editor.remove();
  });
});

describe('<scan-document>', () => {
  it('switches from camera to review as the controller state changes', async () => {
    let frames: (() => void) | null = null;
    const camera: ScannerCamera = {
      width: 40,
      height: 30,
      features: { torch: false, photo: false },
      onFrame: (cb) => {
        frames = () => cb(0);
        return () => {};
      },
      grabFrame: () => gray(40, 30),
      captureStill: async () => ({
        image: gray(40, 30),
        width: 40,
        height: 30,
        matchesPreview: true,
      }),
      setTorch: async () => false,
      onEnded: () => {},
      stop: () => {},
    };
    const element = document.createElement('scan-document') as ScanDocumentElement;
    element.controller = new ScannerController({
      renderer: fakeRenderer(),
      detector: createMainThreadDetector(),
      openCamera: async () => camera,
      autoCapture: false,
    });
    document.body.append(element);
    await flush();
    const root = element.shadowRoot as ShadowRoot;
    expect(element.controller?.state).toBe('scanning');
    expect((root.querySelector('scan-camera') as HTMLElement).hidden).toBe(false);
    expect(frames).not.toBeNull();
    await element.controller?.capture();
    element.controller?.finish();
    expect((root.querySelector('scan-camera') as HTMLElement).hidden).toBe(true);
    expect((root.querySelector('scan-review') as HTMLElement).hidden).toBe(false);
    expect(element.pages).toHaveLength(1);
    // Live attributes are forwarded after connection.
    element.setAttribute('page-size', 'letter');
    expect(root.querySelector('scan-review')?.getAttribute('page-size')).toBe('letter');
    element.setAttribute('auto-capture', 'true');
    expect(element.controller?.getSnapshot().autoCapture).toBe(true);
    element.remove();
  });
});
