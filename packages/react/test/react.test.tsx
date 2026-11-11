import { ScannerController } from '@scanoficus/core/controller';
import { createRgba } from '@scanoficus/core/image';
import type { PageRenderer } from '@scanoficus/core/render';
import { createMainThreadDetector } from '@scanoficus/core/worker';
import type { ScanCompleteDetail, ScanDocumentElement } from '@scanoficus/elements';
import { act, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { DocumentScanner, type DocumentScannerApi, useDocumentScanner } from '../src/index.ts';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const JPEG = new Uint8Array([
  0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x02, 0x00, 0x02, 0x03, 0x01, 0x22, 0x00, 0x02,
  0x11, 0x01, 0x03, 0x11, 0x01, 0xff, 0xd9,
]);
const renderer: PageRenderer = {
  backend: 'cpu',
  render: async () => new Blob([JPEG], { type: 'image/jpeg' }),
  dispose() {},
};

function mount(element: React.ReactNode) {
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  act(() => root.render(element));
  return { container, unmount: () => act(() => root.unmount()) };
}

afterEach(() => {
  document.body.replaceChildren();
});

const captured: { api: DocumentScannerApi | null } = { api: null };

function OwnControllerProbe() {
  const api = useDocumentScanner({ renderer, detector: createMainThreadDetector() });
  captured.api = api;
  return <span>{`${api.state}:${api.pages.length}`}</span>;
}

function ExternalControllerProbe({ controller }: { controller: ScannerController }) {
  const scanner = useDocumentScanner({ controller });
  return <span>{scanner.controller === controller ? 'same' : 'other'}</span>;
}

describe('useDocumentScanner', () => {
  it('re-renders on controller changes and survives StrictMode remounts', async () => {
    const { container, unmount } = mount(
      <StrictMode>
        <OwnControllerProbe />
      </StrictMode>,
    );
    expect(container.textContent).toBe('idle:0');
    const current = captured.api as DocumentScannerApi;
    expect(current.controller.isDisposed).toBe(false);
    const image = createRgba(20, 10);
    await act(async () => {
      await current.addImage(image);
    });
    expect(container.textContent).toBe('idle:1');
    const controller = current.controller;
    unmount();
    expect(controller.isDisposed).toBe(true);
  });

  it('uses an external controller without disposing it', () => {
    const controller = new ScannerController({ renderer });
    const { container, unmount } = mount(<ExternalControllerProbe controller={controller} />);
    expect(container.textContent).toBe('same');
    unmount();
    expect(controller.isDisposed).toBe(false);
  });
});

describe('<DocumentScanner>', () => {
  it('mounts <scan-document> with the controller and bridges events', () => {
    const controller = new ScannerController({ renderer, openCamera: () => new Promise(() => {}) });
    const completed: ScanCompleteDetail[] = [];
    const { container, unmount } = mount(
      <DocumentScanner
        controller={controller}
        pageSize="a4"
        autoCapture={false}
        onComplete={(d) => completed.push(d)}
      />,
    );
    const element = container.querySelector('scan-document') as ScanDocumentElement;
    expect(element).not.toBeNull();
    expect(element.controller).toBe(controller);
    expect(element.getAttribute('page-size')).toBe('a4');
    expect(element.getAttribute('auto-capture')).toBe('false');
    const detail = { pages: [], pdf: new Blob() };
    element.dispatchEvent(new CustomEvent('scan-complete', { detail }));
    expect(completed).toEqual([detail]);
    unmount();
    expect(container.querySelector('scan-document')).toBeNull();
  });

  it('configures the element-created controller from props before it connects', async () => {
    const completed: ScanCompleteDetail[] = [];
    const options = {
      renderer,
      detector: createMainThreadDetector(),
      openCamera: () => new Promise<never>(() => {}),
    };
    const container = document.createElement('div');
    document.body.append(container);
    const root = createRoot(container);
    const render = (pageSize: 'a4' | 'letter') =>
      act(() =>
        root.render(
          <DocumentScanner
            options={options}
            autoCapture={false}
            filter="bw"
            pageSize={pageSize}
            onComplete={(d) => completed.push(d)}
          />,
        ),
      );
    render('a4');
    const element = container.querySelector('scan-document') as ScanDocumentElement;
    const controller = element.controller as ScannerController;
    expect(controller.getSnapshot().autoCapture).toBe(false);
    const page = await act(() => controller.addImage(createRgba(20, 10)));
    expect(page.filter).toBe('bw');

    const review = element.shadowRoot?.querySelector('scan-review') as HTMLElement & {
      save(): Promise<void>;
    };
    const mediaBox = async () => {
      await act(() => review.save());
      const text = new TextDecoder('latin1').decode(await completed.at(-1)!.pdf.arrayBuffer());
      return /\/MediaBox \[([^\]]+)\]/.exec(text)?.[1];
    };
    expect(await mediaBox()).toBe('0 0 595.28 841.89');
    render('letter');
    expect(await mediaBox()).toBe('0 0 612 792');
    act(() => root.unmount());
  });
});
