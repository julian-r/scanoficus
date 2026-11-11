import { CameraError } from '@scanoficus/core/camera';
import { ScannerController, type ScannerOptions, type ScanPage } from '@scanoficus/core/controller';
import type { ScanFilter } from '@scanoficus/core/filters';
import { BASE_STYLES, BaseElement, emit, escapeHtml } from './base.ts';
import type { QuadDetail, ScanCropDetail } from './events.ts';
import type { ScanCameraElement } from './scan-camera.ts';
import type { ScanCropEditorElement } from './scan-crop-editor.ts';
import type { ScanReviewElement } from './scan-review.ts';

const STYLES = `${BASE_STYLES}
:host { height: 100%; min-height: 320px; }
scan-camera, scan-review, scan-crop-editor, .error { position: absolute; inset: 0; }
.error {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 24px;
  text-align: center;
}
.error[hidden] { display: none; }
.error input { display: none; }
`;

type View = 'camera' | 'review' | 'crop' | 'error';

/**
 * `<scan-document>`: the complete scanning flow (camera with auto-capture → review → crop),
 * built on a `ScannerController`. If the camera is unavailable an error view offers retrying
 * and importing a photo instead.
 *
 * Attributes: `auto-capture` (`"false"` starts in manual mode), `filter` (default filter of new
 * pages: `color` | `grayscale` | `bw` | `photo`), `page-size` (PDF: `image` | `a4` | `letter`).
 * Properties: `controller` (created on first connect unless set; `options` configures it).
 * Events (all bubbling, composed): `scan-complete` ({ pages, pdf }), `scan-export-images`,
 * `scan-cancel`, `scan-error`.
 * Parts: `camera`, `review`, `crop-editor`, `error` (style inner parts of the sub-elements by
 * setting CSS custom properties, which inherit through the shadow roots).
 */
export class ScanDocumentElement extends BaseElement {
  /** Options for the controller created on connect (ignored if `controller` is set). */
  options: ScannerOptions = {};
  private readonly root: ShadowRoot;
  private _controller: ScannerController | null = null;
  private ownsController = false;
  private unsubscribe: (() => void) | null = null;
  private cropping: ScanPage | null = null;
  private view: View | null = null;

  constructor() {
    super();
    this.root = this.attachShadow({ mode: 'open' });
    this.root.innerHTML = `<style>${STYLES}</style>
<scan-camera part="camera" hidden></scan-camera>
<scan-review part="review" hidden></scan-review>
<scan-crop-editor part="crop-editor" hidden></scan-crop-editor>
<div class="error" part="error" role="alert" hidden>
  <p class="message"></p>
  <button class="retry pill">Try again</button>
  <button class="import">Import a photo</button>
  <input type="file" accept="image/*" capture="environment">
</div>`;
    this.root.addEventListener('scan-cancel', (e) => {
      if (e.target === this) return;
      this._controller?.clear();
    });
    this.root.addEventListener('scan-crop', (e) => {
      this.cropping = (e as CustomEvent<ScanCropDetail>).detail.page;
      const editor = this.crop;
      editor.source = this.cropping.source;
      editor.quad = this.cropping.quad;
      this.render();
    });
    this.root.addEventListener('crop-cancel', () => this.closeCrop());
    this.root.addEventListener('crop-confirm', (e) => {
      const page = this.cropping;
      this.closeCrop();
      if (page && this._controller?.pages.some((p) => p.id === page.id)) {
        const { quad } = (e as CustomEvent<QuadDetail>).detail;
        this._controller.updatePage(page.id, { quad }).catch((error) => this.fail(error));
      }
    });
    const input = this.$('input') as HTMLInputElement;
    this.$('.import').addEventListener('click', () => input.click());
    input.addEventListener('change', () => {
      const file = input.files?.[0];
      input.value = '';
      if (file) void this.importImage(file);
    });
    this.$('.retry').addEventListener('click', () => {
      this._controller?.retry().catch((error) => this.fail(error));
    });
  }

  /** The controller driving the flow. */
  get controller(): ScannerController | null {
    return this._controller;
  }

  set controller(controller: ScannerController | null) {
    this.detach();
    if (this.ownsController) this._controller?.dispose();
    this._controller = controller;
    this.ownsController = false;
    if (this.isConnected) this.attach();
  }

  /** Pages scanned so far. */
  get pages(): readonly ScanPage[] {
    return this._controller?.pages ?? [];
  }

  /** Builds the PDF of all pages (same as the review's save button, without the event). */
  exportPdf(): Promise<Blob> {
    if (!this._controller) return Promise.reject(new Error('Not connected'));
    const pageSize = this.getAttribute('page-size') as 'image' | 'a4' | 'letter' | null;
    return this._controller.exportPdf(pageSize ? { pageSize } : {});
  }

  /** Adds a page from an image file (also used by the error view's import button). */
  async importImage(file: Blob): Promise<void> {
    const controller = this._controller;
    if (!controller) return;
    try {
      await controller.addImage(file);
      if (controller.state !== 'scanning') controller.finish();
      this.render();
    } catch (error) {
      this.fail(error);
    }
  }

  connectedCallback(): void {
    if (!this._controller) {
      const autoCapture = this.getAttribute('auto-capture') !== 'false';
      const filter = this.getAttribute('filter') as ScanFilter | null;
      this._controller = new ScannerController({
        autoCapture,
        ...(filter ? { defaultFilter: filter } : {}),
        ...this.options,
      });
      this.ownsController = true;
    }
    this.attach();
  }

  disconnectedCallback(): void {
    this.detach();
    if (this.ownsController) {
      this._controller?.dispose();
      this._controller = null;
      this.ownsController = false;
    }
  }

  private get camera(): ScanCameraElement {
    return this.root.querySelector('scan-camera') as ScanCameraElement;
  }

  private get review(): ScanReviewElement {
    return this.root.querySelector('scan-review') as ScanReviewElement;
  }

  private get crop(): ScanCropEditorElement {
    return this.root.querySelector('scan-crop-editor') as ScanCropEditorElement;
  }

  private attach(): void {
    const controller = this._controller;
    if (!controller || this.unsubscribe) return;
    const offChange = controller.subscribe(() => this.render());
    const offError = controller.on('error', ({ error }) => emit(this, 'scan-error', { error }));
    this.unsubscribe = () => {
      offChange();
      offError();
    };
    const pageSize = this.getAttribute('page-size');
    if (pageSize) this.review.setAttribute('page-size', pageSize);
    this.review.controller = controller;
    this.camera.controller = controller;
    this.render();
  }

  private detach(): void {
    this.unsubscribe?.();
    this.unsubscribe = null;
    this.camera.controller = null;
    this.review.controller = null;
  }

  private closeCrop(): void {
    this.cropping = null;
    this.render();
  }

  private render(): void {
    const controller = this._controller;
    if (!controller) return;
    const state = controller.state;
    let view: View;
    if (this.cropping) view = 'crop';
    else if (state === 'error') view = 'error';
    else if (state === 'reviewing' || (state === 'idle' && controller.pages.length > 0))
      view = 'review';
    else view = 'camera';
    this.camera.hidden = view !== 'camera';
    this.review.hidden = view !== 'review';
    this.crop.hidden = view !== 'crop';
    const error = this.$('.error');
    error.hidden = view !== 'error';
    if (view === 'error') {
      const error = controller.getSnapshot().error;
      let message = error?.message ?? 'The camera is not available.';
      if (error instanceof CameraError && error.code === 'permission-denied') {
        message +=
          '. Allow camera access for this site in your browser settings, or import a photo.';
      }
      this.$('.message').innerHTML = escapeHtml(message);
    }
    if (view !== this.view) {
      this.view = view;
      // Move focus into the new view for keyboard and screen-reader users.
      const target = {
        camera: 'scan-camera',
        review: 'scan-review',
        crop: 'scan-crop-editor',
        error: '.retry',
      }[view];
      const element = this.$(target);
      queueMicrotask(() => {
        const focusable =
          element.shadowRoot?.querySelector<HTMLElement>('button:not([hidden]):not(:disabled)') ??
          element;
        focusable.focus({ preventScroll: true });
      });
    }
  }

  private fail(error: unknown): void {
    emit(this, 'scan-error', { error: error instanceof Error ? error : new Error(String(error)) });
  }

  private $(selector: string): HTMLElement {
    return this.root.querySelector(selector) as HTMLElement;
  }
}
