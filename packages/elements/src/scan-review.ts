import type { ScannerController, ScanPage } from '@scanoficus/core/controller';
import { SCAN_FILTERS, type ScanFilter } from '@scanoficus/core/filters';
import type { PdfOptions } from '@scanoficus/core/pdf';
import { BASE_STYLES, BaseElement, emit, escapeHtml, icon } from './base.ts';

const STYLES = `${BASE_STYLES}
:host { height: 100%; }
:host([no-camera]) .add, :host([no-camera]) .retake { visibility: hidden; }
.carousel {
  position: absolute;
  inset: calc(max(8px, env(safe-area-inset-top)) + 60px) 0
    calc(max(12px, env(safe-area-inset-bottom)) + 124px) 0;
  display: flex;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  scrollbar-width: none;
}
.carousel::-webkit-scrollbar { display: none; }
.page {
  flex: 0 0 100%;
  scroll-snap-align: center;
  display: grid;
  place-items: center;
  padding: 12px max(16px, env(safe-area-inset-right)) 12px max(16px, env(safe-area-inset-left));
}
.page img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  box-shadow: 0 4px 24px rgb(0 0 0 / 0.5);
  background: #fff;
}
.title { font-weight: 600; }
.filters {
  position: absolute;
  left: 0;
  right: 0;
  bottom: calc(max(12px, env(safe-area-inset-bottom)) + 64px);
  display: flex;
  justify-content: center;
  gap: 4px;
  padding: 0 8px;
}
.filters button { font-size: 14px; padding: 6px 12px; min-height: 36px; background: var(--_bar); }
.filters button[aria-pressed='true'] { background: var(--_accent); color: #fff; }
.empty { position: absolute; inset: 0; display: grid; place-items: center; opacity: 0.7; }
.busy { opacity: 0.5; }
`;

/** Labels; override via `setMessages` for localization. */
export interface ScanReviewMessages {
  filters: Record<ScanFilter, string>;
  page: (index: number, count: number) => string;
  rotate: string;
  crop: string;
  delete: string;
  retake: string;
  addPages: string;
  save: string;
  exportImages: string;
  empty: string;
}

const DEFAULT_MESSAGES: ScanReviewMessages = {
  filters: { color: 'Color', grayscale: 'Grayscale', bw: 'B&W', photo: 'Photo' },
  page: (i, n) => `${i} of ${n}`,
  rotate: 'Rotate',
  crop: 'Crop',
  delete: 'Delete page',
  retake: 'Retake',
  addPages: 'Scan more pages',
  save: 'Save PDF',
  exportImages: 'Export images',
  empty: 'No pages yet',
};

/**
 * `<scan-review>`: swipe through the scanned pages, change filter, rotate, crop, delete or
 * retake a page, scan more, and export as PDF or images.
 *
 * Set the `controller` property. Attributes: `page-size` (`image` | `a4` | `letter`, PDF page
 * size; default `image`), `no-camera` (hide "Retake" and "Scan more" when there is no live
 * camera, e.g. when pages come from imported photos). Events: `scan-complete` (detail: pages + PDF),
 * `scan-export-images`, `scan-crop` (detail: the page to crop; `<scan-document>` opens its crop
 * editor in response), `scan-error`.
 * Parts: `carousel`, `page`, `filters`, `filter`, `toolbar-top`, `toolbar-bottom`, `save`.
 */
export class ScanReviewElement extends BaseElement {
  private readonly root: ShadowRoot;
  private _controller: ScannerController | null = null;
  private unsubscribe: (() => void) | null = null;
  private messages: ScanReviewMessages = DEFAULT_MESSAGES;
  private index = 0;
  private renderedPages: readonly ScanPage[] = [];
  private busy = false;

  constructor() {
    super();
    this.root = this.attachShadow({ mode: 'open' });
    this.root.innerHTML = `<style>${STYLES}</style>
<div class="toolbar top" part="toolbar-top">
  <button class="add" part="add">${icon('plus')}</button>
  <span class="title" aria-live="polite"></span>
  <button class="save pill" part="save"></button>
</div>
<div class="carousel" part="carousel" tabindex="0"></div>
<div class="empty" hidden></div>
<div class="filters" part="filters" role="group"></div>
<div class="toolbar bottom" part="toolbar-bottom">
  <button class="retake" part="retake">${icon('retake')}</button>
  <button class="crop" part="crop">${icon('crop')}</button>
  <button class="rotate" part="rotate">${icon('rotate')}</button>
  <button class="images" part="images">${icon('share')}</button>
  <button class="delete" part="delete">${icon('trash')}</button>
</div>`;
    const carousel = this.$('.carousel');
    carousel.addEventListener('scroll', () => {
      const i = Math.round(carousel.scrollLeft / Math.max(1, carousel.clientWidth));
      if (i !== this.index) {
        this.index = i;
        this.renderChrome();
      }
    });
    carousel.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') this.goTo(this.index + 1);
      else if (e.key === 'ArrowLeft') this.goTo(this.index - 1);
    });
    this.$('.filters').addEventListener('click', (e) => {
      const filter = (e.target as HTMLElement).closest('button')?.dataset.filter as ScanFilter;
      if (filter)
        void this.edit(
          () => ({ filter }),
          (page) => page.filter !== filter,
        );
    });
    this.$('.rotate').addEventListener('click', () => {
      void this.edit((page) => ({ rotation: page.rotation + 90 }));
    });
    this.$('.crop').addEventListener('click', () => {
      const page = this.current;
      if (page) emit(this, 'scan-crop', { page });
    });
    this.$('.delete').addEventListener('click', () => {
      const page = this.current;
      if (page) this._controller?.removePage(page.id);
    });
    this.$('.retake').addEventListener('click', () => {
      const page = this.current;
      if (page) this._controller?.retake(page.id).catch((error) => this.fail(error));
    });
    this.$('.add').addEventListener('click', () => {
      this._controller?.start().catch((error) => this.fail(error));
    });
    this.$('.save').addEventListener('click', () => void this.save());
    this.$('.images').addEventListener('click', () => {
      const pages = this._controller?.pages ?? [];
      if (pages.length)
        emit(this, 'scan-export-images', { pages, images: pages.map((p) => p.image) });
    });
    this.applyMessages();
  }

  get controller(): ScannerController | null {
    return this._controller;
  }

  set controller(controller: ScannerController | null) {
    this.unsubscribe?.();
    this.unsubscribe = null;
    this._controller = controller;
    if (this.isConnected) this.attach();
  }

  /** The page currently shown. */
  get current(): ScanPage | null {
    return this._controller?.pages[this.index] ?? null;
  }

  /** Localized labels (partial overrides allowed). */
  setMessages(messages: Partial<ScanReviewMessages>): void {
    this.messages = { ...DEFAULT_MESSAGES, ...messages };
    this.applyMessages();
    this.renderPages();
  }

  /** Builds the PDF (honoring `page-size`) and dispatches `scan-complete`. */
  async save(): Promise<void> {
    const controller = this._controller;
    if (!controller || controller.pages.length === 0) return;
    try {
      const pageSize = (this.getAttribute('page-size') ?? 'image') as PdfOptions['pageSize'];
      const pdf = await controller.exportPdf(pageSize ? { pageSize } : {});
      emit(this, 'scan-complete', { pages: controller.pages, pdf });
    } catch (error) {
      this.fail(error);
    }
  }

  connectedCallback(): void {
    this.attach();
  }

  disconnectedCallback(): void {
    this.unsubscribe?.();
    this.unsubscribe = null;
  }

  private attach(): void {
    const controller = this._controller;
    if (!controller || this.unsubscribe) return;
    this.unsubscribe = controller.subscribe(() => this.renderPages());
    this.renderPages();
  }

  private goTo(index: number): void {
    const carousel = this.$('.carousel');
    const count = this._controller?.pages.length ?? 0;
    this.index = Math.max(0, Math.min(count - 1, index));
    carousel.scrollTo({ left: this.index * carousel.clientWidth, behavior: 'smooth' });
    this.renderChrome();
  }

  private async edit(
    changes: (page: ScanPage) => { filter?: ScanFilter; rotation?: number },
    when: (page: ScanPage) => boolean = () => true,
  ): Promise<void> {
    const page = this.current;
    const controller = this._controller;
    if (!page || !controller || this.busy || !when(page)) return;
    this.busy = true;
    this.$('.carousel').classList.add('busy');
    try {
      await controller.updatePage(page.id, changes(page));
    } catch (error) {
      this.fail(error);
    } finally {
      this.busy = false;
      this.$('.carousel').classList.remove('busy');
    }
  }

  private renderPages(): void {
    const pages = this._controller?.pages ?? [];
    if (pages !== this.renderedPages) {
      this.renderedPages = pages;
      this.$('.carousel').innerHTML = pages
        .map(
          (p, i) =>
            `<div class="page" part="page"><img src="${p.url}" alt="${escapeHtml(this.messages.page(i + 1, pages.length))}"></div>`,
        )
        .join('');
      this.index = Math.min(this.index, Math.max(0, pages.length - 1));
      const carousel = this.$('.carousel');
      carousel.scrollLeft = this.index * carousel.clientWidth;
    }
    this.renderChrome();
  }

  private renderChrome(): void {
    const pages = this._controller?.pages ?? [];
    const page = pages[this.index];
    this.$('.title').textContent = page ? this.messages.page(this.index + 1, pages.length) : '';
    const empty = this.$('.empty');
    empty.hidden = pages.length > 0;
    empty.textContent = this.messages.empty;
    for (const selector of ['.save', '.crop', '.rotate', '.delete', '.retake', '.images']) {
      (this.$(selector) as HTMLButtonElement).disabled = !page;
    }
    for (const button of this.root.querySelectorAll<HTMLElement>('.filters button')) {
      button.setAttribute('aria-pressed', String(button.dataset.filter === page?.filter));
    }
  }

  private applyMessages(): void {
    const m = this.messages;
    this.$('.filters').innerHTML = SCAN_FILTERS.map(
      (f) =>
        `<button part="filter" data-filter="${f}" aria-pressed="false">${escapeHtml(m.filters[f])}</button>`,
    ).join('');
    const label = (selector: string, text: string) =>
      this.$(selector).setAttribute('aria-label', text);
    label('.rotate', m.rotate);
    label('.crop', m.crop);
    label('.delete', m.delete);
    label('.retake', m.retake);
    label('.add', m.addPages);
    label('.images', m.exportImages);
    this.$('.save').textContent = m.save;
  }

  private fail(error: unknown): void {
    emit(this, 'scan-error', { error: error instanceof Error ? error : new Error(String(error)) });
  }

  private $(selector: string): HTMLElement {
    return this.root.querySelector(selector) as HTMLElement;
  }
}
