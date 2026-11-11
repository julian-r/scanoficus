import type { Point, Quad } from '@scanoficus/core/geometry';
import type { RgbaImage } from '@scanoficus/core/image';
import type { PageSource } from '@scanoficus/core/render';
import { BASE_STYLES, BaseElement, emit, icon } from './base.ts';
import { type Fit, fitContain, fullQuad, moveCorner, svgPoints } from './geometry.ts';

const LOUPE_SIZE = 112;
const LOUPE_ZOOM = 2.5;
const CORNER_NAMES = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left'];

const STYLES = `${BASE_STYLES}
:host { height: 100%; touch-action: none; }
.stage {
  position: absolute;
  inset: calc(max(8px, env(safe-area-inset-top)) + 60px) max(16px, env(safe-area-inset-right))
    calc(max(12px, env(safe-area-inset-bottom)) + 72px) max(16px, env(safe-area-inset-left));
}
canvas.image, svg { position: absolute; inset: 0; width: 100%; height: 100%; }
svg { overflow: visible; }
.shade { fill: rgb(0 0 0 / 0.55); fill-rule: evenodd; }
.outline {
  fill: none;
  stroke: var(--scan-quad-stroke, var(--_accent));
  stroke-width: 2;
}
.handle {
  position: absolute;
  width: 44px;
  height: 44px;
  margin: -22px 0 0 -22px;
  padding: 0;
  border-radius: 50%;
  touch-action: none;
}
.handle::after {
  content: '';
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: 3px solid var(--scan-quad-stroke, var(--_accent));
  background: rgb(255 255 255 / 0.25);
}
canvas.loupe {
  position: absolute;
  width: ${LOUPE_SIZE}px;
  height: ${LOUPE_SIZE}px;
  border-radius: 50%;
  border: 3px solid var(--_fg);
  box-shadow: 0 4px 16px rgb(0 0 0 / 0.5);
  pointer-events: none;
  display: none;
  z-index: 3;
}
canvas.loupe.on { display: block; }
`;

/**
 * `<scan-crop-editor>`: adjust the four corners of a page. Corners are dragged with mouse,
 * pen or touch (a magnifier loupe shows the area under the finger) or moved with the arrow
 * keys (Shift for larger steps); moves that would make the quad non-convex are rejected.
 *
 * Properties: `source` (the page's source image), `quad` (corners in source pixels).
 * Events: `quad-change` (after each move), `crop-confirm`, `crop-cancel`.
 * Parts: `image`, `outline`, `handle`, `loupe`, `toolbar-top`, `toolbar-bottom`, `confirm`,
 * `cancel`, `reset`.
 */
export class ScanCropEditorElement extends BaseElement {
  private readonly root: ShadowRoot;
  private _source: PageSource | null = null;
  private _quad: Quad | null = null;
  private initialQuad: Quad | null = null;
  private fit: Fit = { scale: 1, x: 0, y: 0, width: 0, height: 0 };
  private resizeObserver: ResizeObserver | null = null;
  private dragging = -1;
  private bitmap: CanvasImageSource | null = null;

  constructor() {
    super();
    this.root = this.attachShadow({ mode: 'open' });
    this.root.innerHTML = `<style>${STYLES}</style>
<div class="toolbar top" part="toolbar-top">
  <button class="cancel" part="cancel">Cancel</button>
  <button class="reset" part="reset">Reset</button>
</div>
<div class="stage">
  <canvas class="image" part="image"></canvas>
  <svg aria-hidden="true"><path class="shade"></path><polygon class="outline" part="outline"></polygon></svg>
  ${CORNER_NAMES.map((name, i) => `<button class="handle" part="handle" data-index="${i}" aria-label="${name} corner"></button>`).join('')}
  <canvas class="loupe" part="loupe" width="${LOUPE_SIZE * 2}" height="${LOUPE_SIZE * 2}"></canvas>
</div>
<div class="toolbar bottom" part="toolbar-bottom">
  <span></span>
  <button class="confirm pill" part="confirm">${icon('check')} Done</button>
</div>`;
    this.$('.cancel').addEventListener('click', () => emit(this, 'crop-cancel', undefined));
    this.$('.confirm').addEventListener('click', () => {
      if (this._quad) emit(this, 'crop-confirm', { quad: this._quad });
    });
    this.$('.reset').addEventListener('click', () => {
      const s = this.sourceSize();
      if (!s) return;
      this.quad = this.initialQuad ?? fullQuad(s.width, s.height);
      this.changed();
    });
    for (const handle of this.root.querySelectorAll<HTMLElement>('.handle')) {
      const index = Number(handle.dataset.index);
      handle.addEventListener('pointerdown', (e) => this.startDrag(e, index));
      handle.addEventListener('pointermove', (e) => this.drag(e, index));
      handle.addEventListener('pointerup', (e) => this.endDrag(e));
      handle.addEventListener('pointercancel', (e) => this.endDrag(e));
      handle.addEventListener('keydown', (e) => this.onKey(e, index));
    }
  }

  /** The image the page is cut from (ImageBitmap, canvas, image element or RGBA pixels). */
  get source(): PageSource | null {
    return this._source;
  }

  set source(source: PageSource | null) {
    this._source = source;
    this.initialQuad = null;
    this.bitmap = source && 'data' in source ? rgbaToCanvas(source) : source;
    this.layout();
  }

  /** Current corners in source pixels. Setting it also defines what "Reset" restores. */
  get quad(): Quad | null {
    return this._quad;
  }

  set quad(quad: Quad | null) {
    this._quad = quad;
    this.initialQuad ??= quad;
    this.renderQuad();
  }

  connectedCallback(): void {
    this.resizeObserver = new ResizeObserver(() => this.layout());
    this.resizeObserver.observe(this.$('.stage'));
    this.layout();
    (this.root.querySelector('.handle') as HTMLElement | null)?.focus({ preventScroll: true });
  }

  disconnectedCallback(): void {
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
  }

  private sourceSize(): { width: number; height: number } | null {
    const s = this._source;
    if (!s) return null;
    return 'naturalWidth' in s ? { width: s.naturalWidth, height: s.naturalHeight } : s;
  }

  private layout(): void {
    const size = this.sourceSize();
    const stage = this.$('.stage');
    const canvas = this.$('canvas.image') as HTMLCanvasElement;
    if (!size || !this.bitmap || stage.clientWidth === 0) return;
    const dpr = globalThis.devicePixelRatio || 1;
    canvas.width = Math.round(stage.clientWidth * dpr);
    canvas.height = Math.round(stage.clientHeight * dpr);
    this.fit = fitContain(size.width, size.height, stage.clientWidth, stage.clientHeight);
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, stage.clientWidth, stage.clientHeight);
      ctx.drawImage(this.bitmap, this.fit.x, this.fit.y, this.fit.width, this.fit.height);
    }
    this.renderQuad();
  }

  private toScreen(p: Point): Point {
    return { x: this.fit.x + p.x * this.fit.scale, y: this.fit.y + p.y * this.fit.scale };
  }

  private renderQuad(): void {
    const quad = this._quad;
    const stage = this.$('.stage');
    const handles = this.root.querySelectorAll<HTMLElement>('.handle');
    if (!quad || this.fit.scale === 0) {
      for (const h of handles) h.hidden = true;
      return;
    }
    const screen = quad.map((p) => this.toScreen(p));
    const w = stage.clientWidth;
    const h = stage.clientHeight;
    this.$('.outline').setAttribute('points', svgPoints(screen));
    this.$('.shade').setAttribute(
      'd',
      `M0 0H${w}V${h}H0Z M${screen.map((p) => `${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join('L')}Z`,
    );
    handles.forEach((handle, i) => {
      handle.hidden = false;
      handle.style.left = `${screen[i].x}px`;
      handle.style.top = `${screen[i].y}px`;
    });
  }

  private startDrag(event: PointerEvent, index: number): void {
    if (!this._quad) return;
    event.preventDefault();
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    this.dragging = index;
    this.drag(event, index);
  }

  private drag(event: PointerEvent, index: number): void {
    if (this.dragging !== index || !this._quad) return;
    const size = this.sourceSize();
    if (!size) return;
    const rect = this.$('.stage').getBoundingClientRect();
    const point = {
      x: (event.clientX - rect.left - this.fit.x) / this.fit.scale,
      y: (event.clientY - rect.top - this.fit.y) / this.fit.scale,
    };
    const next = moveCorner(this._quad, index, point, size.width, size.height);
    if (next) {
      this._quad = next;
      this.renderQuad();
    }
    this.showLoupe(this._quad[index]);
  }

  private endDrag(event: PointerEvent): void {
    if (this.dragging < 0) return;
    (event.currentTarget as HTMLElement).releasePointerCapture?.(event.pointerId);
    this.dragging = -1;
    this.$('.loupe').classList.remove('on');
    this.changed();
  }

  private onKey(event: KeyboardEvent, index: number): void {
    const size = this.sourceSize();
    const step = (event.shiftKey ? 0.05 : 0.01) * Math.max(size?.width ?? 0, size?.height ?? 0);
    const delta: Record<string, Point> = {
      ArrowLeft: { x: -step, y: 0 },
      ArrowRight: { x: step, y: 0 },
      ArrowUp: { x: 0, y: -step },
      ArrowDown: { x: 0, y: step },
    };
    const d = delta[event.key];
    if (!d || !this._quad || !size) return;
    event.preventDefault();
    const p = this._quad[index];
    const next = moveCorner(
      this._quad,
      index,
      { x: p.x + d.x, y: p.y + d.y },
      size.width,
      size.height,
    );
    if (next) {
      this._quad = next;
      this.renderQuad();
      this.changed();
    }
  }

  /** Magnified view around `p`, drawn above-left of the finger so it stays visible. */
  private showLoupe(p: Point): void {
    const loupe = this.$('.loupe') as HTMLCanvasElement;
    const ctx = loupe.getContext('2d');
    if (!ctx || !this.bitmap) return;
    const zoom = LOUPE_ZOOM * this.fit.scale; // source → loupe CSS pixels
    const span = LOUPE_SIZE / zoom; // source pixels across the loupe
    ctx.save();
    ctx.scale(2, 2);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, LOUPE_SIZE, LOUPE_SIZE);
    ctx.drawImage(
      this.bitmap,
      p.x - span / 2,
      p.y - span / 2,
      span,
      span,
      0,
      0,
      LOUPE_SIZE,
      LOUPE_SIZE,
    );
    ctx.strokeStyle = getComputedStyle(this).getPropertyValue('--scan-accent').trim() || '#0a84ff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(LOUPE_SIZE / 2, LOUPE_SIZE / 2 - 12);
    ctx.lineTo(LOUPE_SIZE / 2, LOUPE_SIZE / 2 + 12);
    ctx.moveTo(LOUPE_SIZE / 2 - 12, LOUPE_SIZE / 2);
    ctx.lineTo(LOUPE_SIZE / 2 + 12, LOUPE_SIZE / 2);
    ctx.stroke();
    ctx.restore();
    const s = this.toScreen(p);
    const above = s.y - LOUPE_SIZE - 40;
    loupe.style.left = `${Math.max(0, s.x - LOUPE_SIZE / 2)}px`;
    loupe.style.top = `${above < 0 ? s.y + 40 : above}px`;
    loupe.classList.add('on');
  }

  private changed(): void {
    if (this._quad) emit(this, 'quad-change', { quad: this._quad });
  }

  private $(selector: string): HTMLElement {
    return this.root.querySelector(selector) as HTMLElement;
  }
}

/** Puts RGBA pixels on a canvas so they can be drawn with `drawImage`. */
function rgbaToCanvas(image: RgbaImage): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const pixels = image.data as Uint8ClampedArray<ArrayBuffer>;
  canvas.getContext('2d')?.putImageData(new ImageData(pixels, image.width, image.height), 0, 0);
  return canvas;
}
