import type {
  LiveDetection,
  ScannerController,
  ScannerSnapshot,
} from '@scanoficus/core/controller';
import type { Point, Quad } from '@scanoficus/core/geometry';
import { BASE_STYLES, BaseElement, emit, icon } from './base.ts';
import { svgPoints } from './geometry.ts';

const STYLES = `${BASE_STYLES}
:host { height: 100%; }
video, svg.overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}
svg.overlay { pointer-events: none; }
polygon {
  fill: var(--scan-quad-fill, rgb(10 132 255 / 0.25));
  stroke: var(--scan-quad-stroke, var(--_accent));
  stroke-width: 3;
  stroke-linejoin: round;
  vector-effect: non-scaling-stroke;
  transition: fill 0.2s;
}
polygon.ready { fill: var(--scan-quad-fill-ready, rgb(10 132 255 / 0.45)); }
.status {
  position: absolute;
  left: 50%;
  top: calc(max(8px, env(safe-area-inset-top)) + 72px);
  transform: translateX(-50%);
  background: var(--_bar);
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 15px;
  white-space: nowrap;
  z-index: 2;
}
.status:empty { display: none; }
.flash {
  position: absolute;
  inset: 0;
  background: #fff;
  opacity: 0;
  pointer-events: none;
  z-index: 3;
}
.flash.on { animation: flash 0.35s ease-out; }
@keyframes flash { from { opacity: 0.8; } to { opacity: 0; } }
.shutter {
  width: 72px;
  height: 72px;
  border-radius: 50%;
  border: 4px solid var(--_fg);
  padding: 0;
  background: transparent;
}
.shutter::after {
  content: '';
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: var(--_fg);
  transition: transform 0.1s;
}
.shutter:active::after { transform: scale(0.9); }
.thumb {
  width: 56px;
  height: 56px;
  padding: 0;
  position: relative;
}
.thumb img { width: 48px; height: 48px; object-fit: cover; border-radius: 6px; }
.thumb .count {
  position: absolute;
  top: -2px;
  right: -2px;
  background: var(--_accent);
  border-radius: 999px;
  min-width: 20px;
  height: 20px;
  font-size: 12px;
  font-weight: 600;
  display: grid;
  place-items: center;
}
.thumb:empty, .thumb[data-count='0'] { visibility: hidden; }
@media (orientation: landscape) and (max-height: 540px) {
  .toolbar.bottom {
    top: 0;
    left: auto;
    width: auto;
    flex-direction: column-reverse;
    padding: max(12px, env(safe-area-inset-top)) max(12px, env(safe-area-inset-right))
      max(12px, env(safe-area-inset-bottom)) 12px;
  }
  .toolbar.top { right: auto; flex-direction: column; bottom: 0; padding-left: max(12px, env(safe-area-inset-left)); }
}
`;

/** Status texts; override via the `messages` property for localization. */
export interface ScanCameraMessages {
  requesting: string;
  starting: string;
  searching: string;
  holdStill: string;
  capturing: string;
  manual: string;
  shutter: string;
  done: string;
  cancel: string;
  torch: string;
  auto: string;
  manualMode: string;
  pages: string;
}

const DEFAULT_MESSAGES: ScanCameraMessages = {
  requesting: 'Waiting for camera access…',
  starting: 'Starting camera…',
  searching: 'Position the document in view',
  holdStill: 'Hold steady',
  capturing: 'Capturing…',
  manual: 'Tap the shutter to capture',
  shutter: 'Take photo',
  done: 'Done',
  cancel: 'Cancel',
  torch: 'Flashlight',
  auto: 'Auto',
  manualMode: 'Manual',
  pages: 'Scanned pages',
};

/**
 * `<scan-camera>`: live camera preview with an animated document outline, shutter,
 * auto/manual toggle, torch (when the camera supports it), a thumbnail of the last page with
 * the page count, and Done/Cancel.
 *
 * Set the `controller` property to a `ScannerController`; the element starts the camera when
 * connected and stops it when disconnected. Events: `scan-done`, `scan-cancel`.
 * Parts: `video`, `overlay`, `quad`, `status`, `toolbar-top`, `toolbar-bottom`, `shutter`,
 * `thumbnail`, `done`, `cancel`, `torch`, `mode`.
 */
export class ScanCameraElement extends BaseElement {
  private _controller: ScannerController | null = null;
  private unsubscribe: (() => void) | null = null;
  private messages: ScanCameraMessages = DEFAULT_MESSAGES;
  private readonly root: ShadowRoot;
  private shown: Quad | null = null;
  private target: Quad | null = null;
  private outlineFrame = 0;
  private lastPageCount = 0;

  constructor() {
    super();
    this.root = this.attachShadow({ mode: 'open' });
    this.root.innerHTML = `<style>${STYLES}</style>
<video part="video" playsinline muted autoplay></video>
<svg part="overlay" class="overlay" preserveAspectRatio="xMidYMid slice" viewBox="0 0 1 1"><polygon part="quad" points=""></polygon></svg>
<div class="flash"></div>
<div class="toolbar top" part="toolbar-top">
  <button class="cancel" part="cancel"></button>
  <span>
    <button class="torch" part="torch" aria-pressed="false" hidden>${icon('torch')}</button>
    <button class="mode" part="mode" aria-pressed="true"></button>
  </span>
</div>
<div class="status" part="status" role="status" aria-live="polite"></div>
<div class="toolbar bottom" part="toolbar-bottom">
  <button class="thumb" part="thumbnail" data-count="0"></button>
  <button class="shutter" part="shutter"></button>
  <button class="done pill" part="done"></button>
</div>`;
    this.$('.shutter').addEventListener('click', () => void this._controller?.capture());
    this.$('.done').addEventListener('click', () => this.done());
    this.$('.thumb').addEventListener('click', () => this.done());
    this.$('.cancel').addEventListener('click', () => {
      this._controller?.stop();
      emit(this, 'scan-cancel', undefined);
    });
    this.$('.mode').addEventListener('click', () => {
      const c = this._controller;
      if (c) c.setAutoCapture(!c.getSnapshot().autoCapture);
    });
    this.$('.torch').addEventListener('click', () => {
      const c = this._controller;
      if (c) void c.setTorch(!c.getSnapshot().torch);
    });
    this.applyMessages();
  }

  /** The controller driving this view. */
  get controller(): ScannerController | null {
    return this._controller;
  }

  set controller(controller: ScannerController | null) {
    if (controller === this._controller) return;
    this.detach();
    this._controller = controller;
    if (this.isConnected) this.attach();
  }

  /** Localized labels and status messages (partial overrides allowed). */
  setMessages(messages: Partial<ScanCameraMessages>): void {
    this.messages = { ...DEFAULT_MESSAGES, ...messages };
    this.applyMessages();
  }

  /** The `<video>` element showing the preview. */
  get video(): HTMLVideoElement {
    return this.$('video') as HTMLVideoElement;
  }

  connectedCallback(): void {
    this.attach();
  }

  disconnectedCallback(): void {
    this.detach();
  }

  private attach(): void {
    const controller = this._controller;
    if (!controller || this.unsubscribe) return;
    this.unsubscribe = controller.subscribe(() => this.renderSnapshot(controller.getSnapshot()));
    this.lastPageCount = controller.pages.length;
    this.renderSnapshot(controller.getSnapshot());
    if (controller.state !== 'scanning' && controller.state !== 'starting') {
      void controller.start(this.video);
    }
  }

  private detach(): void {
    this.unsubscribe?.();
    this.unsubscribe = null;
    cancelAnimationFrame(this.outlineFrame);
    this.outlineFrame = 0;
    const c = this._controller;
    if (c && (c.state === 'scanning' || c.state === 'starting' || c.state === 'capturing'))
      c.stop();
  }

  private done(): void {
    this._controller?.finish();
    emit(this, 'scan-done', undefined);
  }

  private renderSnapshot(s: ScannerSnapshot): void {
    const m = this.messages;
    const torch = this.$('.torch');
    torch.hidden = !s.features?.torch;
    torch.setAttribute('aria-pressed', String(s.torch));
    const mode = this.$('.mode');
    mode.setAttribute('aria-pressed', String(s.autoCapture));
    mode.textContent = s.autoCapture ? m.auto : m.manualMode;
    (this.$('.shutter') as HTMLButtonElement).disabled = s.state !== 'scanning';
    (this.$('.done') as HTMLButtonElement).disabled = s.pages.length === 0;
    this.renderThumbnail(s);
    this.$('.status').textContent = this.statusText(s);
    this.renderDetection(s.detection);
    if (s.pages.length > this.lastPageCount) {
      const flash = this.$('.flash');
      flash.classList.remove('on');
      void (flash as HTMLElement).offsetWidth; // restart the animation
      flash.classList.add('on');
    }
    this.lastPageCount = s.pages.length;
  }

  private renderThumbnail(s: ScannerSnapshot): void {
    const thumb = this.$('.thumb');
    const last = s.pages.at(-1);
    thumb.dataset.count = String(s.pages.length);
    thumb.setAttribute('aria-label', `${this.messages.pages}: ${s.pages.length}`);
    const img = thumb.querySelector('img');
    if (!last) {
      thumb.replaceChildren();
    } else if (img?.getAttribute('src') !== last.url) {
      thumb.innerHTML = `<img alt="" src="${last.url}"><span class="count">${s.pages.length}</span>`;
    }
  }

  private statusText(s: ScannerSnapshot): string {
    const m = this.messages;
    if (s.state === 'starting') return s.cameraStage === 'acquiring' ? m.requesting : m.starting;
    if (s.state === 'capturing') return m.capturing;
    if (s.state !== 'scanning') return '';
    if (!s.autoCapture) return m.manual;
    if (!s.detection?.quad) return m.searching;
    return s.detection.progress > 0 ? m.capturing : m.holdStill;
  }

  private renderDetection(d: LiveDetection | null): void {
    const svg = this.$('svg');
    if (d) svg.setAttribute('viewBox', `0 0 ${d.frameWidth} ${d.frameHeight}`);
    this.target = d?.quad ?? null;
    this.$('polygon').classList.toggle('ready', (d?.progress ?? 0) > 0);
    if (!this.outlineFrame) this.outlineFrame = requestAnimationFrame(() => this.stepOutline());
  }

  /** Eases the drawn outline towards the latest detection for a smooth overlay. */
  private stepOutline(): void {
    this.outlineFrame = 0;
    const polygon = this.$('polygon');
    const target = this.target;
    if (!target) {
      this.shown = null;
      polygon.setAttribute('points', '');
      return;
    }
    const shown = this.shown;
    const next = shown
      ? (shown.map((p, i) => lerp(p, target[i], 0.35)) as unknown as Quad)
      : target;
    this.shown = next;
    polygon.setAttribute('points', svgPoints(next));
    const moving = next.some((p, i) => Math.hypot(p.x - target[i].x, p.y - target[i].y) > 0.5);
    if (moving) this.outlineFrame = requestAnimationFrame(() => this.stepOutline());
  }

  private applyMessages(): void {
    const m = this.messages;
    const label = (selector: string, text: string) =>
      this.$(selector).setAttribute('aria-label', text);
    label('.shutter', m.shutter);
    label('.torch', m.torch);
    this.$('.cancel').textContent = m.cancel;
    this.$('.done').textContent = m.done;
  }

  private $(selector: string): HTMLElement {
    return this.root.querySelector(selector) as HTMLElement;
  }
}

function lerp(a: Point, b: Point, t: number): Point {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}
