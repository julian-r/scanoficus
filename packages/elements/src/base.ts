/**
 * `HTMLElement` when running in a browser, an inert class otherwise, so that importing this
 * package during server-side rendering does not throw (elements are only defined in browsers).
 */
export const BaseElement: typeof HTMLElement =
  typeof HTMLElement === 'undefined' ? (class {} as unknown as typeof HTMLElement) : HTMLElement;

/** Escapes text for interpolation into HTML templates. */
export function escapeHtml(text: string): string {
  return text.replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] as string,
  );
}

/** Dispatches a bubbling, composed `CustomEvent` (crosses shadow roots). */
export function emit<T>(target: EventTarget, type: string, detail: T): boolean {
  return target.dispatchEvent(
    new CustomEvent(type, { detail, bubbles: true, composed: true, cancelable: true }),
  );
}

/** 24×24 stroke icons (inline SVG). */
const ICON_PATHS = {
  close: 'M6 6l12 12M18 6L6 18',
  torch: 'M9 2h6l-1 7h4l-8 13 2-9H8z',
  check: 'M5 12l5 5 9-10',
  rotate: 'M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7',
  trash: 'M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13',
  crop: 'M6 2v16h16M2 6h16v16',
  retake: 'M4 4v6h6M20 12a8 8 0 0 0-14.9-4L4 10M4 12a8 8 0 0 0 14.9 4',
  plus: 'M12 5v14M5 12h14',
  share: 'M12 3v12M7 8l5-5 5 5M5 14v6h14v-6',
  image: 'M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4',
  back: 'M15 5l-7 7 7 7',
} as const;

export type IconName = keyof typeof ICON_PATHS;

export function icon(name: IconName): string {
  return `<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${ICON_PATHS[name]}"/></svg>`;
}

/**
 * Shared styles. Theme with CSS custom properties on the element or any ancestor:
 * `--scan-accent`, `--scan-background`, `--scan-foreground`, `--scan-toolbar-background`,
 * `--scan-quad-fill`, `--scan-quad-stroke`, `--scan-font-family`, `--scan-radius`.
 */
export const BASE_STYLES = `
:host {
  --_accent: var(--scan-accent, #0a84ff);
  --_bg: var(--scan-background, #000);
  --_fg: var(--scan-foreground, #fff);
  --_bar: var(--scan-toolbar-background, rgb(0 0 0 / 0.55));
  --_radius: var(--scan-radius, 12px);
  display: block;
  position: relative;
  overflow: hidden;
  background: var(--_bg);
  color: var(--_fg);
  font-family: var(--scan-font-family, system-ui, -apple-system, sans-serif);
  -webkit-tap-highlight-color: transparent;
  user-select: none;
}
:host([hidden]) { display: none; }
* { box-sizing: border-box; }
button {
  font: inherit;
  color: inherit;
  background: none;
  border: 0;
  padding: 10px;
  min-width: 44px;
  min-height: 44px;
  border-radius: var(--_radius);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  cursor: pointer;
  touch-action: manipulation;
}
button:focus-visible { outline: 2px solid var(--_accent); outline-offset: 2px; }
button:disabled { opacity: 0.4; cursor: default; }
button[aria-pressed='true'] { color: var(--_accent); }
.toolbar {
  position: absolute;
  left: 0;
  right: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px max(12px, env(safe-area-inset-right)) 8px max(12px, env(safe-area-inset-left));
  background: var(--_bar);
  z-index: 2;
}
.toolbar.top { top: 0; padding-top: max(8px, env(safe-area-inset-top)); }
.toolbar.bottom { bottom: 0; padding-bottom: max(12px, env(safe-area-inset-bottom)); }
.pill {
  background: var(--_accent);
  color: #fff;
  padding: 10px 16px;
  font-weight: 600;
}
`;
