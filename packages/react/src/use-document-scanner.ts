import {
  type PageChanges,
  ScannerController,
  type ScannerOptions,
  type ScannerSnapshot,
  type ScanPage,
} from '@scanoficus/core/controller';
import type { PdfOptions } from '@scanoficus/core/pdf';
import type { PageSource } from '@scanoficus/core/render';
import { useEffect, useMemo, useReducer, useRef, useSyncExternalStore } from 'react';

/** Options of {@link useDocumentScanner}. */
export interface UseDocumentScannerOptions extends ScannerOptions {
  /** Use an existing controller instead of creating one (the hook then does not dispose it). */
  controller?: ScannerController;
}

/** State and actions returned by {@link useDocumentScanner}. */
export interface DocumentScannerApi extends ScannerSnapshot {
  controller: ScannerController;
  /** Opens the camera into the video element and starts live detection. */
  start(video?: HTMLVideoElement): Promise<void>;
  stop(): void;
  /** Retries opening the camera after an error. */
  retry(): Promise<void>;
  capture(): Promise<ScanPage | null>;
  finish(): void;
  setAutoCapture(enabled: boolean): void;
  setTorch(on: boolean): Promise<void>;
  addImage(image: Blob | PageSource): Promise<ScanPage>;
  updatePage(id: string, changes: PageChanges): Promise<ScanPage>;
  removePage(id: string): void;
  movePage(id: string, toIndex: number): void;
  retake(id: string): Promise<void>;
  clear(): void;
  exportPdf(options?: PdfOptions): Promise<Blob>;
}

/**
 * React binding of the headless `ScannerController`, for building custom scanner UIs:
 * re-renders on every state change (via `useSyncExternalStore`) and exposes the controller's
 * actions. The controller is created lazily (no browser APIs are touched during server
 * rendering) and disposed on unmount. Options are read when the controller is created.
 *
 * ```tsx
 * const scanner = useDocumentScanner({ autoCapture: true });
 * <video ref={(v) => { if (v && scanner.state === 'idle') void scanner.start(v); }} />
 * ```
 */
export function useDocumentScanner(options: UseDocumentScannerOptions = {}): DocumentScannerApi {
  const optionsRef = useRef(options);
  optionsRef.current = options;
  const external = options.controller;
  // Constructing a controller has no side effects (the detector worker is created lazily),
  // so creating it during render is safe on the server and in discarded renders.
  const own = useRef<ScannerController | null>(null);
  if (!external && !own.current) own.current = new ScannerController(options);
  const [, rerender] = useReducer((n: number) => n + 1, 0);

  useEffect(() => {
    if (external) return;
    // React StrictMode runs effect cleanups once on mount: replace the disposed controller.
    if (!own.current || own.current.isDisposed) {
      own.current = new ScannerController(optionsRef.current);
      rerender();
    }
    const controller = own.current;
    return () => controller.dispose();
  }, [external]);

  const controller = (external ?? own.current) as ScannerController;
  const snapshot = useSyncExternalStore(
    controller.subscribe,
    controller.getSnapshot,
    controller.getSnapshot,
  );
  const actions = useMemo(
    () => ({
      controller,
      start: (video?: HTMLVideoElement) => controller.start(video),
      stop: () => controller.stop(),
      retry: () => controller.retry(),
      capture: () => controller.capture(),
      finish: () => controller.finish(),
      setAutoCapture: (enabled: boolean) => controller.setAutoCapture(enabled),
      setTorch: (on: boolean) => controller.setTorch(on),
      addImage: (image: Blob | PageSource) => controller.addImage(image),
      updatePage: (id: string, changes: PageChanges) => controller.updatePage(id, changes),
      removePage: (id: string) => controller.removePage(id),
      movePage: (id: string, toIndex: number) => controller.movePage(id, toIndex),
      retake: (id: string) => controller.retake(id),
      clear: () => controller.clear(),
      exportPdf: (pdfOptions?: PdfOptions) => controller.exportPdf(pdfOptions),
    }),
    [controller],
  );
  return { ...snapshot, ...actions };
}
