import type { ScannerController, ScannerOptions } from '@scanoficus/core/controller';
import type { ScanFilter } from '@scanoficus/core/filters';
import type { Quad } from '@scanoficus/core/geometry';
import type { PageSource } from '@scanoficus/core/render';
import {
  defineScanElements,
  type ScanCameraElement,
  type ScanCompleteDetail,
  type ScanCropDetail,
  type ScanCropEditorElement,
  type ScanDocumentElement,
  type ScanErrorDetail,
  type ScanExportImagesDetail,
  type ScanReviewElement,
} from '@scanoficus/elements';
import {
  type CSSProperties,
  type JSX,
  type RefObject,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';

/** Runs layout effects on the client only (avoids the SSR warning of `useLayoutEffect`). */
const useClientLayoutEffect = typeof document === 'undefined' ? useEffect : useLayoutEffect;

type Handlers = Record<string, ((event: Event) => void) | undefined>;

interface HostProps {
  className?: string;
  style?: CSSProperties;
}

/**
 * Mounts a scanoficus custom element imperatively inside a `<div>`: the element is created
 * with `document.createElement` on the client, its properties and attributes are applied
 * before it is connected (so its initial configuration is honored and it never creates a
 * controller of its own when one is passed), and attributes, properties and event listeners
 * are kept in sync with later prop changes. This avoids the differences in how React 18
 * and 19 pass props to custom elements and keeps server rendering to an empty `<div>`.
 */
function useScanElement<T extends HTMLElement>(
  tag: string,
  properties: Partial<T>,
  attributes: Record<string, string | null>,
  handlers: Handlers,
): { container: RefObject<HTMLDivElement | null>; element: T | null } {
  const container = useRef<HTMLDivElement | null>(null);
  const [element, setElement] = useState<T | null>(null);
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;
  const propertiesRef = useRef(properties);
  propertiesRef.current = properties;
  const attributesRef = useRef(attributes);
  attributesRef.current = attributes;

  useClientLayoutEffect(() => {
    defineScanElements();
    const el = document.createElement(tag) as T;
    el.style.height = '100%';
    // Properties and attributes are applied before connecting: the element reads its
    // configuration (and may create its controller) in connectedCallback.
    Object.assign(el, propertiesRef.current);
    for (const [name, value] of Object.entries(attributesRef.current)) {
      if (value !== null) el.setAttribute(name, value);
    }
    const listeners = Object.keys(handlersRef.current).map((type) => {
      const listener = (event: Event) => handlersRef.current[type]?.(event);
      el.addEventListener(type, listener);
      return [type, listener] as const;
    });
    container.current?.append(el);
    setElement(el);
    return () => {
      for (const [type, listener] of listeners) el.removeEventListener(type, listener);
      el.remove();
      setElement(null);
    };
  }, [tag]);

  useClientLayoutEffect(() => {
    if (!element) return;
    for (const [key, value] of Object.entries(properties)) {
      if ((element as unknown as Record<string, unknown>)[key] !== value) {
        (element as unknown as Record<string, unknown>)[key] = value;
      }
    }
  });

  useClientLayoutEffect(() => {
    if (!element) return;
    for (const [name, value] of Object.entries(attributes)) {
      if (value === null) element.removeAttribute(name);
      else if (element.getAttribute(name) !== value) element.setAttribute(name, value);
    }
  });

  return { container, element };
}

/** Removes `undefined` entries so that unset props do not overwrite element defaults. */
function defined<T extends object>(value: T): Partial<T> {
  return Object.fromEntries(Object.entries(value).filter(([, v]) => v !== undefined)) as Partial<T>;
}

function detail<T>(handler: ((detail: T) => void) | undefined): ((e: Event) => void) | undefined {
  return handler && ((e: Event) => handler((e as CustomEvent<T>).detail));
}

/** Props of {@link DocumentScanner}. */
export interface DocumentScannerProps extends HostProps {
  /** Controller to drive (e.g. from `useDocumentScanner`); otherwise the element creates one. */
  controller?: ScannerController;
  /** Options for the element-created controller (read once, on mount). */
  options?: ScannerOptions;
  /** Start with auto-capture on (default) or off. */
  autoCapture?: boolean;
  /** Filter of new pages. */
  filter?: ScanFilter;
  /** PDF page size. */
  pageSize?: 'image' | 'a4' | 'letter';
  onComplete?: (detail: ScanCompleteDetail) => void;
  onExportImages?: (detail: ScanExportImagesDetail) => void;
  onCancel?: () => void;
  onError?: (detail: ScanErrorDetail) => void;
}

/** The complete scanning flow (`<scan-document>`): camera, auto-capture, review, crop, PDF. */
export function DocumentScanner(props: DocumentScannerProps): JSX.Element {
  const { container } = useScanElement<ScanDocumentElement>(
    'scan-document',
    defined({ controller: props.controller, options: props.options }),
    {
      'auto-capture': props.autoCapture === false ? 'false' : null,
      filter: props.filter ?? null,
      'page-size': props.pageSize ?? null,
    },
    {
      'scan-complete': detail(props.onComplete),
      'scan-export-images': detail(props.onExportImages),
      'scan-cancel': props.onCancel && (() => props.onCancel?.()),
      'scan-error': detail(props.onError),
    },
  );
  return <div ref={container} className={props.className} style={props.style} />;
}

/** Props of {@link ScanCamera}. */
export interface ScanCameraProps extends HostProps {
  controller: ScannerController;
  onDone?: () => void;
  onCancel?: () => void;
}

/** Live camera view (`<scan-camera>`) for a controller; starts the camera while mounted. */
export function ScanCamera(props: ScanCameraProps): JSX.Element {
  const { container } = useScanElement<ScanCameraElement>(
    'scan-camera',
    { controller: props.controller },
    {},
    {
      'scan-done': props.onDone && (() => props.onDone?.()),
      'scan-cancel': props.onCancel && (() => props.onCancel?.()),
    },
  );
  return <div ref={container} className={props.className} style={props.style} />;
}

/** Props of {@link ScanReview}. */
export interface ScanReviewProps extends HostProps {
  controller: ScannerController;
  pageSize?: 'image' | 'a4' | 'letter';
  /** Hide "Retake" and "Scan more" (pages come from imported photos). */
  noCamera?: boolean;
  onComplete?: (detail: ScanCompleteDetail) => void;
  onCrop?: (detail: ScanCropDetail) => void;
  onExportImages?: (detail: ScanExportImagesDetail) => void;
  onError?: (detail: ScanErrorDetail) => void;
}

/** Page review (`<scan-review>`): filters, rotate, delete, retake, export. */
export function ScanReview(props: ScanReviewProps): JSX.Element {
  const { container } = useScanElement<ScanReviewElement>(
    'scan-review',
    { controller: props.controller },
    { 'page-size': props.pageSize ?? null, 'no-camera': props.noCamera ? '' : null },
    {
      'scan-complete': detail(props.onComplete),
      'scan-crop': detail(props.onCrop),
      'scan-export-images': detail(props.onExportImages),
      'scan-error': detail(props.onError),
    },
  );
  return <div ref={container} className={props.className} style={props.style} />;
}

/** Props of {@link CropEditor}. */
export interface CropEditorProps extends HostProps {
  source: PageSource;
  quad: Quad;
  onChange?: (quad: Quad) => void;
  onConfirm?: (quad: Quad) => void;
  onCancel?: () => void;
}

/** Corner editor (`<scan-crop-editor>`) with loupe and keyboard support. */
export function CropEditor(props: CropEditorProps): JSX.Element {
  const { container } = useScanElement<ScanCropEditorElement>(
    'scan-crop-editor',
    // `source` must be assigned before `quad` (setting the source resets "Reset").
    { source: props.source, quad: props.quad },
    {},
    {
      'quad-change': detail<{ quad: Quad }>(props.onChange && ((d) => props.onChange?.(d.quad))),
      'crop-confirm': detail<{ quad: Quad }>(props.onConfirm && ((d) => props.onConfirm?.(d.quad))),
      'crop-cancel': props.onCancel && (() => props.onCancel?.()),
    },
  );
  return <div ref={container} className={props.className} style={props.style} />;
}
