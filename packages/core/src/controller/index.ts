/**
 * Headless scanner controller: camera, live detection, auto-capture, pages and export as a
 * state machine with typed events. UI layers (custom elements, React) build on it.
 *
 * @module
 */
export { openScannerCamera, type ScannerCamera } from './camera-source.ts';
export {
  type LiveDetection,
  type PageChanges,
  ScannerController,
  type ScannerEvents,
  type ScannerOptions,
  type ScannerSnapshot,
  type ScannerState,
  type ScanPage,
} from './controller.ts';
