import {
  type DetectorWorkerLike,
  type DetectorWorkerMessage,
  type DetectRequest,
  FrameProcessor,
  handleDetectRequest,
} from '../../src/worker/index.ts';

/**
 * In-process stand-in for a detector Worker: runs the real request handler asynchronously and
 * can simulate load failures, crashes, deserialization errors and silence.
 */
export class FakeWorker implements DetectorWorkerLike {
  readonly processor = new FrameProcessor();
  readonly transfers: Transferable[][] = [];
  private readonly listeners = new Map<string, ((event: never) => void)[]>();
  terminated = false;
  /** When false, requests are swallowed (a hung worker). */
  responsive = true;

  constructor(options: { ready?: boolean } = {}) {
    if (options.ready !== false) setTimeout(() => this.message({ ready: true }));
  }

  postMessage(message: DetectRequest, transfer: Transferable[]): void {
    if (this.terminated) return;
    this.transfers.push(transfer);
    const cloned = structuredClone(message, { transfer });
    if (!this.responsive) return;
    setTimeout(() => {
      if (!this.terminated) this.message(handleDetectRequest(this.processor, cloned));
    });
  }

  addEventListener(type: string, listener: (event: never) => void): void {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }

  terminate(): void {
    this.terminated = true;
  }

  message(data: DetectorWorkerMessage): void {
    this.dispatch('message', new MessageEvent('message', { data }));
  }

  /** Simulates a worker `error` event (load failure or uncaught exception). */
  error(details: { message?: string; filename?: string; lineno?: number; colno?: number } = {}) {
    this.dispatch('error', Object.assign(new Event('error'), details));
  }

  messageError(): void {
    this.dispatch('messageerror', new Event('messageerror'));
  }

  private dispatch(type: string, event: Event): void {
    for (const listener of this.listeners.get(type) ?? []) (listener as (e: Event) => void)(event);
  }
}
