/** Minimal typed event emitter. `Events` maps event names to their payload types. */
export class Emitter<Events extends Record<string, unknown>> {
  private readonly listeners = new Map<keyof Events, Set<(payload: never) => void>>();

  /** Subscribes to `type`; returns an unsubscribe function. */
  on<K extends keyof Events>(type: K, listener: (payload: Events[K]) => void): () => void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(listener as (payload: never) => void);
    return () => {
      set.delete(listener as (payload: never) => void);
    };
  }

  /** True if anyone listens to `type` (lets emitters skip building expensive payloads). */
  has(type: keyof Events): boolean {
    return (this.listeners.get(type)?.size ?? 0) > 0;
  }

  emit<K extends keyof Events>(type: K, payload: Events[K]): void {
    const set = this.listeners.get(type);
    if (!set) return;
    for (const listener of [...set]) (listener as (payload: Events[K]) => void)(payload);
  }

  clear(): void {
    this.listeners.clear();
  }
}
