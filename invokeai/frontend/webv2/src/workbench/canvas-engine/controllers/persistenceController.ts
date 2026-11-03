import type { BitmapStore, FlushPendingUploadsOptions } from '@workbench/canvas-engine/document/bitmapStore';

/**
 * How a one-layer flush ended: its pixels persisted, there was nothing to persist, they could not be, or the wait
 * aborted.
 */
export type LayerFlushResult = 'flushed' | 'clean' | 'failed' | 'aborted';

export interface LayerFlushOptions {
  readonly signal?: AbortSignal;
}

export class PersistenceController {
  readonly store: BitmapStore;
  private disposed = false;

  constructor(store: BitmapStore) {
    this.store = store;
  }

  flush(options?: FlushPendingUploadsOptions): Promise<void> {
    return this.store.flushPendingUploads(options);
  }

  /**
   * The paint barrier scoped to one layer: waits only for that layer's dirty and in-flight uploads, including while an
   * open edit holds its pixels.
   */
  async flushLayer(layerId: string, { signal }: LayerFlushOptions = {}): Promise<LayerFlushResult> {
    if (signal?.aborted) {
      return 'aborted';
    }
    if (!this.store.hasPendingWork(layerId)) {
      return 'clean';
    }
    try {
      await this.store.flushPendingUploads({ layerIds: [layerId], signal });
      return 'flushed';
    } catch (error) {
      return signal?.aborted || (error instanceof Error && error.name === 'AbortError') ? 'aborted' : 'failed';
    }
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.store.dispose();
  }
}
