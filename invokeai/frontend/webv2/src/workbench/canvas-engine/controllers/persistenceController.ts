import type { BitmapStore, FlushPendingUploadsOptions } from '@workbench/canvas-engine/document/bitmapStore';

export class PersistenceController {
  readonly store: BitmapStore;
  private disposed = false;

  constructor(store: BitmapStore) {
    this.store = store;
  }

  flush(options?: FlushPendingUploadsOptions): Promise<void> {
    return this.store.flushPendingUploads(options);
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.store.dispose();
  }
}
