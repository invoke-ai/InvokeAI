import type { BitmapStore } from '@workbench/canvas-engine/document/bitmapStore';

export class PersistenceController {
  readonly store: BitmapStore;
  private disposed = false;

  constructor(store: BitmapStore) {
    this.store = store;
  }

  flush(): Promise<void> {
    return this.store.flushPendingUploads();
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.store.dispose();
  }
}
