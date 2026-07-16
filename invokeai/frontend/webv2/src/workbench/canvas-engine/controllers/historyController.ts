import { createHistory, HISTORY_BYTE_BUDGET, type History } from '@workbench/canvas-engine/history/history';

export const INACTIVE_HISTORY_BYTE_BUDGET = 64 * 1024 * 1024;

export interface HistoryControllerOptions {
  readonly activeByteBudget?: number;
  readonly inactiveByteBudget?: number;
  readonly canEdit?: () => boolean;
  readonly isGestureActive?: () => boolean;
  readonly endBurst?: () => void;
  readonly canUndoStore?: { set(value: boolean): void };
  readonly canRedoStore?: { set(value: boolean): void };
}

export class HistoryController {
  readonly history: History;
  private readonly inactiveByteBudget: number;
  private readonly canEdit: () => boolean;
  private readonly isGestureActive: () => boolean;
  private readonly endBurst: () => void;
  private readonly unsubscribe: () => void;
  private disposed = false;

  constructor(options: HistoryControllerOptions = {}) {
    this.inactiveByteBudget = options.inactiveByteBudget ?? INACTIVE_HISTORY_BYTE_BUDGET;
    this.history = createHistory({ byteBudget: options.activeByteBudget ?? HISTORY_BYTE_BUDGET });
    this.canEdit = options.canEdit ?? (() => true);
    this.isGestureActive = options.isGestureActive ?? (() => false);
    this.endBurst = options.endBurst ?? (() => undefined);
    const syncStores = (): void => {
      try {
        options.canUndoStore?.set(this.history.canUndo());
      } catch {
        // Store observers are ancillary and must not break history transactions.
      }
      try {
        options.canRedoStore?.set(this.history.canRedo());
      } catch {
        // Keep the two notifications isolated from one another.
      }
    };
    this.unsubscribe = this.history.subscribe(syncStores);
    syncStores();
  }

  undo(): void {
    if (this.disposed || !this.canEdit() || this.isGestureActive()) {
      return;
    }
    this.endBurst();
    this.history.undo();
  }

  redo(): void {
    if (this.disposed || !this.canEdit() || this.isGestureActive()) {
      return;
    }
    this.endBurst();
    this.history.redo();
  }

  clear(): void {
    if (this.disposed || !this.canEdit()) {
      return;
    }
    this.endBurst();
    this.history.clear();
  }

  cooldown(): void {
    if (!this.disposed) {
      this.history.trimToBytes(this.inactiveByteBudget);
    }
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.unsubscribe();
    this.history.clear();
  }
}
