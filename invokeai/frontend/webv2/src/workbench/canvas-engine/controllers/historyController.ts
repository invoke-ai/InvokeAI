import {
  createHistory,
  HISTORY_BYTE_BUDGET,
  type History,
  type HistoryReplayResult,
} from '@workbench/canvas-engine/history/history';

export const INACTIVE_HISTORY_BYTE_BUDGET = 64 * 1024 * 1024;

export interface HistoryControllerOptions {
  readonly activeByteBudget?: number;
  readonly inactiveByteBudget?: number;
  readonly canEdit?: () => boolean;
  readonly isGestureActive?: () => boolean;
  /** Puts back unrecorded live state (a floating selection, a structural preview) before a replay lands over it. */
  readonly beforeReplay?: () => void;
  readonly canUndoStore?: { set(value: boolean): void };
  readonly canRedoStore?: { set(value: boolean): void };
  /** Reports a step whose replay failed and therefore stayed where it was. */
  readonly reportFailure?: (label: string, error: unknown) => void;
}

export type HistoryCommandResult = HistoryReplayResult | { status: 'refused' };

/** Owns engine history, its guarded replay commands and the undo/redo store mirrors. */
export class HistoryController {
  readonly history: History;
  private readonly inactiveByteBudget: number;
  private readonly canEdit: () => boolean;
  private readonly isGestureActive: () => boolean;
  private readonly unsubscribe: () => void;
  private disposed = false;

  constructor(private readonly options: HistoryControllerOptions = {}) {
    this.inactiveByteBudget = options.inactiveByteBudget ?? INACTIVE_HISTORY_BYTE_BUDGET;
    this.history = createHistory({ byteBudget: options.activeByteBudget ?? HISTORY_BYTE_BUDGET });
    this.canEdit = options.canEdit ?? (() => true);
    this.isGestureActive = options.isGestureActive ?? (() => false);
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

  undo(): Promise<HistoryCommandResult> {
    return this.replay('undo');
  }

  redo(): Promise<HistoryCommandResult> {
    return this.replay('redo');
  }

  clear(): void {
    if (this.disposed || !this.canEdit()) {
      return;
    }
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

  private async replay(direction: 'undo' | 'redo'): Promise<HistoryCommandResult> {
    if (this.disposed || !this.canEdit() || this.isGestureActive()) {
      return { status: 'refused' };
    }
    // Nothing to replay disturbs nothing: live state is put back only for a step that will land.
    if (!(direction === 'undo' ? this.history.canUndo() : this.history.canRedo())) {
      return { status: 'empty' };
    }
    this.options.beforeReplay?.();
    const result = await (direction === 'undo' ? this.history.undo() : this.history.redo());
    if (result.status === 'failed') {
      try {
        this.options.reportFailure?.(result.label, result.error);
      } catch {
        // Diagnostics cannot change the replay outcome.
      }
    }
    return result;
  }
}
