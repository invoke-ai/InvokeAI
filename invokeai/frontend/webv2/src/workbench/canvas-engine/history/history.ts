/**
 * Engine-owned canvas history: a bounded undo/redo stack of opaque entries.
 *
 * Project-level undo (`pushUndo` in the reducer) deliberately no longer covers
 * the canvas (Phase 0) — pixels never cross the reducer boundary, so the reducer
 * can't cheaply snapshot them. Instead the canvas keeps its OWN history here,
 * living entirely inside the engine. Each {@link HistoryEntry} is a self-contained
 * undo/redo pair (a paint {@link createImagePatchEntry | pixel patch} or a
 * structural {@link createDocumentPatchEntry | document patch}) plus a byte cost
 * used to bound memory.
 *
 * Budgets (evict oldest beyond EITHER dimension):
 * - at most {@link HISTORY_MAX_ENTRIES} undo entries, and
 * - at most {@link HISTORY_BYTE_BUDGET} bytes across both stacks.
 *
 * Re-entrancy: while an entry's `undo`/`redo` runs, {@link History.isApplying}
 * is `true`. The engine's paint/apply paths check it and skip recording, and
 * `push` is a hard no-op during apply — so replaying an entry can never spawn a
 * new entry (which would corrupt the stacks).
 *
 * Zero React, zero DOM, zero import-time side effects.
 */

/** Max number of undo entries retained before the oldest is evicted. */
export const HISTORY_MAX_ENTRIES = 64;

/** Max total bytes retained across the undo + redo stacks before the oldest is evicted (256 MB). */
export const HISTORY_BYTE_BUDGET = 256 * 1024 * 1024;

/** One reversible step. `bytes` is the (approximate) memory the entry pins, for the budget. */
export interface HistoryEntry {
  /** Human-readable label (e.g. "Brush stroke"). */
  readonly label: string;
  /** Approximate retained size in bytes (e.g. before+after ImageData byteLength). */
  readonly bytes: number;
  /**
   * Opts into failure-atomic replay. When true, History moves this entry only
   * after `undo`/`redo` returns successfully, so a preparation failure remains
   * exactly retryable. The callback must leave its domain unchanged on throw.
   *
   * Legacy entries default to move-before-replay semantics: a throw is treated
   * as post-application observer failure and the entry stays on its destination
   * stack, preventing a retry from applying the same mutation twice.
   */
  readonly replayFailureAtomic?: boolean;
  /** Reverts the change. Must not push new history entries. */
  undo(): void;
  /** Re-applies the change. Must not push new history entries. */
  redo(): void;
}

/** Options for {@link createHistory}. */
export interface CreateHistoryOptions {
  /** Undo-entry cap (default {@link HISTORY_MAX_ENTRIES}). */
  maxEntries?: number;
  /** Total-byte cap across both stacks (default {@link HISTORY_BYTE_BUDGET}). */
  byteBudget?: number;
}

/** The imperative history handle. */
export interface History {
  /** Records a new entry (clearing the redo stack) and enforces the budgets. No-op while applying. */
  push(entry: HistoryEntry): void;
  /**
   * Replaces the most recent undo entry in place (adjusting the byte total),
   * used to coalesce a rapid burst of same-target edits — e.g. arrow-key nudges —
   * into a single reversible step. Falls back to {@link push} when the undo stack
   * is empty. Clears the redo stack like `push`. No-op while applying.
   */
  amendLast(entry: HistoryEntry): void;
  /** Reverts the most recent entry (moving it onto the redo stack). No-op when empty or already applying. */
  undo(): void;
  /** Re-applies the most recently undone entry (moving it back onto the undo stack). No-op when empty or applying. */
  redo(): void;
  /** True when there is at least one entry that can be undone. */
  canUndo(): boolean;
  /** True when there is at least one entry that can be redone. */
  canRedo(): boolean;
  /** True while an entry's `undo`/`redo` is executing (re-entrancy guard). */
  isApplying(): boolean;
  /** Drops both stacks (document replace / project switch / snapshot restore). */
  clear(): void;
  /** Subscribes to any change in `canUndo`/`canRedo`. Returns an unsubscribe function. */
  subscribe(listener: () => void): () => void;
}

/** Creates a bounded history stack. */
export const createHistory = (opts: CreateHistoryOptions = {}): History => {
  const maxEntries = Math.max(1, opts.maxEntries ?? HISTORY_MAX_ENTRIES);
  const byteBudget = Math.max(0, opts.byteBudget ?? HISTORY_BYTE_BUDGET);

  const undoStack: HistoryEntry[] = [];
  const redoStack: HistoryEntry[] = [];
  const listeners = new Set<() => void>();

  // Running byte totals, kept in sync with the stacks so eviction is O(1) per drop.
  let undoBytes = 0;
  let redoBytes = 0;
  // Re-entrancy flag: true while replaying an entry.
  let applying = false;

  const notify = (): void => {
    for (const listener of listeners) {
      try {
        listener();
      } catch {
        // Stack mutation is already complete. One faulty observer must neither
        // report a false operation failure nor block later subscribers.
      }
    }
  };

  const clearRedo = (): void => {
    if (redoStack.length === 0) {
      return;
    }
    redoStack.length = 0;
    redoBytes = 0;
  };

  /** Evicts the oldest undo entries until BOTH budgets are satisfied. */
  const enforceBudgets = (): void => {
    while (undoStack.length > maxEntries && undoStack.length > 0) {
      const evicted = undoStack.shift();
      if (evicted) {
        undoBytes -= evicted.bytes;
      }
    }
    while (undoBytes + redoBytes > byteBudget && undoStack.length > 0) {
      const evicted = undoStack.shift();
      if (evicted) {
        undoBytes -= evicted.bytes;
      }
    }
  };

  const push = (entry: HistoryEntry): void => {
    // Replaying an entry must never record a new one; drop it defensively.
    if (applying) {
      return;
    }
    clearRedo();
    undoStack.push(entry);
    undoBytes += entry.bytes;
    enforceBudgets();
    // A push always clears redo and grows undo, so both booleans may have moved
    // (canUndo→true on the first push, canRedo→false when redo was non-empty).
    notify();
  };

  const amendLast = (entry: HistoryEntry): void => {
    if (applying) {
      return;
    }
    if (undoStack.length === 0) {
      push(entry);
      return;
    }
    clearRedo();
    const replaced = undoStack.pop();
    if (replaced) {
      undoBytes -= replaced.bytes;
    }
    undoStack.push(entry);
    undoBytes += entry.bytes;
    enforceBudgets();
    notify();
  };

  const undo = (): void => {
    if (applying || undoStack.length === 0) {
      return;
    }
    const entry = undoStack.at(-1);
    if (!entry) {
      return;
    }
    if (!entry.replayFailureAtomic) {
      // Legacy callbacks may mutate their domain before an observer throws.
      // Move first so a retry cannot apply that mutation twice. If replay
      // clears history, `clear()` owns the notification and the reset wins.
      undoStack.pop();
      undoBytes -= entry.bytes;
      redoStack.push(entry);
      redoBytes += entry.bytes;
      let replayError: unknown;
      let didThrow = false;
      applying = true;
      try {
        entry.undo();
      } catch (error) {
        replayError = error;
        didThrow = true;
      } finally {
        applying = false;
      }
      if (redoStack.at(-1) === entry) {
        notify();
      }
      if (didThrow) {
        // Preserve the callback's exact exception value for legacy callers.
        // eslint-disable-next-line no-throw-literal
        throw replayError;
      }
      return;
    }
    applying = true;
    try {
      entry.undo();
    } finally {
      applying = false;
    }
    // A replay callback may intentionally clear history (for example a
    // synchronous document replacement). Let that reset win; never resurrect
    // the entry onto the opposite stack after its original stack changed.
    if (undoStack.at(-1) !== entry) {
      return;
    }
    // Move the entry only after replay succeeds. A fallible callback (for
    // example detached raster-cache preparation) may throw before applying
    // anything; keeping the stacks and byte totals untouched makes retry exact.
    undoStack.pop();
    undoBytes -= entry.bytes;
    redoStack.push(entry);
    redoBytes += entry.bytes;
    notify();
  };

  const redo = (): void => {
    if (applying || redoStack.length === 0) {
      return;
    }
    const entry = redoStack.at(-1);
    if (!entry) {
      return;
    }
    if (!entry.replayFailureAtomic) {
      redoStack.pop();
      redoBytes -= entry.bytes;
      undoStack.push(entry);
      undoBytes += entry.bytes;
      let replayError: unknown;
      let didThrow = false;
      applying = true;
      try {
        entry.redo();
      } catch (error) {
        replayError = error;
        didThrow = true;
      } finally {
        applying = false;
      }
      if (undoStack.at(-1) === entry) {
        notify();
      }
      if (didThrow) {
        // Preserve the callback's exact exception value for legacy callers.
        // eslint-disable-next-line no-throw-literal
        throw replayError;
      }
      return;
    }
    applying = true;
    try {
      entry.redo();
    } finally {
      applying = false;
    }
    if (redoStack.at(-1) !== entry) {
      return;
    }
    redoStack.pop();
    redoBytes -= entry.bytes;
    undoStack.push(entry);
    undoBytes += entry.bytes;
    notify();
  };

  const clear = (): void => {
    if (undoStack.length === 0 && redoStack.length === 0) {
      return;
    }
    undoStack.length = 0;
    redoStack.length = 0;
    undoBytes = 0;
    redoBytes = 0;
    notify();
  };

  return {
    amendLast,
    canRedo: () => redoStack.length > 0,
    canUndo: () => undoStack.length > 0,
    clear,
    isApplying: () => applying,
    push,
    redo,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    undo,
  };
};
