/**
 * Engine-owned undo/redo stacks for pixel, selection and structural edits. An edit is admitted before it mutates
 * anything: admission reserves byte capacity (without clearing redo or evicting older steps), so an edit that could
 * never keep its undo entry is refused up front. Publication consumes the admission, clears redo and evicts the
 * oldest entries beyond {@link HISTORY_MAX_ENTRIES} or the byte budget. Replay is asynchronous and failure-atomic:
 * an entry moves to the opposite stack only after its callback completes.
 */

import { collectRestorableAssetRefs } from '@workbench/mediaReferences';

/** Max number of undo entries retained before the oldest is evicted. */
export const HISTORY_MAX_ENTRIES = 64;

/** Max total bytes retained across the undo + redo stacks before the oldest is evicted (256 MB). */
export const HISTORY_BYTE_BUDGET = 256 * 1024 * 1024;

/** Fixed bytes every undo entry is admitted for beyond the pixels or actions it retains. */
export const HISTORY_ENTRY_OVERHEAD_BYTES = 256;

export interface HeldAssetRefs {
  readonly images: readonly string[];
  readonly videos: readonly string[];
}

/** For entries that restore pixels or selection state only. */
export const NO_HELD_ASSET_REFS: HeldAssetRefs = { images: [], videos: [] };

/** Media names captured by an undo entry, including sources no longer in the live document. */
export const collectHistoryMediaRefs = (...values: unknown[]): HeldAssetRefs => {
  const { images, videos } = collectRestorableAssetRefs(...values);
  return { images: [...images], videos: [...videos] };
};

/** One reversible step. `bytes` is the memory the entry retains. */
export interface HistoryEntry {
  /** Human-readable label (e.g. "Brush stroke"). */
  readonly label: string;
  /** Retained size in bytes; publication requires it to fit the admission. */
  readonly bytes: number;
  /**
   * Media names this entry can restore after they leave the current document; cleanup keeps them while the entry is
   * on either stack.
   */
  readonly heldAssetRefs: HeldAssetRefs;
  /** Releases resources retained only by this entry when it is permanently dropped. */
  readonly dispose?: () => void;
  /** Reverts the change. A throw or rejection must leave the domain unchanged; the entry then stays in place. */
  undo(): void | Promise<void>;
  /** Re-applies the change, with the same failure contract as `undo`. */
  redo(): void | Promise<void>;
}

/** Identifies one published entry; equality is the only operation. */
export interface HistoryEntryToken {
  readonly __historyEntryToken: unique symbol;
}

/** Byte capacity reserved for an edit that has not published its entry yet. */
export interface HistoryAdmission {
  readonly bytes: number;
  /** Reserves `bytes` more; false (and unchanged) when the larger entry could never be retained. */
  grow(bytes: number): boolean;
  /**
   * Records `entry`, which must fit the admission, and consumes it. With `replacing`, the entry takes the place of
   * that top entry instead (coalescing); callers check {@link History.top} first.
   */
  publish(entry: HistoryEntry, replacing?: HistoryEntryToken): HistoryEntryToken;
  /** Returns unpublished capacity. Idempotent; a no-op after `publish`. */
  release(): void;
}

export type HistoryReplayResult =
  | { status: 'applied' | 'empty' | 'busy' }
  | { status: 'failed'; label: string; error: unknown };

/** Options for {@link createHistory}. */
export interface CreateHistoryOptions {
  /** Undo-entry cap (default {@link HISTORY_MAX_ENTRIES}). */
  maxEntries?: number;
  /** Total-byte cap across both stacks (default {@link HISTORY_BYTE_BUDGET}). */
  byteBudget?: number;
}

/** The imperative history handle. */
export interface History {
  /** Reserves capacity for an edit's entry, or null when an entry that size could never be retained. */
  admit(bytes: number): HistoryAdmission | null;
  /** The newest undo entry, or null. */
  top(): HistoryEntryToken | null;
  /** Reverts the newest entry. Resolves `busy` while another replay runs. */
  undo(): Promise<HistoryReplayResult>;
  /** Re-applies the most recently undone entry. */
  redo(): Promise<HistoryReplayResult>;
  canUndo(): boolean;
  canRedo(): boolean;
  /** True while an entry's replay is running; edits are refused meanwhile. */
  isReplaying(): boolean;
  /** Drops both stacks (document replace / project switch / snapshot restore). */
  clear(): void;
  /** Current retained bytes across undo and redo stacks. */
  byteSize(): number;
  /** Evicts oldest entries until retained bytes are at or below `budgetBytes`. */
  trimToBytes(budgetBytes: number): void;
  /**
   * Labels of every retained step: `past` oldest-first (its last element is
   * what `undo()` reverts), `future` next-redo-first. Fresh arrays per call.
   */
  entries(): { past: readonly string[]; future: readonly string[] };
  /** Union of media references retained by undo and redo entries. */
  heldAssetRefs(): HeldAssetRefs;
  /** Subscribes to every stack mutation (publish, undo, redo, clear, eviction). Returns an unsubscribe function. */
  subscribe(listener: () => void): () => void;
}

interface StackRecord {
  readonly entry: HistoryEntry;
  readonly token: HistoryEntryToken;
}

const normalizedBytes = (bytes: number): number => (Number.isFinite(bytes) ? Math.max(0, Math.ceil(bytes)) : Infinity);

/** Creates a bounded history stack. */
export const createHistory = (opts: CreateHistoryOptions = {}): History => {
  const maxEntries = Math.max(1, opts.maxEntries ?? HISTORY_MAX_ENTRIES);
  const byteBudget = Math.max(0, opts.byteBudget ?? HISTORY_BYTE_BUDGET);

  const undoStack: StackRecord[] = [];
  const redoStack: StackRecord[] = [];
  const listeners = new Set<() => void>();
  let undoBytes = 0;
  let redoBytes = 0;
  let admittedBytes = 0;
  let replaying = false;

  const disposeEntry = (entry: HistoryEntry): void => {
    try {
      entry.dispose?.();
    } catch {
      // Stack ownership has already ended. Resource cleanup cannot restore the
      // entry and must not prevent the remaining history from being released.
    }
  };

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
    const discarded = redoStack.splice(0);
    discarded.forEach(({ entry }) => disposeEntry(entry));
    redoBytes = 0;
  };

  const evictOldest = (): void => {
    while (undoStack.length > maxEntries || (undoBytes + redoBytes > byteBudget && undoStack.length > 1)) {
      const evicted = undoStack.shift()!;
      undoBytes -= evicted.entry.bytes;
      disposeEntry(evicted.entry);
    }
  };

  const admit = (requestedBytes: number): HistoryAdmission | null => {
    let bytes = normalizedBytes(requestedBytes);
    if (admittedBytes + bytes > byteBudget) {
      return null;
    }
    admittedBytes += bytes;
    let state: 'open' | 'published' | 'released' = 'open';
    const close = (): void => {
      admittedBytes -= bytes;
    };
    return {
      get bytes() {
        return bytes;
      },
      grow: (additional) => {
        const extra = normalizedBytes(additional);
        if (state !== 'open' || admittedBytes + extra > byteBudget) {
          return false;
        }
        admittedBytes += extra;
        bytes += extra;
        return true;
      },
      publish: (entry, replacing) => {
        if (state !== 'open') {
          throw new Error('History admission was already consumed.');
        }
        if (entry.bytes > bytes) {
          throw new Error(`History entry "${entry.label}" exceeds its admission (${entry.bytes} > ${bytes} bytes).`);
        }
        if (replacing !== undefined && undoStack.at(-1)?.token !== replacing) {
          throw new Error('History entry to replace is no longer the newest step.');
        }
        state = 'published';
        close();
        clearRedo();
        if (replacing !== undefined) {
          const replaced = undoStack.pop()!;
          undoBytes -= replaced.entry.bytes;
          disposeEntry(replaced.entry);
        }
        const token = Object.freeze({}) as HistoryEntryToken;
        undoStack.push({ entry, token });
        undoBytes += entry.bytes;
        evictOldest();
        notify();
        return token;
      },
      release: () => {
        if (state === 'open') {
          state = 'released';
          close();
        }
      },
    };
  };

  /** Replays the top of `from`; moves it to `to` only when its callback completes and nothing reset the stacks. */
  const replay = async (direction: 'undo' | 'redo'): Promise<HistoryReplayResult> => {
    const from = direction === 'undo' ? undoStack : redoStack;
    if (replaying) {
      return { status: 'busy' };
    }
    const record = from.at(-1);
    if (!record) {
      return { status: 'empty' };
    }
    replaying = true;
    notify();
    try {
      await (direction === 'undo' ? record.entry.undo() : record.entry.redo());
    } catch (error) {
      return { error, label: record.entry.label, status: 'failed' };
    } finally {
      replaying = false;
    }
    // A replay may itself clear history (a document replacement); never resurrect the entry onto the other stack.
    if (from.at(-1) === record) {
      from.pop();
      if (direction === 'undo') {
        undoBytes -= record.entry.bytes;
        redoStack.push(record);
        redoBytes += record.entry.bytes;
      } else {
        redoBytes -= record.entry.bytes;
        undoStack.push(record);
        undoBytes += record.entry.bytes;
      }
    }
    notify();
    return { status: 'applied' };
  };

  const clear = (): void => {
    if (undoStack.length === 0 && redoStack.length === 0) {
      return;
    }
    const discarded = [...undoStack, ...redoStack];
    undoStack.length = 0;
    redoStack.length = 0;
    discarded.forEach(({ entry }) => disposeEntry(entry));
    undoBytes = 0;
    redoBytes = 0;
    notify();
  };

  const trimToBytes = (budgetBytes: number): void => {
    const budget = Math.max(0, budgetBytes);
    let changed = false;
    while (undoBytes + redoBytes > budget && undoStack.length > 0) {
      const evicted = undoStack.shift()!;
      undoBytes -= evicted.entry.bytes;
      disposeEntry(evicted.entry);
      changed = true;
    }
    while (undoBytes + redoBytes > budget && redoStack.length > 0) {
      const evicted = redoStack.shift()!;
      redoBytes -= evicted.entry.bytes;
      disposeEntry(evicted.entry);
      changed = true;
    }
    if (changed) {
      notify();
    }
  };

  // Reused while the stacks hold the same entries, so an unchanged union keeps its identity for the hold lease.
  let heldUnion: { parts: HeldAssetRefs[]; refs: HeldAssetRefs } | null = null;
  const heldAssetRefs = (): HeldAssetRefs => {
    const parts = [...undoStack, ...redoStack].map(({ entry }) => entry.heldAssetRefs);
    if (
      heldUnion &&
      heldUnion.parts.length === parts.length &&
      parts.every((part, i) => part === heldUnion!.parts[i])
    ) {
      return heldUnion.refs;
    }
    const images = new Set<string>();
    const videos = new Set<string>();
    for (const part of parts) {
      part.images.forEach((name) => images.add(name));
      part.videos.forEach((name) => videos.add(name));
    }
    heldUnion = { parts, refs: { images: [...images], videos: [...videos] } };
    return heldUnion.refs;
  };

  return {
    admit,
    byteSize: () => undoBytes + redoBytes,
    canRedo: () => redoStack.length > 0,
    canUndo: () => undoStack.length > 0,
    clear,
    entries: () => ({
      future: redoStack.map(({ entry }) => entry.label).reverse(),
      past: undoStack.map(({ entry }) => entry.label),
    }),
    heldAssetRefs,
    isReplaying: () => replaying,
    redo: () => replay('redo'),
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    top: () => undoStack.at(-1)?.token ?? null,
    trimToBytes,
    undo: () => replay('undo'),
  };
};
