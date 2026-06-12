import { useSyncExternalStore } from 'react';

/**
 * Ephemeral live-progress store for in-flight queue items, keyed by the local
 * queue item id. Progress is high-frequency transient data, so it deliberately
 * lives outside the workbench reducer: routing it through dispatch would
 * re-render every consumer of workbench state and churn autosave on each step
 * of every generation. Any widget can subscribe to a single item via
 * `useQueueItemProgress` and only re-renders when that item's progress moves.
 */

export interface QueueItemProgress {
  /** 1-based image slot currently executing inside this local batch. */
  activeItemIndex?: number;
  completedItemCount?: number;
  message: string;
  /** 0..1, or null while indeterminate. */
  percentage: number | null;
  totalItemCount?: number;
}

export interface QueueItemProgressSink {
  clearAll?(): void;
  set(queueItemId: string, progress: QueueItemProgress): void;
  clear(queueItemId: string): void;
}

const progressByQueueItemId = new Map<string, QueueItemProgress>();
const listeners = new Set<() => void>();

const emit = (): void => {
  for (const listener of listeners) {
    listener();
  }
};

const subscribe = (listener: () => void): (() => void) => {
  listeners.add(listener);

  return () => {
    listeners.delete(listener);
  };
};

export const queueItemProgressStore: QueueItemProgressSink = {
  clearAll() {
    if (progressByQueueItemId.size > 0) {
      progressByQueueItemId.clear();
      emit();
    }
  },
  clear(queueItemId) {
    if (progressByQueueItemId.delete(queueItemId)) {
      emit();
    }
  },
  set(queueItemId, progress) {
    progressByQueueItemId.set(queueItemId, progress);
    emit();
  },
};

export const useQueueItemProgress = (queueItemId: string): QueueItemProgress | null =>
  useSyncExternalStore(subscribe, () => progressByQueueItemId.get(queueItemId) ?? null);
