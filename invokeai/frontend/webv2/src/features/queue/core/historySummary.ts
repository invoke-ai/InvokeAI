import type { QueueItem } from '@features/queue/core/historyTypes';
import type { QueueItemProgress } from '@features/queue/core/types';

import { getQueueItemSnapshotBatchCount } from '@features/queue/core/historySnapshot';

export interface QueueSummary {
  current: number;
  remaining: number;
  runningQueueItemId: string | null;
  total: number;
}

export interface ProjectQueueIndicatorState {
  hasOpenQueueWork: boolean;
  progressState: QueueProgressBarState;
  runningQueueItemId: string | null;
}

export type QueueProgressBarState =
  | { kind: 'determinate'; value: number }
  | { kind: 'idle'; value: 0 }
  | { kind: 'indeterminate'; value: null };

export const getQueueProgressBarState = ({
  isConnected,
  isRunning,
  loadingModelsCount,
  progress,
}: {
  isConnected: boolean;
  isRunning: boolean;
  loadingModelsCount: number;
  progress: QueueItemProgress | null;
}): QueueProgressBarState => {
  if (!isConnected) {
    return { kind: 'idle', value: 0 };
  }

  if (loadingModelsCount > 0) {
    return { kind: 'indeterminate', value: null };
  }

  if (isRunning && (!progress || progress.percentage === null || progress.percentage === 0)) {
    return { kind: 'indeterminate', value: null };
  }

  if (isRunning) {
    return { kind: 'determinate', value: progress?.percentage ?? 0 };
  }

  return { kind: 'idle', value: 0 };
};

export const getQueueProgressBarValue = (params: Parameters<typeof getQueueProgressBarState>[0]): number | null =>
  getQueueProgressBarState(params).value;

export const isOpenQueueItem = (item: QueueItem): boolean => item.status === 'pending' || item.status === 'running';

export const getQueueItemExpectedImageCount = (item: QueueItem): number => {
  if (item.backendItemIds?.length) {
    return item.backendItemIds.length;
  }

  return getQueueItemSnapshotBatchCount(item);
};

const compareQueuePosition = (a: QueueItem, b: QueueItem): number => {
  const aTime = Date.parse(a.snapshot.submittedAt);
  const bTime = Date.parse(b.snapshot.submittedAt);

  return (Number.isFinite(aTime) ? aTime : 0) - (Number.isFinite(bTime) ? bTime : 0);
};

export const getFinishedImageCount = (item: QueueItem, progress: QueueItemProgress | null = null): number => {
  const settledCount = new Set([...(item.completedBackendItemIds ?? []), ...(item.cancelledBackendItemIds ?? [])]).size;
  return Math.min(getQueueItemExpectedImageCount(item), Math.max(progress?.completedItemCount ?? 0, settledCount));
};

const getRunningImageOrdinal = (item: QueueItem, progress: QueueItemProgress | null, isConcurrent: boolean): number => {
  const expectedCount = getQueueItemExpectedImageCount(item);
  const settledOrdinal = getFinishedImageCount(item, progress) + 1;
  // Only use the settled count in place of the executing ordinal after observing concurrency.
  const activeIndex = isConcurrent ? settledOrdinal : (progress?.activeItemIndex ?? settledOrdinal);
  return Math.min(expectedCount, Math.max(1, activeIndex));
};

export const getQueueSummary = (
  items: QueueItem[],
  progress: QueueItemProgress | null = null,
  isConcurrent = false
): QueueSummary => {
  const openItems = items.filter(isOpenQueueItem).sort(compareQueuePosition);
  const total = openItems.reduce((sum, item) => sum + getQueueItemExpectedImageCount(item), 0);

  if (total === 0) {
    return { current: 0, remaining: 0, runningQueueItemId: null, total: 0 };
  }

  const runningItem = openItems.find((item) => item.status === 'running');
  // Progress belongs to the selected running batch only. Other open batches
  // contribute their own settled slots; queued work must still be counted.
  const remaining = openItems.reduce((sum, item) => {
    const unsettled =
      getQueueItemExpectedImageCount(item) - getFinishedImageCount(item, item.id === runningItem?.id ? progress : null);
    // Backend completion precedes result delivery. A running local item is still open until delivery closes it.
    return sum + Math.max(item.status === 'running' ? 1 : 0, unsettled);
  }, 0);

  if (!runningItem) {
    return { current: 0, remaining, runningQueueItemId: null, total };
  }

  let current = 0;

  for (const item of openItems) {
    if (item.id === runningItem.id) {
      current += getRunningImageOrdinal(item, progress, isConcurrent);
      break;
    }

    current += getQueueItemExpectedImageCount(item);
  }

  current = Math.min(current, total);

  // The active ordinal and the all-batch countdown describe different things.
  return { current, remaining, runningQueueItemId: runningItem.id, total };
};

export const getProjectQueueIndicatorState = ({
  isConnected,
  loadingModelsCount,
  progress,
  queueItems,
}: {
  isConnected: boolean;
  loadingModelsCount: number;
  progress: QueueItemProgress | null;
  queueItems: QueueItem[];
}): ProjectQueueIndicatorState => {
  const summary = getQueueSummary(queueItems, progress);
  const hasOpenQueueWork = summary.total > 0;
  const isRunning = summary.runningQueueItemId !== null;

  return {
    hasOpenQueueWork,
    progressState: hasOpenQueueWork
      ? getQueueProgressBarState({ isConnected, isRunning, loadingModelsCount, progress })
      : { kind: 'idle', value: 0 },
    runningQueueItemId: summary.runningQueueItemId,
  };
};
