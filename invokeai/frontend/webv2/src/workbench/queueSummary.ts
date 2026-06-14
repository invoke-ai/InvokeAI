import type { QueueItem } from './types';
import type { QueueItemProgress } from './backend/progressStore';

export interface QueueSummary {
  current: number;
  runningQueueItemId: string | null;
  total: number;
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

  // Only Generate snapshots batch by count; other sources (project graph) run once.
  const batchCount = item.snapshot.sourceId === 'generate' ? item.snapshot.widgetStates.generate?.values.batchCount : 1;

  return typeof batchCount === 'number' && Number.isFinite(batchCount) ? Math.max(1, Math.round(batchCount)) : 1;
};

const compareQueuePosition = (a: QueueItem, b: QueueItem): number => {
  const aTime = Date.parse(a.snapshot.submittedAt);
  const bTime = Date.parse(b.snapshot.submittedAt);

  return (Number.isFinite(aTime) ? aTime : 0) - (Number.isFinite(bTime) ? bTime : 0);
};

const getRunningImageOrdinal = (item: QueueItem, progress: QueueItemProgress | null): number => {
  const expectedCount = getQueueItemExpectedImageCount(item);
  const completedCount = progress?.completedItemCount ?? 0;
  const activeIndex = progress?.activeItemIndex ?? completedCount + 1;

  return Math.min(expectedCount, Math.max(1, activeIndex));
};

export const getQueueSummary = (items: QueueItem[], progress: QueueItemProgress | null = null): QueueSummary => {
  const openItems = items.filter(isOpenQueueItem).sort(compareQueuePosition);
  const total = openItems.reduce((sum, item) => sum + getQueueItemExpectedImageCount(item), 0);

  if (total === 0) {
    return { current: 0, runningQueueItemId: null, total: 0 };
  }

  const runningItem = openItems.find((item) => item.status === 'running');

  if (!runningItem) {
    return { current: 0, runningQueueItemId: null, total };
  }

  let current = 0;

  for (const item of openItems) {
    if (item.id === runningItem.id) {
      current += getRunningImageOrdinal(item, progress);
      break;
    }

    current += getQueueItemExpectedImageCount(item);
  }

  return { current: Math.min(current, total), runningQueueItemId: runningItem.id, total };
};
