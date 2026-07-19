import type { QueueItem } from './historyTypes';

export const getQueueItemSnapshotBatchCount = (item: QueueItem): number => item.snapshot.presentation.batchCount;

export const getQueueItemSnapshotDimensions = (
  item: QueueItem,
  fallback: { height: number; width: number }
): { height: number; width: number } => ({
  height: item.snapshot.presentation.height || fallback.height,
  width: item.snapshot.presentation.width || fallback.width,
});
