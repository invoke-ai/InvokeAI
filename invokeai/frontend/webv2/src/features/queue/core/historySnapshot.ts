import type { QueueItem } from './historyTypes';

const getPresentation = (item: QueueItem): Partial<QueueItem['snapshot']['presentation']> =>
  ((item as Partial<QueueItem>).snapshot as Partial<QueueItem['snapshot']> | undefined)?.presentation ?? {};

export const getQueueItemSnapshotBatchCount = (item: QueueItem): number => {
  const batchCount = getPresentation(item).batchCount;
  return typeof batchCount === 'number' && Number.isFinite(batchCount) && batchCount > 0 ? batchCount : 1;
};

export const getQueueItemSnapshotPositivePrompt = (item: QueueItem): string => {
  const positivePrompt = getPresentation(item).positivePrompt;
  return typeof positivePrompt === 'string' ? positivePrompt : '';
};

export const getQueueItemSnapshotDimensions = (
  item: QueueItem,
  fallback: { height: number; width: number }
): { height: number; width: number } => ({
  height: getPresentation(item).height || fallback.height,
  width: getPresentation(item).width || fallback.width,
});
