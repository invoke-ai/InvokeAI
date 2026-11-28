import type { QueueItemReadModel } from '@features/queue/core/types';

export const getCurrentBatchItems = ({
  current,
  items,
  next,
}: {
  current: QueueItemReadModel | null;
  items: QueueItemReadModel[];
  next: QueueItemReadModel | null;
}): QueueItemReadModel[] => {
  const batchId = current?.batchId ?? next?.batchId ?? null;
  const itemsById = new Map<number, QueueItemReadModel>();

  for (const item of [current, next, ...items]) {
    if (!item) {
      continue;
    }

    const isRunning = item.status === 'in_progress';
    const isPendingCurrentBatch = batchId !== null && item.status === 'pending' && item.batchId === batchId;

    if (isRunning || isPendingCurrentBatch) {
      itemsById.set(item.id, item);
    }
  }

  return [...itemsById.values()].sort((left, right) => {
    if (left.status === 'in_progress' && right.status !== 'in_progress') {
      return -1;
    }

    if (left.status !== 'in_progress' && right.status === 'in_progress') {
      return 1;
    }

    return left.id - right.id;
  });
};
