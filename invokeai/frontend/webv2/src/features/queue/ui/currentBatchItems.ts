import type { QueueItemReadModel } from '@features/queue/core/types';

const isCurrentBatchStatus = (status: QueueItemReadModel['status']): boolean =>
  status === 'pending' || status === 'in_progress';

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

  if (!batchId) {
    return [];
  }

  const itemsById = new Map<number, QueueItemReadModel>();

  for (const item of [current, next, ...items]) {
    if (item && item.batchId === batchId && isCurrentBatchStatus(item.status)) {
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
