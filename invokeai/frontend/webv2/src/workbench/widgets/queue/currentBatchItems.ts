import type { QueueServerItem } from './queueServerApi';

const isCurrentBatchStatus = (status: QueueServerItem['status']): boolean =>
  status === 'pending' || status === 'in_progress';

export const getCurrentBatchItems = ({
  current,
  items,
  next,
}: {
  current: QueueServerItem | null;
  items: QueueServerItem[];
  next: QueueServerItem | null;
}): QueueServerItem[] => {
  const batchId = current?.batch_id ?? next?.batch_id ?? null;

  if (!batchId) {
    return [];
  }

  const itemsById = new Map<number, QueueServerItem>();

  for (const item of [current, next, ...items]) {
    if (item && item.batch_id === batchId && isCurrentBatchStatus(item.status)) {
      itemsById.set(item.item_id, item);
    }
  }

  return [...itemsById.values()].sort((left, right) => {
    if (left.status === 'in_progress' && right.status !== 'in_progress') {
      return -1;
    }

    if (left.status !== 'in_progress' && right.status === 'in_progress') {
      return 1;
    }

    return left.item_id - right.item_id;
  });
};
