import type { QueueItem } from '@features/queue/core/historyTypes';

import { createStableSelector } from '@platform/state/selectors';

export interface QueueRow {
  item: QueueItem;
  projectId: string;
  projectName: string;
}

export interface QueueActionState {
  cancellableCount: number;
  currentItemId: string | null;
  hasPendingQueueWork: boolean;
  hasRunningItem: boolean;
}

export const areQueueRowsEqual = (left: readonly QueueRow[], right: readonly QueueRow[]): boolean =>
  left.length === right.length &&
  left.every((row, index) => {
    const other = right[index];

    return (
      other !== undefined &&
      row.item === other.item &&
      row.projectId === other.projectId &&
      row.projectName === other.projectName
    );
  });

export const createQueueRowsSelector = () =>
  createStableSelector(
    (projects: readonly { id: string; name: string; queue: { items: QueueItem[] } }[]): QueueRow[] =>
      projects.flatMap((project) =>
        project.queue.items.map((item) => ({ item, projectId: project.id, projectName: project.name }))
      ),
    areQueueRowsEqual
  );

export const getPendingQueueCount = (items: readonly QueueItem[]): number =>
  items.reduce((count, item) => (item.status === 'pending' || item.status === 'running' ? count + 1 : count), 0);

export const hasPendingWorkflowQueueItem = (items: readonly QueueItem[]): boolean =>
  items.some(
    (item) => item.snapshot.sourceId === 'workflow' && (item.status === 'running' || item.status === 'pending')
  );

const isCancellableQueueItem = (item: QueueItem): boolean =>
  item.cancellable && (item.status === 'pending' || item.status === 'running');

const compareQueueSubmittedAt = (left: QueueItem, right: QueueItem): number => {
  const leftTime = Date.parse(left.snapshot.submittedAt);
  const rightTime = Date.parse(right.snapshot.submittedAt);

  return (Number.isFinite(leftTime) ? leftTime : 0) - (Number.isFinite(rightTime) ? rightTime : 0);
};

const getCurrentQueueActionItem = (queueItems: readonly QueueItem[]): QueueItem | null => {
  const runningItem = queueItems.filter(isCancellableQueueItem).find((item) => item.status === 'running');

  if (runningItem) {
    return runningItem;
  }

  return queueItems.filter(isCancellableQueueItem).sort(compareQueueSubmittedAt)[0] ?? null;
};

const getOpenBackendSlotCount = (item: QueueItem): number => {
  if (!item.backendItemIds?.length) {
    return item.status === 'pending' || item.status === 'running' ? 1 : 0;
  }

  const terminalBackendItemIds = new Set([
    ...(item.completedBackendItemIds ?? []),
    ...(item.cancelledBackendItemIds ?? []),
  ]);

  return item.backendItemIds.filter((backendItemId) => !terminalBackendItemIds.has(backendItemId)).length;
};

export const getQueueActionState = (queueItems: readonly QueueItem[]): QueueActionState => {
  const currentItem = getCurrentQueueActionItem(queueItems);

  return {
    cancellableCount: queueItems.filter(isCancellableQueueItem).length,
    currentItemId: currentItem?.id ?? null,
    hasPendingQueueWork: queueItems.some((item) =>
      item.id === currentItem?.id
        ? item.status === 'running' && getOpenBackendSlotCount(item) > 1
        : item.status === 'pending'
    ),
    hasRunningItem: queueItems.some((item) => isCancellableQueueItem(item) && item.status === 'running'),
  };
};
