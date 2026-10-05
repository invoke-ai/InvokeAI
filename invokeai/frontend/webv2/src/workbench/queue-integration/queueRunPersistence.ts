import type { WorkbenchQueueItem } from '@workbench/queueHistoryContracts';

import type { QueueRecallCache, QueueRecallSnapshotInput } from './queueRecallCache';
import type { QueueRunJournal } from './queueRunJournal';

export const withQueueRecallCache = <T extends Pick<QueueRunJournal, 'record' | 'settle'>>(
  journal: T,
  cache: Pick<QueueRecallCache, 'put'>,
  getItem: (projectId: string, queueItemId: string) => WorkbenchQueueItem | undefined
): T => {
  const pendingRecall = new Map<string, QueueRecallSnapshotInput>();
  const keyOf = (projectId: string, queueItemId: string) => JSON.stringify([projectId, queueItemId]);
  const remember = (projectId: string, item: WorkbenchQueueItem): void => {
    const { recall, sourceId, submittedAt } = item.snapshot;
    if (recall?.generateValues || recall?.videoValues) {
      pendingRecall.set(keyOf(projectId, item.id), {
        ...recall,
        projectId,
        queueItemId: item.id,
        sourceId,
        submittedAt,
      });
    }
  };
  return {
    ...journal,
    record(projectId, item) {
      const liveItem = item as WorkbenchQueueItem;
      remember(projectId, liveItem);
      return journal.record(projectId, item);
    },
    async settle(projectId, queueItemId) {
      const key = keyOf(projectId, queueItemId);
      const item = getItem(projectId, queueItemId);
      if (item) {
        remember(projectId, item);
      }
      const snapshot = pendingRecall.get(key);
      if (snapshot) {
        await cache.put(snapshot).catch(() => undefined);
        pendingRecall.delete(key);
      }
      return journal.settle(projectId, queueItemId);
    },
  };
};
