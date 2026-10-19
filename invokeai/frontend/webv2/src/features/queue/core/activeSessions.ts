import type { QueueItem } from './historyTypes';
import type { QueueItemProgressTarget, QueueSourceId } from './types';

import { getQueueItemSnapshotBatchCount, getQueueItemSnapshotDimensions } from './historySnapshot';

/** A running backend session, not a prediction of its output images or boards. */
export interface QueueActiveSession extends QueueItemProgressTarget {
  id: string;
  backendItemId: number;
  label: string;
  sourceId: QueueSourceId;
  width: number;
  height: number;
  itemCount: number;
  state: 'running' | 'settling';
}

/** Presentation policy shared by Gallery and Preview; discovery remains destination-neutral. */
export const isGalleryProgressItem = (item: QueueItem): boolean => item.snapshot.destination === 'gallery';

export const getQueueActiveSessions = (
  items: readonly QueueItem[],
  running: readonly QueueItemProgressTarget[],
  followed: readonly QueueItemProgressTarget[]
): QueueActiveSession[] => {
  const itemsById = new Map(items.map((item) => [item.id, item]));
  const runningKeys = new Set(running.map((target) => `${target.queueItemId}:${target.itemIndex}`));
  const sessions: QueueActiveSession[] = [];

  for (const target of followed) {
    const item = itemsById.get(target.queueItemId);
    const backendItemId = item?.backendItemIds?.[target.itemIndex - 1];
    if (!item || backendItemId === undefined) {
      continue;
    }
    const id = `${target.queueItemId}:${target.itemIndex}`;
    sessions.push({
      ...target,
      ...getQueueItemSnapshotDimensions(item, { width: 1024, height: 1024 }),
      id,
      backendItemId,
      label: item.snapshot.graph.label,
      sourceId: item.snapshot.sourceId,
      itemCount: item.backendItemIds!.length,
      state: runningKeys.has(id) ? 'running' : 'settling',
    });
  }

  return sessions.sort((left, right) => left.backendItemId - right.backendItemId);
};

/**
 * The session the live preview shows and the gallery highlights: the pinned
 * one, else the first running, else the first still settling; queued work is
 * never followed.
 */
export const getFollowedProgressSession = <T extends { id: string; state: QueueProgressSession['state'] }>(
  sessions: readonly T[],
  pinnedSessionId: string | null
): T | null =>
  sessions.find((session) => session.id === pinnedSessionId) ??
  sessions.find((session) => session.state === 'running') ??
  sessions.find((session) => session.state === 'settling') ??
  null;

/** A tile per unfinished batch slot, including work waiting to start. */
export type QueueProgressSession = Omit<QueueActiveSession, 'backendItemId' | 'state'> & {
  backendItemId: number | null;
  state: 'queued' | 'running' | 'settling';
};

export const getQueueProgressSessions = (
  items: readonly QueueItem[],
  active: readonly QueueActiveSession[]
): QueueProgressSession[] => {
  const sessions: QueueProgressSession[] = [...active];
  const activeIds = new Set(active.map((session) => session.id));
  for (const item of items) {
    if (item.status !== 'pending' && item.status !== 'running') {
      continue;
    }
    const finished = new Set([...(item.completedBackendItemIds ?? []), ...(item.cancelledBackendItemIds ?? [])]);
    const itemCount = item.backendItemIds?.length ?? getQueueItemSnapshotBatchCount(item);
    for (let index = 0; index < itemCount; index += 1) {
      const id = `${item.id}:${index + 1}`;
      const backendItemId = item.backendItemIds?.[index] ?? null;
      if (activeIds.has(id) || (backendItemId !== null && finished.has(backendItemId))) {
        continue;
      }
      sessions.push({
        ...getQueueItemSnapshotDimensions(item, { width: 1024, height: 1024 }),
        id,
        queueItemId: item.id,
        itemIndex: index + 1,
        backendItemId,
        label: item.snapshot.graph.label,
        sourceId: item.snapshot.sourceId,
        itemCount,
        state: 'queued',
      });
    }
  }
  const submittedAt = new Map(items.map((item) => [item.id, item.snapshot.submittedAt]));
  return sessions.sort(
    (left, right) =>
      submittedAt.get(left.queueItemId)!.localeCompare(submittedAt.get(right.queueItemId)!) ||
      left.queueItemId.localeCompare(right.queueItemId) ||
      left.itemIndex - right.itemIndex
  );
};
