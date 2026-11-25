import type { QueueItem } from './historyTypes';
import type { QueueItemProgressTarget, QueueSourceId } from './types';

import { getQueueItemSnapshotBatchCount, getQueueItemSnapshotDimensions } from './historySnapshot';

export interface QueueActiveSession extends QueueItemProgressTarget {
  id: string;
  backendItemId: number;
  label: string;
  sourceId: QueueSourceId;
  width: number;
  height: number;
  itemCount: number;
  /** The session has shown a frame of its own; live surfaces follow it only from then on. */
  hasOwnFrame: boolean;
  state: 'running' | 'settling';
}

/** Presentation policy shared by Gallery and Preview; discovery remains destination-neutral. */
export const isGalleryProgressItem = (item: QueueItem): boolean => item.snapshot.destination === 'gallery';

const getTargetKey = (target: QueueItemProgressTarget): string => `${target.queueItemId}:${target.itemIndex}`;

export const getQueueActiveSessions = (
  items: readonly QueueItem[],
  running: readonly QueueItemProgressTarget[],
  followed: readonly QueueItemProgressTarget[],
  framed: readonly QueueItemProgressTarget[]
): QueueActiveSession[] => {
  const itemsById = new Map(items.map((item) => [item.id, item]));
  const runningKeys = new Set(running.map(getTargetKey));
  const framedKeys = new Set(framed.map(getTargetKey));
  const sessions: QueueActiveSession[] = [];

  for (const target of followed) {
    const item = itemsById.get(target.queueItemId);
    const backendItemId = item?.backendItemIds?.[target.itemIndex - 1];
    if (!item || backendItemId === undefined) {
      continue;
    }
    const id = getTargetKey(target);
    sessions.push({
      ...target,
      ...getQueueItemSnapshotDimensions(item, { width: 1024, height: 1024 }),
      id,
      backendItemId,
      label: item.snapshot.graph.label,
      sourceId: item.snapshot.sourceId,
      itemCount: item.backendItemIds!.length,
      hasOwnFrame: framedKeys.has(id),
      state: runningKeys.has(id) ? 'running' : 'settling',
    });
  }

  return sessions.sort((left, right) => left.backendItemId - right.backendItemId);
};

/**
 * Shared by Preview and Canvas staging: `latest` and `progress` select each landed result; `off` only when none is.
 * Only `progress` shows generating slots, from their own first frame; picking a result pauses it (Off).
 */
export type AutoSwitchMode = 'latest' | 'off' | 'progress';

export interface AutoSwitchLiveSlot {
  id: string;
  hasOwnFrame: boolean;
  state: 'queued' | 'running' | 'settling';
}

/** A pin holds only while its slot runs; a slot that settles, ends or never started releases it. */
export const isAutoSwitchPinHeld = (lives: readonly AutoSwitchLiveSlot[], pinnedId: string | null): boolean =>
  pinnedId !== null && lives.some((live) => live.id === pinnedId && live.state === 'running');

/** In progress only: the pinned running slot, else the newest running one with a frame, else a settling one. */
export const getAutoSwitchFollowedLive = <T extends AutoSwitchLiveSlot>(
  mode: AutoSwitchMode,
  lives: readonly T[],
  pinnedId: string | null
): T | null => {
  if (mode !== 'progress') {
    return null;
  }

  return (
    (isAutoSwitchPinHeld(lives, pinnedId) ? lives.find((live) => live.id === pinnedId) : undefined) ??
    lives.filter((live) => live.state === 'running' && live.hasOwnFrame).at(-1) ??
    lives.find((live) => live.state === 'settling' && live.hasOwnFrame) ??
    null
  );
};

export const pickAutoSwitch = (
  pick: { kind: 'live'; id: string } | { kind: 'result' }
): { mode: AutoSwitchMode; pinnedId: string | null } =>
  pick.kind === 'live' ? { mode: 'progress', pinnedId: pick.id } : { mode: 'off', pinnedId: null };

export const selectsLandedResult = (mode: AutoSwitchMode, hasSelectedResult: boolean): boolean =>
  mode !== 'off' || !hasSelectedResult;

/** A tile per unfinished batch slot, including work waiting to start. */
export type QueueProgressSession = Omit<QueueActiveSession, 'backendItemId' | 'hasOwnFrame' | 'state'> & {
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
