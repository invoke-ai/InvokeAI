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
 * Auto-switch, one policy for every surface that shows generation results: Preview (`viewerAutoSwitch`) and Canvas
 * staging (`autoSwitchMode`). Each surface keeps its own selected result and adapts its own slots; these rules decide
 * what follows what.
 *
 * - `latest` (Newest) and `progress` (In progress) select each result as it lands; `off` selects one only while
 *   nothing else is selected (`selectsLandedResult`).
 * - Only `progress` shows generating slots, and only from their first frame of their own (`getAutoSwitchFollowedLive`):
 *   until then the previous result stays on screen.
 * - The user picking something to show steers it (`pickAutoSwitch`): a result pauses Auto-switch (Off) until the mode
 *   is set again or a generating slot is picked; a generating slot turns it to In progress and pins that slot while it
 *   runs.
 */
export type AutoSwitchMode = 'latest' | 'off' | 'progress';

/** A generating slot as Auto-switch sees it: waiting to start, running, or finished with its result still on its way. */
export interface AutoSwitchLiveSlot {
  id: string;
  hasOwnFrame: boolean;
  state: 'queued' | 'running' | 'settling';
}

/** A pin holds only while its slot runs; a slot that settles, ends or never started releases it. */
export const isAutoSwitchPinHeld = (lives: readonly AutoSwitchLiveSlot[], pinnedId: string | null): boolean =>
  pinnedId !== null && lives.some((live) => live.id === pinnedId && live.state === 'running');

/**
 * The generating slot Auto-switch shows, or null to show the selected result: under In progress only, the pinned slot
 * while it runs, else the newest running slot with a frame of its own, else the first settling one with a frame (so a
 * finished slot's last frame stays up while its result is on its way). `lives` are in the order they started.
 */
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

/** What the user picking something to show does to Auto-switch. */
export const pickAutoSwitch = (
  pick: { kind: 'live'; id: string } | { kind: 'result' }
): { mode: AutoSwitchMode; pinnedId: string | null } =>
  pick.kind === 'live' ? { mode: 'progress', pinnedId: pick.id } : { mode: 'off', pinnedId: null };

/** Whether a result that just landed becomes the selected one. */
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
