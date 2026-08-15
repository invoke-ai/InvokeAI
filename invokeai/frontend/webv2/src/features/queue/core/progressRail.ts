import type { QueueItem } from './historyTypes';

import { isOpenQueueItem } from './historySummary';

/**
 * The read model behind the top bar's progress rail — the one full-width,
 * always-present generation indicator.
 *
 * It is deliberately a *list* of segments rather than a single value. With
 * `generation_devices` (default `auto`) the backend runs one session per GPU,
 * so several images render at once; every other progress surface in the app
 * collapses to `runningQueueItemId` and therefore reports one of them as if it
 * were the whole story. The rail gives each live session its own segment, the
 * way the legacy progress bar did.
 */

export type ProgressRailModel =
  | { kind: 'hidden' }
  /** Work is open but no session has reported yet — one indeterminate segment. */
  | { kind: 'pending' }
  | { kind: 'sessions'; itemIds: readonly number[] };

const HIDDEN: ProgressRailModel = { kind: 'hidden' };
const PENDING: ProgressRailModel = { kind: 'pending' };
const NO_ITEM_IDS: number[] = [];

export const getProgressRailModel = ({
  hasOpenWork,
  isConnected,
  sessionItemIds,
}: {
  hasOpenWork: boolean;
  isConnected: boolean;
  sessionItemIds: readonly number[];
}): ProgressRailModel => {
  // Offline, a frozen bar would claim progress that cannot be arriving. The
  // server-status widget owns saying why.
  if (!isConnected || !hasOpenWork) {
    return HIDDEN;
  }

  return sessionItemIds.length > 0 ? { itemIds: sessionItemIds, kind: 'sessions' } : PENDING;
};

/**
 * A determinate fill fraction, or null for the indeterminate sweep.
 *
 * Zero is indeterminate rather than an empty bar: the backend reports `0` for
 * the span between "session started" and "first step done", which is exactly
 * when models are loading and the wait feels longest. An empty determinate
 * bar (or a "0%" label) reads as stalled there; the sweep reads as working.
 * Every progress surface routes through this so they cannot disagree.
 */
export const getDeterminateProgressFraction = (percentage: number | null | undefined): number | null =>
  typeof percentage === 'number' && percentage > 0 ? percentage : null;

/** Whole percent for text surfaces (tab title, chip label), or null while indeterminate. */
export const getDeterminateProgressPercent = (percentage: number | null | undefined): number | null => {
  const fraction = getDeterminateProgressFraction(percentage);

  return fraction === null ? null : Math.round(fraction * 100);
};

/** A rail segment's fill: also indeterminate while its model is still loading. */
export const getProgressRailSegmentValue = ({
  isLoadingModels,
  percentage,
}: {
  isLoadingModels: boolean;
  percentage: number | null | undefined;
}): number | null => (isLoadingModels ? null : getDeterminateProgressFraction(percentage));

/**
 * The live sessions belonging to this project, in ascending id order.
 *
 * `itemProgressStore` is keyed by backend item id and spans the whole server
 * queue, which on a multi-user backend includes other people's work; the rail
 * sits beside a project-scoped queue readout, so it has to filter down to the
 * ids this project actually enqueued.
 */
export const selectProjectProgressItemIds = (
  queueItems: readonly QueueItem[],
  activeItemIds: readonly number[]
): number[] => {
  if (activeItemIds.length === 0) {
    return NO_ITEM_IDS;
  }

  const projectItemIds = new Set<number>();

  for (const item of queueItems) {
    if (!isOpenQueueItem(item)) {
      continue;
    }

    for (const backendItemId of item.backendItemIds ?? []) {
      projectItemIds.add(backendItemId);
    }
  }

  // `activeItemIds` arrives sorted, so filtering preserves a stable
  // left-to-right segment order across frames.
  return activeItemIds.filter((itemId) => projectItemIds.has(itemId));
};
