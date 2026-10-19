import type { QueueItem } from './historyTypes';

import { isOpenQueueItem } from './historySummary';

/** Represent concurrent GPU sessions as separate segments rather than collapsing to one running item. */

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
  // Hide frozen progress offline; the server-status widget explains the connection state.
  if (!isConnected || !hasOpenWork) {
    return HIDDEN;
  }

  return sessionItemIds.length > 0 ? { itemIds: sessionItemIds, kind: 'sessions' } : PENDING;
};

/** Treat zero as indeterminate during startup/model loading; all progress surfaces share this interpretation. */
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

/** Filter server-wide progress to this project's enqueued IDs, then order by ID. */
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
