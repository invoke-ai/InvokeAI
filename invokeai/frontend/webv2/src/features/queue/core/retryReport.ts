/**
 * What a retry did, as a notice: the first rerun's id when the server queued any, how many it queued of those asked
 * for, and whether a Canvas rerun goes to a tab other than this one. The queue's Retry button and a failure notice's
 * Retry report the same way.
 */
export interface QueueRetryReport {
  kind: 'info' | 'success';
  messageKey: string;
  /** The first item that re-runs the retried ones; absent when the server queued none. */
  rerunItemId?: number;
  titleKey: string;
  /**
   * `count`: items queued again, or items asked for when none was. `id`: the one item asked for, named only then.
   * `total`: items asked for, when the server queued fewer.
   */
  values: Readonly<Record<string, number>>;
}

export const describeQueueRetry = ({
  isProjectClosedHere,
  itemIds,
  retriedItemIds,
}: {
  /** A Canvas run whose project is not open here: its rerun is staged by another tab. */
  isProjectClosedHere: boolean;
  /** The failed or canceled items asked to re-run. */
  itemIds: readonly number[];
  retriedItemIds: readonly number[];
}): QueueRetryReport => {
  const [rerunItemId] = retriedItemIds;
  // A single item asked for is named; the server may skip some of several, so those are only counted.
  const named: Record<string, number> = itemIds.length === 1 ? { id: itemIds[0]! } : {};

  if (rerunItemId === undefined) {
    return {
      kind: 'info',
      messageKey: 'widgets.queue.retryNotQueuedDescription',
      titleKey: 'widgets.queue.retryNotQueued',
      values: { count: itemIds.length, ...named },
    };
  }

  const isPartial = retriedItemIds.length < itemIds.length;
  const messageKey = isPartial
    ? isProjectClosedHere
      ? 'widgets.queue.retryQueuedPartialProjectClosedDescription'
      : 'widgets.queue.retryQueuedPartialDescription'
    : isProjectClosedHere
      ? 'widgets.queue.retryQueuedProjectClosedDescription'
      : 'widgets.queue.retryQueuedDescription';

  return {
    kind: 'success',
    messageKey,
    rerunItemId,
    titleKey: 'widgets.queue.retryQueued',
    values: isPartial
      ? { count: retriedItemIds.length, total: itemIds.length }
      : { count: retriedItemIds.length, ...named },
  };
};
