/**
 * What a retry did, as a notice: the rerun's id when the server queued one, and whether a Canvas rerun goes to a tab
 * other than this one. The queue's Retry button and a failure notice's Retry report the same way.
 */
export interface QueueRetryReport {
  kind: 'info' | 'success';
  messageKey: string;
  /** The item that re-runs the retried one; absent when the server queued none. */
  rerunItemId?: number;
  titleKey: string;
  values: { id: number };
}

export const describeQueueRetry = ({
  isProjectClosedHere,
  itemId,
  retriedItemIds,
}: {
  /** A Canvas run whose project is not open here: its rerun is staged by another tab. */
  isProjectClosedHere: boolean;
  itemId: number;
  retriedItemIds: readonly number[];
}): QueueRetryReport => {
  const [rerunItemId] = retriedItemIds;

  if (rerunItemId === undefined) {
    return {
      kind: 'info',
      messageKey: 'widgets.queue.retryNotQueuedDescription',
      titleKey: 'widgets.queue.retryNotQueued',
      values: { id: itemId },
    };
  }

  return {
    kind: 'success',
    messageKey: isProjectClosedHere
      ? 'widgets.queue.retryQueuedProjectClosedDescription'
      : 'widgets.queue.retryQueuedDescription',
    rerunItemId,
    titleKey: 'widgets.queue.retryQueued',
    values: { id: itemId },
  };
};
