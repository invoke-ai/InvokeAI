import type { S } from 'services/api/types';

type QueueStatusResponse = S['SessionQueueAndProcessorStatus'];
type QueueItemStatusChangedEvent = S['QueueItemStatusChangedEvent'];

const TERMINAL_QUEUE_ITEM_STATUSES = new Set<QueueItemStatusChangedEvent['status']>([
  'completed',
  'failed',
  'canceled',
]);

let latestObservedQueueStatus: S['SessionQueueStatus'] | null = null;
const observedItemStatusSequences = new Map<number, number>();
const observedTerminalItemIds = new Set<number>();

const shouldAcceptQueueItemStatusEvent = (event: QueueItemStatusChangedEvent): boolean => {
  const statusSequence = event.status_sequence ?? undefined;
  if (statusSequence === undefined) {
    return true;
  }

  const previousSequence = observedItemStatusSequences.get(event.item_id);
  return previousSequence === undefined || statusSequence >= previousSequence;
};

const recordQueueItemStatusChangedEvent = (event: QueueItemStatusChangedEvent): void => {
  if (!shouldAcceptQueueItemStatusEvent(event)) {
    return;
  }

  if (event.status_sequence !== null && event.status_sequence !== undefined) {
    observedItemStatusSequences.set(event.item_id, event.status_sequence);
  }

  if (TERMINAL_QUEUE_ITEM_STATUSES.has(event.status)) {
    observedTerminalItemIds.add(event.item_id);
  } else {
    observedTerminalItemIds.delete(event.item_id);
  }

  latestObservedQueueStatus = event.queue_status;
};

/**
 * The queue status embedded in a full queue_item_status_changed event carries the OWNER's
 * per-user counts, so the owner's optimistic cache write below is complete and personal UI
 * (progress bar, spinner, favicon) updates without waiting for a refetch. Statuses without
 * per-user counts still occur — the sanitized companion nulls them — and for those, retain the
 * previous cached per-user counts: for a non-owner recipient they are still accurate, and the
 * tag invalidation that accompanies every status change refetches fresh counts immediately.
 */
const withRetainedUserCounts = (
  next: S['SessionQueueStatus'],
  previous: S['SessionQueueStatus']
): S['SessionQueueStatus'] => ({
  ...next,
  user_pending: next.user_pending ?? previous.user_pending,
  user_in_progress: next.user_in_progress ?? previous.user_in_progress,
});

export const getQueueStatusWithObservedEvents = (queueStatus: QueueStatusResponse): QueueStatusResponse => {
  const currentItemId = queueStatus.queue.item_id;
  if (latestObservedQueueStatus && currentItemId !== null && observedTerminalItemIds.has(currentItemId)) {
    return {
      ...queueStatus,
      queue: withRetainedUserCounts(latestObservedQueueStatus, queueStatus.queue),
    };
  }

  return queueStatus;
};

export const getUpdatedQueueStatusOnQueueItemStatusChanged = (
  queueStatus: QueueStatusResponse,
  event: QueueItemStatusChangedEvent
): QueueStatusResponse => {
  recordQueueItemStatusChangedEvent(event);
  return {
    ...queueStatus,
    queue: withRetainedUserCounts(event.queue_status, queueStatus.queue),
  };
};

export const resetObservedQueueStatusEventsForTests = (): void => {
  latestObservedQueueStatus = null;
  observedItemStatusSequences.clear();
  observedTerminalItemIds.clear();
};
