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

export const getQueueStatusWithObservedEvents = (queueStatus: QueueStatusResponse): QueueStatusResponse => {
  const currentItemId = queueStatus.queue.item_id;
  if (latestObservedQueueStatus && currentItemId !== null && observedTerminalItemIds.has(currentItemId)) {
    return {
      ...queueStatus,
      queue: latestObservedQueueStatus,
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
    queue: event.queue_status,
  };
};

export const resetObservedQueueStatusEventsForTests = (): void => {
  latestObservedQueueStatus = null;
  observedItemStatusSequences.clear();
  observedTerminalItemIds.clear();
};
