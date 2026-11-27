import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

export interface QueueItemRevealRequest {
  itemId: number;
  requestId: number;
}

let nextRevealRequestId = 0;
const queueRevealStore = createExternalStore<{ pendingReveal: QueueItemRevealRequest | null }>({
  pendingReveal: null,
});

export const requestQueueItemReveal = (itemId: number): void => {
  queueRevealStore.setSnapshot({ pendingReveal: { itemId, requestId: ++nextRevealRequestId } });
};

/** Settles the request once: false when it was already settled or replaced. */
export const clearPendingQueueItemReveal = (requestId: number): boolean => {
  if (queueRevealStore.getSnapshot().pendingReveal?.requestId !== requestId) {
    return false;
  }

  queueRevealStore.setSnapshot({ pendingReveal: null });
  return true;
};

const resetQueueUi = (): void => {
  nextRevealRequestId = 0;
  queueRevealStore.setSnapshot({ pendingReveal: null });
};

registerAccountOwnedResource({
  clear: resetQueueUi,
  name: 'queue-ui',
});

export const getPendingQueueItemReveal = (): QueueItemRevealRequest | null =>
  queueRevealStore.getSnapshot().pendingReveal;

export const usePendingQueueItemReveal = (): QueueItemRevealRequest | null =>
  queueRevealStore.useSelector((snapshot) => snapshot.pendingReveal, Object.is);
