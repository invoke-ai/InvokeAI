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

export const clearPendingQueueItemReveal = (requestId: number): void => {
  if (queueRevealStore.getSnapshot().pendingReveal?.requestId === requestId) {
    queueRevealStore.setSnapshot({ pendingReveal: null });
  }
};

export const getPendingQueueItemReveal = (): QueueItemRevealRequest | null =>
  queueRevealStore.getSnapshot().pendingReveal;

export const usePendingQueueItemReveal = (): QueueItemRevealRequest | null =>
  queueRevealStore.useSelector((snapshot) => snapshot.pendingReveal, Object.is);
