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

export const clearPendingQueueItemReveal = (requestId: number): void => {
  if (queueRevealStore.getSnapshot().pendingReveal?.requestId === requestId) {
    queueRevealStore.setSnapshot({ pendingReveal: null });
  }
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
