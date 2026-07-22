import { createExternalStore } from '@platform/state/externalStore';

/**
 * Cross-context requests into the queue widget. Shell surfaces (the command
 * palette) can't reach the widget's local state, so they park a request here
 * and the mounted widget consumes it — the same bridge workflowUiStore uses
 * for pending library-workflow loads.
 */
const queueUiStore = createExternalStore<{ pendingRevealItemId: number | null }>({ pendingRevealItemId: null });

export const requestQueueItemReveal = (itemId: number): void => {
  queueUiStore.setSnapshot({ pendingRevealItemId: itemId });
};

export const clearPendingQueueItemReveal = (): void => {
  queueUiStore.setSnapshot({ pendingRevealItemId: null });
};

export const usePendingQueueItemReveal = (): number | null =>
  queueUiStore.useSelector((snapshot) => snapshot.pendingRevealItemId, Object.is);
