import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/** Queue actions callers outside the queue UI (hotkeys, the palette) may ask for. */
export type RequestableQueueActionId = 'cancel-all' | 'cancel-current';

export interface QueueActionRequest {
  actionId: RequestableQueueActionId;
  requestId: number;
}

let nextRequestId = 0;
const queueActionRequestStore = createExternalStore<{ request: QueueActionRequest | null }>({ request: null });

/**
 * Runs a queue action the way its menu item does, confirmation included. The queue's always-mounted host takes the
 * request, so a shortcut pressed before the queue UI loaded still runs once it has.
 */
export const requestQueueAction = (actionId: RequestableQueueActionId): void => {
  queueActionRequestStore.setSnapshot({ request: { actionId, requestId: ++nextRequestId } });
};

/** Takes the request once: false when it was already taken or replaced. */
export const takeQueueActionRequest = (requestId: number): boolean => {
  if (queueActionRequestStore.getSnapshot().request?.requestId !== requestId) {
    return false;
  }

  queueActionRequestStore.setSnapshot({ request: null });
  return true;
};

registerAccountOwnedResource({
  clear: () => queueActionRequestStore.setSnapshot({ request: null }),
  name: 'queue-action-requests',
});

export const useQueueActionRequest = (): QueueActionRequest | null =>
  queueActionRequestStore.useSelector((snapshot) => snapshot.request, Object.is);
