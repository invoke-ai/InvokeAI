import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { $lastCanvasProgressEvent } from 'features/controlLayers/store/canvasSlice';
import { queueApi } from 'services/api/endpoints/queue';

/**
 * To prevent a race condition where a progress event arrives after a successful cancellation, we need to keep track of
 * cancellations:
 * - In the route handlers above, we track and update the cancellations object
 * - When the user queues a, we should reset the cancellations, also handled int he route handlers above
 * - When we get a progress event, we should check if the event is cancelled before setting the event
 *
 * We have a few ways that cancellations are effected, so we need to track them all:
 * - by queue item id (in this case, we will compare the session_id and not the item_id)
 * - by batch id
 * - by destination
 * - by clearing the queue
 */
type Cancellations = {
  sessionIds: Set<string>;
  batchIds: Set<string>;
  destinations: Set<string>;
  clearQueue: boolean;
};

const resetCancellations = (): void => {
  cancellations.clearQueue = false;
  cancellations.sessionIds.clear();
  cancellations.batchIds.clear();
  cancellations.destinations.clear();
};

const cancellations: Cancellations = {
  sessionIds: new Set(),
  batchIds: new Set(),
  destinations: new Set(),
  clearQueue: false,
} as Readonly<Cancellations>;

/**
 * Checks if an item is cancelled, used to prevent race conditions with event handling.
 *
 * To use this, provide the session_id, batch_id and destination from the event payload.
 */
export const getIsCancelled = (item: {
  session_id: string;
  batch_id: string;
  destination?: string | null;
}): boolean => {
  if (cancellations.clearQueue) {
    return true;
  }
  if (cancellations.sessionIds.has(item.session_id)) {
    return true;
  }
  if (cancellations.batchIds.has(item.batch_id)) {
    return true;
  }
  if (item.destination && cancellations.destinations.has(item.destination)) {
    return true;
  }
  return false;
};

export const addCancellationsListeners = (startAppListening: AppStartListening) => {
  // When we get a cancellation, we may need to clear the last progress event - next few listeners handle those cases.
  // Maybe we could use the `getIsCancelled` util here, but I think that could introduce _another_ race condition...
  startAppListening({
    matcher: queueApi.endpoints.enqueueBatch.matchFulfilled,
    effect: () => {
      resetCancellations();
    },
  });

  startAppListening({
    matcher: queueApi.endpoints.cancelByBatchDestination.matchFulfilled,
    effect: (action) => {
      cancellations.destinations.add(action.meta.arg.originalArgs.destination);

      const event = $lastCanvasProgressEvent.get();
      if (!event) {
        return;
      }
      const { session_id, batch_id, destination } = event;
      if (getIsCancelled({ session_id, batch_id, destination })) {
        $lastCanvasProgressEvent.set(null);
      }
    },
  });

  startAppListening({
    matcher: queueApi.endpoints.cancelQueueItem.matchFulfilled,
    effect: (action) => {
      cancellations.sessionIds.add(action.payload.session_id);

      const event = $lastCanvasProgressEvent.get();
      if (!event) {
        return;
      }
      const { session_id, batch_id, destination } = event;
      if (getIsCancelled({ session_id, batch_id, destination })) {
        $lastCanvasProgressEvent.set(null);
      }
    },
  });

  startAppListening({
    matcher: queueApi.endpoints.cancelByBatchIds.matchFulfilled,
    effect: (action) => {
      for (const batch_id of action.meta.arg.originalArgs.batch_ids) {
        cancellations.batchIds.add(batch_id);
      }
      const event = $lastCanvasProgressEvent.get();
      if (!event) {
        return;
      }
      const { session_id, batch_id, destination } = event;
      if (getIsCancelled({ session_id, batch_id, destination })) {
        $lastCanvasProgressEvent.set(null);
      }
    },
  });

  startAppListening({
    matcher: queueApi.endpoints.clearQueue.matchFulfilled,
    effect: () => {
      cancellations.clearQueue = true;
      const event = $lastCanvasProgressEvent.get();
      if (!event) {
        return;
      }
      const { session_id, batch_id, destination } = event;
      if (getIsCancelled({ session_id, batch_id, destination })) {
        $lastCanvasProgressEvent.set(null);
      }
    },
  });
};
