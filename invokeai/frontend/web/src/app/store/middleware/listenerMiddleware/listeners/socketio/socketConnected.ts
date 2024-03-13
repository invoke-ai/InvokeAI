import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { $baseUrl } from 'app/store/nanostores/baseUrl';
import { isEqual } from 'lodash-es';
import { atom } from 'nanostores';
import { api } from 'services/api';
import { modelsApi } from 'services/api/endpoints/models';
import { queueApi, selectQueueStatus } from 'services/api/endpoints/queue';
import { socketConnected } from 'services/events/actions';

const log = logger('socketio');

const $isFirstConnection = atom(true);

export const addSocketConnectedEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketConnected,
    effect: async (action, { dispatch, getState, cancelActiveListeners, delay }) => {
      log.debug('Connected');

      /**
       * The rest of this listener has recovery logic for when the socket disconnects and reconnects.
       *
       * We need to re-fetch if something has changed while we were disconnected. In practice, the only
       * thing that could change while disconnected is a queue item finishes processing.
       *
       * The queue status is a proxy for this - if the queue status has changed, we need to re-fetch
       * the queries that may have changed while we were disconnected.
       */

      // Bail on the recovery logic if this is the first connection - we don't need to recover anything
      if ($isFirstConnection.get()) {
        // The TI models are used in a component that is not always rendered, so when users open the prompt triggers
        // box has a delay while it does the initial fetch. We need to both pre-fetch the data and maintain an RTK
        // Query subscription to it, so the cache doesn't clear itself when the user closes the prompt triggers box.
        // So, we explicitly do not unsubscribe from this query!
        dispatch(modelsApi.endpoints.getTextualInversionModels.initiate());

        $isFirstConnection.set(false);
        return;
      }

      // If we are in development mode, reset the whole API state. In this scenario, reconnects will
      // typically be caused by reloading the server, in which case we do want to reset the whole API.
      if (import.meta.env.MODE === 'development') {
        dispatch(api.util.resetApiState());
      }

      // Else, we need to compare the last-known queue status with the current queue status, re-fetching
      // everything if it has changed.

      if ($baseUrl.get()) {
        // If we have a baseUrl (e.g. not localhost), we need to debounce the re-fetch to not hammer server
        cancelActiveListeners();
        // Add artificial jitter to the debounce
        await delay(1000 + Math.random() * 1000);
      }

      const prevQueueStatusData = selectQueueStatus(getState()).data;

      try {
        // Fetch the queue status again
        const queueStatusRequest = dispatch(
          await queueApi.endpoints.getQueueStatus.initiate(undefined, {
            forceRefetch: true,
          })
        );
        const nextQueueStatusData = await queueStatusRequest.unwrap();
        queueStatusRequest.unsubscribe();

        // If the queue hasn't changed, we don't need to do anything.
        if (isEqual(prevQueueStatusData?.queue, nextQueueStatusData.queue)) {
          return;
        }

        //The queue has changed. We need to re-fetch everything that may have changed while we were
        // disconnected.
        dispatch(api.util.invalidateTags(['FetchOnReconnect']));
      } catch {
        // no-op
        log.debug('Unable to get current queue status on reconnect');
      }
    },
  });
};
