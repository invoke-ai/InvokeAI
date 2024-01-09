import { logger } from 'app/logging/logger';
import { $baseUrl } from 'app/store/nanostores/baseUrl';
import { size } from 'lodash-es';
import { atom } from 'nanostores';
import { api } from 'services/api';
import { queueApi, selectQueueStatus } from 'services/api/endpoints/queue';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { socketConnected } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

const $isFirstConnection = atom(true);

export const addSocketConnectedEventListener = () => {
  startAppListening({
    actionCreator: socketConnected,
    effect: async (
      action,
      { dispatch, getState, cancelActiveListeners, delay }
    ) => {
      log.debug('Connected');

      /**
       * The rest of this listener has recovery logic for when the socket disconnects and reconnects.
       * If we had generations in progress, we need to reset the API state to re-fetch everything.
       */

      if ($isFirstConnection.get()) {
        // Bail on the recovery logic if this is the first connection
        $isFirstConnection.set(false);
        return;
      }

      const { data: prevQueueStatusData } = selectQueueStatus(getState());

      // Bail if queue was empty before - we have nothing to recover
      if (
        !prevQueueStatusData?.queue.in_progress &&
        !prevQueueStatusData?.queue.pending
      ) {
        return;
      }

      // Else, we need to re-fetch the queue status to see if it has changed

      if ($baseUrl.get()) {
        // If we have a baseUrl (e.g. not localhost), we need to debounce the re-fetch to not hammer server
        cancelActiveListeners();
        // Add artificial jitter to the debounce
        await delay(1000 + Math.random() * 1000);
      }

      try {
        // Fetch the queue status again
        const queueStatusRequest = dispatch(
          await queueApi.endpoints.getQueueStatus.initiate(undefined, {
            forceRefetch: true,
          })
        );

        const nextQueueStatusData = await queueStatusRequest.unwrap();
        queueStatusRequest.unsubscribe();

        // If we haven't completed more items since the last check, bail
        if (
          prevQueueStatusData.queue.completed ===
          nextQueueStatusData.queue.completed
        ) {
          return;
        }

        /**
         * Else, we need to reset the API state to update everything.
         * 
         * TODO: This is rather inefficient. We don't actually need to re-fetch *all* network requests,
         * but determining which ones to re-fetch is non-trivial. It's at least the queue related ones
         * and gallery, but likely others. We'd also need to keep track of which requests need to be
         * re-fetch in this situation, which opens the door for bugs.
         * 
         * Optimize this later.
         */
        dispatch(api.util.resetApiState());
      } catch {
        // no-op
        log.debug('Unable to get current queue status on reconnect');
      }
    },
  });

  startAppListening({
    actionCreator: socketConnected,
    effect: async (action, { dispatch, getState }) => {
      const { nodeTemplates, config } = getState();
      if (
        !size(nodeTemplates.templates) &&
        !config.disabledTabs.includes('nodes')
      ) {
        // This request is a createAsyncThunk - resetting API state as in the above listener
        // will not trigger this request, so we need to manually do it.
        dispatch(receivedOpenAPISchema());
      }
    },
  });
};
