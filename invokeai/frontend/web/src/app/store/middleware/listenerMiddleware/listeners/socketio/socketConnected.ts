import { logger } from 'app/logging/logger';
import { $baseUrl } from 'app/store/nanostores/baseUrl';
import { isEqual, size } from 'lodash-es';
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
       *
       * If the queue totals have changed while we were disconnected, re-fetch everything, updating
       * the gallery and queue to recover.
       */

      // Bail on the recovery logic if this is the first connection - we don't need to recover anything
      if ($isFirstConnection.get()) {
        $isFirstConnection.set(false);
        return;
      }

      /**
       * Else, we need to compare the last-known queue status with the current queue status, re-fetching
       * everything if it has changed.
       */

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

        // If the queue hasn't changed, we don't need to recover
        if (isEqual(prevQueueStatusData?.queue, nextQueueStatusData.queue)) {
          return;
        }

        /**
         * The queue has changed. We need to reset the API state to update everything and recover
         * from the disconnect.
         *
         * TODO: This is rather inefficient. We don't actually need to re-fetch *all* queries, but
         * determining which queries to re-fetch and how to re-initialize them is non-trivial:
         *
         * - We need to keep track of which queries might have different data in this scenario. This
         * could be handled via tags, but it feels risky - if we miss tagging a critical query, we
         * could end up with a de-sync'd UI.
         *
         * - We need to re-initialize the queries with *the right query args*. This is very tricky,
         * because the query args are not stored in the API state, but rather in the component state.
         *
         * By totally resetting the API state, we also re-fetch things like model lists, which is
         * probably a good idea anyways.
         *
         * PS: RTKQ provides a related abstraction for recovery:
         * https://redux-toolkit.js.org/rtk-query/api/setupListeners
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
      // We only want to re-fetch the schema if we don't have any node templates
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
