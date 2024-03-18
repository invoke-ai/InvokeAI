import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketSessionRetrievalError } from 'services/events/actions';

const log = logger('socketio');

export const addSessionRetrievalErrorEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketSessionRetrievalError,
    effect: (action) => {
      log.error(action.payload, `Session retrieval error (${action.payload.data.graph_execution_state_id})`);
    },
  });
};
