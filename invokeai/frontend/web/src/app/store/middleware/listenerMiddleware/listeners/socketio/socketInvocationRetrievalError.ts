import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketInvocationRetrievalError } from 'services/events/actions';

const log = logger('socketio');

export const addInvocationRetrievalErrorEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketInvocationRetrievalError,
    effect: (action) => {
      log.error(action.payload, `Invocation retrieval error (${action.payload.data.graph_execution_state_id})`);
    },
  });
};
