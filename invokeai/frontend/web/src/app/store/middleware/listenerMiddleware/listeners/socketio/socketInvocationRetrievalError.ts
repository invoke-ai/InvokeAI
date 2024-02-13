import { logger } from 'app/logging/logger';
import { socketInvocationRetrievalError } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addInvocationRetrievalErrorEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationRetrievalError,
    effect: (action) => {
      log.error(action.payload, `Invocation retrieval error (${action.payload.data.graph_execution_state_id})`);
    },
  });
};
