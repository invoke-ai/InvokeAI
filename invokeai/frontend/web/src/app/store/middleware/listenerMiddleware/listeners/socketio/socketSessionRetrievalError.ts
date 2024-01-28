import { logger } from 'app/logging/logger';
import { socketSessionRetrievalError } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addSessionRetrievalErrorEventListener = () => {
  startAppListening({
    actionCreator: socketSessionRetrievalError,
    effect: (action) => {
      log.error(action.payload, `Session retrieval error (${action.payload.data.graph_execution_state_id})`);
    },
  });
};
