import { logger } from 'app/logging/logger';
import {
  appSocketSessionRetrievalError,
  socketSessionRetrievalError,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addSessionRetrievalErrorEventListener = () => {
  startAppListening({
    actionCreator: socketSessionRetrievalError,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      log.error(
        action.payload,
        `Session retrieval error (${action.payload.data.graph_execution_state_id})`
      );
      dispatch(appSocketSessionRetrievalError(action.payload));
    },
  });
};
