import { logger } from 'app/logging/logger';
import {
  appSocketInvocationRetrievalError,
  socketInvocationRetrievalError,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addInvocationRetrievalErrorEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationRetrievalError,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      log.error(
        action.payload,
        `Invocation retrieval error (${action.payload.data.graph_execution_state_id})`
      );
      dispatch(appSocketInvocationRetrievalError(action.payload));
    },
  });
};
