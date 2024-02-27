import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketGraphExecutionStateComplete } from 'services/events/actions';

const log = logger('socketio');

export const addGraphExecutionStateCompleteEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketGraphExecutionStateComplete,
    effect: (action) => {
      log.debug(action.payload, 'Session complete');
    },
  });
};
