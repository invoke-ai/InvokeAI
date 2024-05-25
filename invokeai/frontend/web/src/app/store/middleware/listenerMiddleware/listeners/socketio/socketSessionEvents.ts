import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketSessionCanceled, socketSessionComplete, socketSessionStarted } from 'services/events/actions';

const log = logger('socketio');

export const addSessionEventListeners = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketSessionStarted,
    effect: (action) => {
      log.debug(action.payload, 'Session started');
    },
  });
  startAppListening({
    actionCreator: socketSessionComplete,
    effect: (action) => {
      log.debug(action.payload, 'Session complete');
    },
  });
  startAppListening({
    actionCreator: socketSessionCanceled,
    effect: (action) => {
      log.debug(action.payload, 'Session canceled');
    },
  });
};
