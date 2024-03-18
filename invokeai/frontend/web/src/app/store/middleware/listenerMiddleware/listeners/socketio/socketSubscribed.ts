import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketSubscribedSession } from 'services/events/actions';

const log = logger('socketio');

export const addSocketSubscribedEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketSubscribedSession,
    effect: (action) => {
      log.debug(action.payload, 'Subscribed');
    },
  });
};
