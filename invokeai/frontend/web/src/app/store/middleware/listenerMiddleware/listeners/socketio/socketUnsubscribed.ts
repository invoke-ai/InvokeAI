import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketUnsubscribedSession } from 'services/events/actions';
const log = logger('socketio');

export const addSocketUnsubscribedEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketUnsubscribedSession,
    effect: (action) => {
      log.debug(action.payload, 'Unsubscribed');
    },
  });
};
