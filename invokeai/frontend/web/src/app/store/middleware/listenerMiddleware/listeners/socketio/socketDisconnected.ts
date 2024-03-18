import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketDisconnected } from 'services/events/actions';

const log = logger('socketio');

export const addSocketDisconnectedEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketDisconnected,
    effect: () => {
      log.debug('Disconnected');
    },
  });
};
