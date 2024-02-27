import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketInvocationError } from 'services/events/actions';

const log = logger('socketio');

export const addInvocationErrorEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketInvocationError,
    effect: (action) => {
      log.error(action.payload, `Invocation error (${action.payload.data.node.type})`);
    },
  });
};
