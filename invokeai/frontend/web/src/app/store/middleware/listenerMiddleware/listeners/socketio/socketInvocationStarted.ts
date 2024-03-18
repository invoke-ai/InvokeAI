import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketInvocationStarted } from 'services/events/actions';

const log = logger('socketio');

export const addInvocationStartedEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketInvocationStarted,
    effect: (action) => {
      log.debug(action.payload, `Invocation started (${action.payload.data.node.type})`);
    },
  });
};
