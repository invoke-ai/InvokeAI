import { logger } from 'app/logging/logger';
import { socketInvocationStarted } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addInvocationStartedEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationStarted,
    effect: (action) => {
      log.debug(action.payload, `Invocation started (${action.payload.data.node.type})`);
    },
  });
};
