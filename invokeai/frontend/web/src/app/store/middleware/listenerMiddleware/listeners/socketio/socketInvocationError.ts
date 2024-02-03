import { logger } from 'app/logging/logger';
import { socketInvocationError } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addInvocationErrorEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationError,
    effect: (action) => {
      log.error(action.payload, `Invocation error (${action.payload.data.node.type})`);
    },
  });
};
