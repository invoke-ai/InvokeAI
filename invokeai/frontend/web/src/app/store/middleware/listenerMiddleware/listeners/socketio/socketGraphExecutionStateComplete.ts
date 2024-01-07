import { logger } from 'app/logging/logger';
import { socketGraphExecutionStateComplete } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addGraphExecutionStateCompleteEventListener = () => {
  startAppListening({
    actionCreator: socketGraphExecutionStateComplete,
    effect: (action) => {
      log.debug(action.payload, 'Session complete');
    },
  });
};
