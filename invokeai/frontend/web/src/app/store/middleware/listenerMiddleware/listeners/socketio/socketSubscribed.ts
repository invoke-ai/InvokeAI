import { logger } from 'app/logging/logger';
import { socketSubscribedSession } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addSocketSubscribedEventListener = () => {
  startAppListening({
    actionCreator: socketSubscribedSession,
    effect: (action) => {
      log.debug(action.payload, 'Subscribed');
    },
  });
};
