import { logger } from 'app/logging/logger';
import { socketUnsubscribedSession } from 'services/events/actions';

import { startAppListening } from '../..';
const log = logger('socketio');

export const addSocketUnsubscribedEventListener = () => {
  startAppListening({
    actionCreator: socketUnsubscribedSession,
    effect: (action) => {
      log.debug(action.payload, 'Unsubscribed');
    },
  });
};
