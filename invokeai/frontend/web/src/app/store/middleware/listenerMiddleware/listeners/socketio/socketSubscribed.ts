import { logger } from 'app/logging/logger';
import {
  appSocketSubscribedSession,
  socketSubscribedSession,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addSocketSubscribedEventListener = () => {
  startAppListening({
    actionCreator: socketSubscribedSession,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      log.debug(action.payload, 'Subscribed');
      dispatch(appSocketSubscribedSession(action.payload));
    },
  });
};
