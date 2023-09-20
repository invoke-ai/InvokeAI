import { logger } from 'app/logging/logger';
import {
  appSocketUnsubscribedSession,
  socketUnsubscribedSession,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addSocketUnsubscribedEventListener = () => {
  startAppListening({
    actionCreator: socketUnsubscribedSession,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      log.debug(action.payload, 'Unsubscribed');
      dispatch(appSocketUnsubscribedSession(action.payload));
    },
  });
};
