import { logger } from 'app/logging/logger';
import {
  appSocketUnsubscribed,
  socketUnsubscribed,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addSocketUnsubscribedEventListener = () => {
  startAppListening({
    actionCreator: socketUnsubscribed,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      log.debug(action.payload, 'Unsubscribed');
      dispatch(appSocketUnsubscribed(action.payload));
    },
  });
};
