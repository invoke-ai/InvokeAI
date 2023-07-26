import { logger } from 'app/logging/logger';
import { appSocketSubscribed, socketSubscribed } from 'services/events/actions';
import { startAppListening } from '../..';

export const addSocketSubscribedEventListener = () => {
  startAppListening({
    actionCreator: socketSubscribed,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      log.debug(action.payload, 'Subscribed');
      dispatch(appSocketSubscribed(action.payload));
    },
  });
};
