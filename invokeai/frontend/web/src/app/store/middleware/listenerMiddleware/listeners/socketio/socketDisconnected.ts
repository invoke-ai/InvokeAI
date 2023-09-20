import { logger } from 'app/logging/logger';
import { api } from 'services/api';
import {
  appSocketDisconnected,
  socketDisconnected,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addSocketDisconnectedEventListener = () => {
  startAppListening({
    actionCreator: socketDisconnected,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      log.debug('Disconnected');

      dispatch(api.util.resetApiState());

      // pass along the socket event as an application action
      dispatch(appSocketDisconnected(action.payload));
    },
  });
};
