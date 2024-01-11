import { logger } from 'app/logging/logger';
import { socketDisconnected } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addSocketDisconnectedEventListener = () => {
  startAppListening({
    actionCreator: socketDisconnected,
    effect: () => {
      log.debug('Disconnected');
    },
  });
};
