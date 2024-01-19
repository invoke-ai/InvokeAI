// socketUploadStarted.ts
import { logger } from 'app/logging/logger';
import { socketUploadStarted } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addSocketUploadStartedEventListener = () => {
  startAppListening({
    actionCreator: socketUploadStarted,
    effect: (action) => {
      log.trace(action.payload, `Upload started`);
    //   console.log(action.payload); // TODO: remove, debugging purposes
    },
  });
};
