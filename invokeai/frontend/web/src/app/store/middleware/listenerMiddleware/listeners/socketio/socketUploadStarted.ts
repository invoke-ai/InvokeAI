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
<<<<<<< HEAD
      console.log(action.payload); // Log the payload for debugging
=======
    //   console.log(action.payload); // TODO: remove, debugging purposes
>>>>>>> f9a7a0639 (clean up before making a draft PR)
    },
  });
};
