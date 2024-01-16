import { logger } from 'app/logging/logger';
import { socketGeneratorProgress } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addGeneratorProgressEventListener = () => {
  startAppListening({
    actionCreator: socketGeneratorProgress,
    effect: (action) => {
      log.trace(action.payload, `Generator progress`);
    },
  });
};
