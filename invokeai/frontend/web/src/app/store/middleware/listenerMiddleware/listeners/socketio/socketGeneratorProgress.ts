import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketGeneratorProgress } from 'services/events/actions';

const log = logger('socketio');

export const addGeneratorProgressEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketGeneratorProgress,
    effect: (action) => {
      log.trace(action.payload, `Generator progress`);
    },
  });
};
