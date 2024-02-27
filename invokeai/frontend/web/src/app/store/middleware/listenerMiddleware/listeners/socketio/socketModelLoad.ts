import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketModelLoadCompleted, socketModelLoadStarted } from 'services/events/actions';

const log = logger('socketio');

export const addModelLoadEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketModelLoadStarted,
    effect: (action) => {
      const { base_model, model_name, model_type, submodel } = action.payload.data;

      let message = `Model load started: ${base_model}/${model_type}/${model_name}`;

      if (submodel) {
        message = message.concat(`/${submodel}`);
      }

      log.debug(action.payload, message);
    },
  });

  startAppListening({
    actionCreator: socketModelLoadCompleted,
    effect: (action) => {
      const { base_model, model_name, model_type, submodel } = action.payload.data;

      let message = `Model load complete: ${base_model}/${model_type}/${model_name}`;

      if (submodel) {
        message = message.concat(`/${submodel}`);
      }

      log.debug(action.payload, message);
    },
  });
};
