import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketModelLoadCompleted, socketModelLoadStarted } from 'services/events/actions';

const log = logger('socketio');

export const addModelLoadEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketModelLoadStarted,
    effect: (action) => {
      const { model_config, submodel_type } = action.payload.data;
      const { name, base, type } = model_config;

      const extras: string[] = [base, type];
      if (submodel_type) {
        extras.push(submodel_type);
      }

      const message = `Model load started: ${name} (${extras.join(', ')})`;

      log.debug(action.payload, message);
    },
  });

  startAppListening({
    actionCreator: socketModelLoadCompleted,
    effect: (action) => {
      const { model_config, submodel_type } = action.payload.data;
      const { name, base, type } = model_config;

      const extras: string[] = [base, type];
      if (submodel_type) {
        extras.push(submodel_type);
      }

      const message = `Model load complete: ${name} (${extras.join(', ')})`;

      log.debug(action.payload, message);
    },
  });
};
