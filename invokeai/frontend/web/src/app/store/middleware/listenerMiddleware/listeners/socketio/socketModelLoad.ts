import { logger } from 'app/logging/logger';
import {
  appSocketModelLoadCompleted,
  appSocketModelLoadStarted,
  socketModelLoadCompleted,
  socketModelLoadStarted,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addModelLoadEventListener = () => {
  startAppListening({
    actionCreator: socketModelLoadStarted,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      const { base_model, model_name, model_type, submodel } =
        action.payload.data;

      let message = `Model load started: ${base_model}/${model_type}/${model_name}`;

      if (submodel) {
        message = message.concat(`/${submodel}`);
      }

      log.debug(action.payload, message);

      // pass along the socket event as an application action
      dispatch(appSocketModelLoadStarted(action.payload));
    },
  });

  startAppListening({
    actionCreator: socketModelLoadCompleted,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      const { base_model, model_name, model_type, submodel } =
        action.payload.data;

      let message = `Model load complete: ${base_model}/${model_type}/${model_name}`;

      if (submodel) {
        message = message.concat(`/${submodel}`);
      }

      log.debug(action.payload, message);
      // pass along the socket event as an application action
      dispatch(appSocketModelLoadCompleted(action.payload));
    },
  });
};
