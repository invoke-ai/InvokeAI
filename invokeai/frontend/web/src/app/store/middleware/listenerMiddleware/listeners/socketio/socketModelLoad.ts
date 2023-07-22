import { logger } from 'app/logging/logger';
import { ModelType } from 'services/api/types';
import {
  appSocketModelLoadCompleted,
  appSocketModelLoadStarted,
  socketModelLoadCompleted,
  socketModelLoadStarted,
} from 'services/events/actions';
import { startAppListening } from '../..';

const MODEL_TYPES: Record<ModelType, string> = {
  main: 'main',
  vae: 'VAE',
  lora: 'LoRA',
  controlnet: 'ControlNet',
  embedding: 'embedding',
};

export const addModelLoadEventListener = () => {
  startAppListening({
    actionCreator: socketModelLoadStarted,
    effect: (action, { dispatch, getState }) => {
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
    effect: (action, { dispatch, getState }) => {
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
