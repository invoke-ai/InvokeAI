import { log } from 'app/logging/useLogger';
import {
  appSocketModelLoadCompleted,
  socketModelLoadCompleted,
} from 'services/events/actions';
import { startAppListening } from '../..';

const moduleLog = log.child({ namespace: 'socketio' });

export const addModelLoadCompletedEventListener = () => {
  startAppListening({
    actionCreator: socketModelLoadCompleted,
    effect: (action, { dispatch, getState }) => {
      const { model_name, model_type, submodel } = action.payload.data;

      let modelString = `${model_type} model: ${model_name}`;

      if (submodel) {
        modelString = modelString.concat(`, submodel: ${submodel}`);
      }

      moduleLog.debug(action.payload, `Model load completed (${modelString})`);

      // pass along the socket event as an application action
      dispatch(appSocketModelLoadCompleted(action.payload));
    },
  });
};
