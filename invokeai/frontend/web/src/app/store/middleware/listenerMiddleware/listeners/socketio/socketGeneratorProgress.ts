import { logger } from 'app/logging/logger';
import {
  appSocketGeneratorProgress,
  socketGeneratorProgress,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addGeneratorProgressEventListener = () => {
  startAppListening({
    actionCreator: socketGeneratorProgress,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');

      log.trace(action.payload, `Generator progress`);

      dispatch(appSocketGeneratorProgress(action.payload));
    },
  });
};
