import { logger } from 'app/logging/logger';
import {
  appSocketInvocationError,
  socketInvocationError,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addInvocationErrorEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationError,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      log.error(
        action.payload,
        `Invocation error (${action.payload.data.node.type})`
      );
      dispatch(appSocketInvocationError(action.payload));
    },
  });
};
