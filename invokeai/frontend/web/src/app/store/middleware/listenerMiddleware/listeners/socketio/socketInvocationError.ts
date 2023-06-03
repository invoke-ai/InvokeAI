import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import {
  appSocketInvocationError,
  socketInvocationError,
} from 'services/events/actions';

const moduleLog = log.child({ namespace: 'socketio' });

export const addInvocationErrorEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationError,
    effect: (action, { dispatch, getState }) => {
      moduleLog.error(
        action.payload,
        `Invocation error (${action.payload.data.node.type})`
      );
      dispatch(appSocketInvocationError(action.payload));
    },
  });
};
