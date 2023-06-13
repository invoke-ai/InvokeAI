import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import {
  appSocketInvocationWarning,
  socketInvocationWarning,
} from 'services/events/actions';

const moduleLog = log.child({ namespace: 'socketio' });

export const addInvocationWarningEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationWarning,
    effect: (action, { dispatch, getState }) => {
      moduleLog.warn(
        action.payload,
        `Invocation warning (${action.payload.data.node.type})`
      );
      dispatch(appSocketInvocationWarning(action.payload));
    },
  });
};
