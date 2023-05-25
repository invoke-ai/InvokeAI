import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import { invocationError } from 'services/events/actions';

const moduleLog = log.child({ namespace: 'socketio' });

export const addInvocationErrorListener = () => {
  startAppListening({
    actionCreator: invocationError,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(
        action.payload,
        `Invocation error (${action.payload.data.node.type})`
      );
    },
  });
};
