import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import {
  appSocketInvocationStarted,
  socketInvocationStarted,
} from 'services/events/actions';

const moduleLog = log.child({ namespace: 'socketio' });

export const addInvocationStartedEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationStarted,
    effect: (action, { dispatch, getState }) => {
      if (
        getState().system.canceledSession ===
        action.payload.data.graph_execution_state_id
      ) {
        moduleLog.trace(
          action.payload,
          'Ignored invocation started for canceled session'
        );
        return;
      }

      moduleLog.debug(
        action.payload,
        `Invocation started (${action.payload.data.node.type})`
      );
      dispatch(appSocketInvocationStarted(action.payload));
    },
  });
};
