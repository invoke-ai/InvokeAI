import { logger } from 'app/logging/logger';
import {
  appSocketInvocationStarted,
  socketInvocationStarted,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addInvocationStartedEventListener = () => {
  startAppListening({
    actionCreator: socketInvocationStarted,
    effect: (action, { dispatch, getState }) => {
      const log = logger('socketio');
      if (
        getState().system.canceledSession ===
        action.payload.data.graph_execution_state_id
      ) {
        log.trace(
          action.payload,
          'Ignored invocation started for canceled session'
        );
        return;
      }

      log.debug(
        action.payload,
        `Invocation started (${action.payload.data.node.type})`
      );
      dispatch(appSocketInvocationStarted(action.payload));
    },
  });
};
