import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import {
  appSocketGeneratorProgress,
  socketGeneratorProgress,
} from 'services/events/actions';

const moduleLog = log.child({ namespace: 'socketio' });

export const addGeneratorProgressEventListener = () => {
  startAppListening({
    actionCreator: socketGeneratorProgress,
    effect: (action, { dispatch, getState }) => {
      if (
        getState().system.canceledSession ===
        action.payload.data.graph_execution_state_id
      ) {
        moduleLog.trace(
          action.payload,
          'Ignored generator progress for canceled session'
        );
        return;
      }

      moduleLog.trace(
        action.payload,
        `Generator progress (${action.payload.data.node.type})`
      );

      // pass along the socket event as an application action
      dispatch(appSocketGeneratorProgress(action.payload));
    },
  });
};
