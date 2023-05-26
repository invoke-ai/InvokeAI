import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import { generatorProgress } from 'services/events/actions';

const moduleLog = log.child({ namespace: 'socketio' });

export const addGeneratorProgressListener = () => {
  startAppListening({
    actionCreator: generatorProgress,
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
    },
  });
};
