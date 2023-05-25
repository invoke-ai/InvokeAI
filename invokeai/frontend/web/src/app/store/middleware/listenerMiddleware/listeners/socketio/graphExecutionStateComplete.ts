import { log } from 'app/logging/useLogger';
import { graphExecutionStateComplete } from 'services/events/actions';
import { startAppListening } from '../..';

const moduleLog = log.child({ namespace: 'socketio' });

export const addGraphExecutionStateCompleteListener = () => {
  startAppListening({
    actionCreator: graphExecutionStateComplete,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(
        action.payload,
        `Graph execution state complete (${action.payload.data.graph_execution_state_id})`
      );
    },
  });
};
