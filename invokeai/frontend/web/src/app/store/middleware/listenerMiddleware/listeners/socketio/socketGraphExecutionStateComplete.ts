import { log } from 'app/logging/useLogger';
import {
  appSocketGraphExecutionStateComplete,
  socketGraphExecutionStateComplete,
} from 'services/events/actions';
import { startAppListening } from '../..';

const moduleLog = log.child({ namespace: 'socketio' });

export const addGraphExecutionStateCompleteEventListener = () => {
  startAppListening({
    actionCreator: socketGraphExecutionStateComplete,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(
        action.payload,
        `Session invocation complete (${action.payload.data.graph_execution_state_id})`
      );
      // pass along the socket event as an application action
      dispatch(appSocketGraphExecutionStateComplete(action.payload));
    },
  });
};
