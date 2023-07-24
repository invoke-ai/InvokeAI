import { logger } from 'app/logging/logger';
import {
  appSocketGraphExecutionStateComplete,
  socketGraphExecutionStateComplete,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addGraphExecutionStateCompleteEventListener = () => {
  startAppListening({
    actionCreator: socketGraphExecutionStateComplete,
    effect: (action, { dispatch }) => {
      const log = logger('socketio');
      log.debug(action.payload, 'Session complete');
      // pass along the socket event as an application action
      dispatch(appSocketGraphExecutionStateComplete(action.payload));
    },
  });
};
