import { logger } from 'app/logging/logger';
import {
  appSocketGeneratorProgress,
  socketGeneratorProgress,
} from 'services/events/actions';
import { startAppListening } from '../..';

export const addGeneratorProgressEventListener = () => {
  startAppListening({
    actionCreator: socketGeneratorProgress,
    effect: (action, { dispatch, getState }) => {
      const log = logger('socketio');
      if (
        getState().system.canceledSession ===
        action.payload.data.graph_execution_state_id
      ) {
        log.trace(
          action.payload,
          'Ignored generator progress for canceled session'
        );
        return;
      }

      log.trace(
        action.payload,
        `Generator progress (${action.payload.data.node.type})`
      );

      // pass along the socket event as an application action
      dispatch(appSocketGeneratorProgress(action.payload));
    },
  });
};
