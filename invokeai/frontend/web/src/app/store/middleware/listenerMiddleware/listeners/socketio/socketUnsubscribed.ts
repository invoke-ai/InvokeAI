import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import { socketUnsubscribed } from 'services/events/actions';

const moduleLog = log.child({ namespace: 'socketio' });

export const addSocketUnsubscribedListener = () => {
  startAppListening({
    actionCreator: socketUnsubscribed,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(
        action.payload,
        `Unsubscribed (${action.payload.sessionId})`
      );
    },
  });
};
