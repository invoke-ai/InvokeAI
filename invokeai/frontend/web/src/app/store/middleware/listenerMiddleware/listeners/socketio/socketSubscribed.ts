import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import { socketSubscribed } from 'services/events/actions';

const moduleLog = log.child({ namespace: 'socketio' });

export const addSocketSubscribedListener = () => {
  startAppListening({
    actionCreator: socketSubscribed,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(
        action.payload,
        `Subscribed (${action.payload.sessionId}))`
      );
    },
  });
};
