import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import {
  appSocketUnsubscribed,
  socketUnsubscribed,
} from 'services/events/actions';

const moduleLog = log.child({ namespace: 'socketio' });

export const addSocketUnsubscribedEventListener = () => {
  startAppListening({
    actionCreator: socketUnsubscribed,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(
        action.payload,
        `Unsubscribed (${action.payload.sessionId})`
      );
      dispatch(appSocketUnsubscribed(action.payload));
    },
  });
};
