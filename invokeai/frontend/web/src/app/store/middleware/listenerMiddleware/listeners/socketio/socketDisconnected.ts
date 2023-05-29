import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import {
  socketDisconnected,
  appSocketDisconnected,
} from 'services/events/actions';

const moduleLog = log.child({ namespace: 'socketio' });

export const addSocketDisconnectedEventListener = () => {
  startAppListening({
    actionCreator: socketDisconnected,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(action.payload, 'Disconnected');
      // pass along the socket event as an application action
      dispatch(appSocketDisconnected(action.payload));
    },
  });
};
