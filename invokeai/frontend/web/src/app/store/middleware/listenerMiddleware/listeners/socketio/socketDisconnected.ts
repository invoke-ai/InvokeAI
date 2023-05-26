import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import { socketDisconnected } from 'services/events/actions';

const moduleLog = log.child({ namespace: 'socketio' });

export const addSocketDisconnectedListener = () => {
  startAppListening({
    actionCreator: socketDisconnected,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(action.payload, 'Disconnected');
    },
  });
};
