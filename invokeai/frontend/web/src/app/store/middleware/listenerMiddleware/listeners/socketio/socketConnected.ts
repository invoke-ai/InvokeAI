import { logger } from 'app/logging/logger';
import { size } from 'lodash-es';
import { api } from 'services/api';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { appSocketConnected, socketConnected } from 'services/events/actions';
import { startAppListening } from '../..';

export const addSocketConnectedEventListener = () => {
  startAppListening({
    actionCreator: socketConnected,
    effect: (action, { dispatch, getState }) => {
      const log = logger('socketio');

      log.debug('Connected');

      const { nodes, config } = getState();

      const { disabledTabs } = config;

      if (!size(nodes.nodeTemplates) && !disabledTabs.includes('nodes')) {
        dispatch(receivedOpenAPISchema());
      }

      dispatch(api.util.resetApiState());

      // pass along the socket event as an application action
      dispatch(appSocketConnected(action.payload));
    },
  });
};
