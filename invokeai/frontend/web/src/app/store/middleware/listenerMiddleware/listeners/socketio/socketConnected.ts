import { logger } from 'app/logging/logger';
import { isInitializedChanged } from 'features/system/store/systemSlice';
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

      const { nodes, config, system } = getState();

      const { disabledTabs } = config;

      if (!size(nodes.nodeTemplates) && !disabledTabs.includes('nodes')) {
        dispatch(receivedOpenAPISchema());
      }

      if (system.isInitialized) {
        // only reset the query caches if this connect event is a *reconnect* event
        dispatch(api.util.resetApiState());
      } else {
        dispatch(isInitializedChanged(true));
      }

      // pass along the socket event as an application action
      dispatch(appSocketConnected(action.payload));
    },
  });
};
