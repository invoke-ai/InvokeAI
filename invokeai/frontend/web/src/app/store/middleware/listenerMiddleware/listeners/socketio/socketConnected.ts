import { logger } from 'app/logging/logger';
import { isInitializedChanged } from 'features/system/store/systemSlice';
import { size } from 'lodash-es';
import { api } from 'services/api';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { socketConnected } from 'services/events/actions';

import { startAppListening } from '../..';

const log = logger('socketio');

export const addSocketConnectedEventListener = () => {
  startAppListening({
    actionCreator: socketConnected,
    effect: (action, { dispatch, getState }) => {
      log.debug('Connected');

      const { nodeTemplates, config, system } = getState();

      const { disabledTabs } = config;

      if (!size(nodeTemplates.templates) && !disabledTabs.includes('nodes')) {
        dispatch(receivedOpenAPISchema());
      }

      if (system.isInitialized) {
        // only reset the query caches if this connect event is a *reconnect* event
        dispatch(api.util.resetApiState());
      } else {
        dispatch(isInitializedChanged(true));
      }
    },
  });
};
