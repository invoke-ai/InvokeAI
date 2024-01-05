import { logger } from 'app/logging/logger';
import { size } from 'lodash-es';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { appSocketConnected, socketConnected } from 'services/events/actions';

import { startAppListening } from '../..';

export const addSocketConnectedEventListener = () => {
  startAppListening({
    actionCreator: socketConnected,
    effect: (action, { dispatch, getState }) => {
      const log = logger('socketio');

      log.debug('Connected');

      const { nodeTemplates, config } = getState();

      const { disabledTabs } = config;

      if (!size(nodeTemplates.templates) && !disabledTabs.includes('nodes')) {
        dispatch(receivedOpenAPISchema());
      }

      // pass along the socket event as an application action
      dispatch(appSocketConnected(action.payload));
    },
  });
};
