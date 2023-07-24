import { logger } from 'app/logging/logger';
import { modelsApi } from 'services/api/endpoints/models';
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

      if (!nodes.schema && !disabledTabs.includes('nodes')) {
        dispatch(receivedOpenAPISchema());
      }

      // pass along the socket event as an application action
      dispatch(appSocketConnected(action.payload));

      // update all server state
      dispatch(modelsApi.endpoints.getMainModels.initiate());
      dispatch(modelsApi.endpoints.getControlNetModels.initiate());
      dispatch(modelsApi.endpoints.getLoRAModels.initiate());
      dispatch(modelsApi.endpoints.getTextualInversionModels.initiate());
      dispatch(modelsApi.endpoints.getVaeModels.initiate());
    },
  });
};
