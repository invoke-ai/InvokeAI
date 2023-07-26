import { logger } from 'app/logging/logger';
import { modelsApi } from 'services/api/endpoints/models';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { appSocketConnected, socketConnected } from 'services/events/actions';
import { startAppListening } from '../..';
import {
  ALL_BASE_MODELS,
  NON_REFINER_BASE_MODELS,
  REFINER_BASE_MODELS,
} from 'services/api/constants';

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
      dispatch(modelsApi.endpoints.getMainModels.initiate(REFINER_BASE_MODELS));
      dispatch(
        modelsApi.endpoints.getMainModels.initiate(NON_REFINER_BASE_MODELS)
      );
      dispatch(modelsApi.endpoints.getMainModels.initiate(ALL_BASE_MODELS));
      dispatch(modelsApi.endpoints.getControlNetModels.initiate());
      dispatch(modelsApi.endpoints.getLoRAModels.initiate());
      dispatch(modelsApi.endpoints.getTextualInversionModels.initiate());
      dispatch(modelsApi.endpoints.getVaeModels.initiate());
    },
  });
};
