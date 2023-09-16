import { logger } from 'app/logging/logger';
import { LIST_TAG } from 'services/api';
import { appInfoApi } from 'services/api/endpoints/appInfo';
import { modelsApi } from 'services/api/endpoints/models';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { appSocketConnected, socketConnected } from 'services/events/actions';
import { startAppListening } from '../..';
import { size } from 'lodash-es';

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

      // pass along the socket event as an application action
      dispatch(appSocketConnected(action.payload));

      // update all server state
      dispatch(
        modelsApi.util.invalidateTags([
          { type: 'MainModel', id: LIST_TAG },
          { type: 'SDXLRefinerModel', id: LIST_TAG },
          { type: 'LoRAModel', id: LIST_TAG },
          { type: 'ControlNetModel', id: LIST_TAG },
          { type: 'VaeModel', id: LIST_TAG },
          { type: 'TextualInversionModel', id: LIST_TAG },
          { type: 'ScannedModels', id: LIST_TAG },
        ])
      );
      dispatch(appInfoApi.util.invalidateTags(['AppConfig', 'AppVersion']));
    },
  });
};
