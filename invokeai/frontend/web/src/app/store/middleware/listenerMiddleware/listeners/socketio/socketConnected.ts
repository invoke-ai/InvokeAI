import { log } from 'app/logging/useLogger';
import { appSocketConnected, socketConnected } from 'services/events/actions';
import { receivedPageOfImages } from 'services/thunks/image';
import { getModels } from 'services/thunks/model';
import { receivedOpenAPISchema } from 'services/thunks/schema';
import { startAppListening } from '../..';

const moduleLog = log.child({ namespace: 'socketio' });

export const addSocketConnectedEventListener = () => {
  startAppListening({
    actionCreator: socketConnected,
    effect: (action, { dispatch, getState }) => {
      const { timestamp } = action.payload;

      moduleLog.debug({ timestamp }, 'Connected');

      const { sd1pipelinemodels, sd2pipelinemodels, nodes, config, images } =
        getState();

      const { disabledTabs } = config;

      if (!images.ids.length) {
        dispatch(receivedPageOfImages());
      }

      if (!sd1pipelinemodels.ids.length) {
        dispatch(getModels({ baseModel: 'sd-1', modelType: 'pipeline' }));
      }

      if (!sd2pipelinemodels.ids.length) {
        dispatch(getModels({ baseModel: 'sd-2', modelType: 'pipeline' }));
      }

      if (!nodes.schema && !disabledTabs.includes('nodes')) {
        dispatch(receivedOpenAPISchema());
      }

      // pass along the socket event as an application action
      dispatch(appSocketConnected(action.payload));
    },
  });
};
