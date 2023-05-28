import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import { socketConnected } from 'services/events/actions';
import { receivedPageOfImages } from 'services/thunks/image';
import { receivedModels } from 'services/thunks/model';
import { receivedOpenAPISchema } from 'services/thunks/schema';

const moduleLog = log.child({ namespace: 'socketio' });

export const addSocketConnectedListener = () => {
  startAppListening({
    actionCreator: socketConnected,
    effect: (action, { dispatch, getState }) => {
      const { timestamp } = action.payload;

      moduleLog.debug({ timestamp }, 'Connected');

      const { models, nodes, config, images } = getState();

      const { disabledTabs } = config;

      if (!images.ids.length) {
        dispatch(receivedPageOfImages());
      }

      if (!models.ids.length) {
        dispatch(receivedModels());
      }

      if (!nodes.schema && !disabledTabs.includes('nodes')) {
        dispatch(receivedOpenAPISchema());
      }
    },
  });
};
