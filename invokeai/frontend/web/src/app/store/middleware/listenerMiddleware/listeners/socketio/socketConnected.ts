import { startAppListening } from '../..';
import { log } from 'app/logging/useLogger';
import { socketConnected } from 'services/events/actions';
import {
  receivedGalleryImages,
  receivedUploadImages,
} from 'services/thunks/gallery';
import { receivedModels } from 'services/thunks/model';
import { receivedOpenAPISchema } from 'services/thunks/schema';

const moduleLog = log.child({ namespace: 'socketio' });

export const addSocketConnectedListener = () => {
  startAppListening({
    actionCreator: socketConnected,
    effect: (action, { dispatch, getState }) => {
      const { timestamp } = action.payload;

      moduleLog.debug({ timestamp }, 'Connected');

      const { results, uploads, models, nodes, config } = getState();

      const { disabledTabs } = config;

      // These thunks need to be dispatch in middleware; cannot handle in a reducer
      if (!results.ids.length) {
        dispatch(receivedGalleryImages());
      }

      if (!uploads.ids.length) {
        dispatch(receivedUploadImages());
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
