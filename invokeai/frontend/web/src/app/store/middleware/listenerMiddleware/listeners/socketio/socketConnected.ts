import { log } from 'app/logging/useLogger';
import { appSocketConnected, socketConnected } from 'services/events/actions';
import { receivedPageOfImages } from 'services/api/thunks/image';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { startAppListening } from '../..';

const moduleLog = log.child({ namespace: 'socketio' });

export const addSocketConnectedEventListener = () => {
  startAppListening({
    actionCreator: socketConnected,
    effect: (action, { dispatch, getState }) => {
      const { timestamp } = action.payload;

      moduleLog.debug({ timestamp }, 'Connected');

      const { nodes, config, images } = getState();

      const { disabledTabs } = config;

      if (!images.ids.length) {
        dispatch(
          receivedPageOfImages({
            categories: ['general'],
            is_intermediate: false,
          })
        );
      }

      if (!nodes.schema && !disabledTabs.includes('nodes')) {
        dispatch(receivedOpenAPISchema());
      }

      // pass along the socket event as an application action
      dispatch(appSocketConnected(action.payload));
    },
  });
};
