import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { serializeError } from 'serialize-error';
import { receivedPageOfImages } from 'services/thunks/image';

const moduleLog = log.child({ namespace: 'gallery' });

export const addReceivedPageOfImagesFulfilledListener = () => {
  startAppListening({
    actionCreator: receivedPageOfImages.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const page = action.payload;
      moduleLog.debug(
        { data: { payload: action.payload } },
        `Received ${page.items.length} images`
      );
    },
  });
};

export const addReceivedPageOfImagesRejectedListener = () => {
  startAppListening({
    actionCreator: receivedPageOfImages.rejected,
    effect: (action, { getState, dispatch }) => {
      if (action.payload) {
        moduleLog.debug(
          { data: { error: serializeError(action.payload) } },
          'Problem receiving images'
        );
      }
    },
  });
};
