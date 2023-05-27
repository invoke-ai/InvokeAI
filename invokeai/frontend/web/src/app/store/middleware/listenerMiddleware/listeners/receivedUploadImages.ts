import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { receivedUploadImages } from 'services/thunks/gallery';
import { serializeError } from 'serialize-error';

const moduleLog = log.child({ namespace: 'gallery' });

export const addReceivedUploadImagesPageFulfilledListener = () => {
  startAppListening({
    actionCreator: receivedUploadImages.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const page = action.payload;
      moduleLog.debug(
        { data: { page } },
        `Received ${page.items.length} uploaded images`
      );
    },
  });
};

export const addReceivedUploadImagesPageRejectedListener = () => {
  startAppListening({
    actionCreator: receivedUploadImages.rejected,
    effect: (action, { getState, dispatch }) => {
      if (action.payload) {
        moduleLog.debug(
          { data: { error: serializeError(action.payload.error) } },
          'Problem receiving uploaded images'
        );
      }
    },
  });
};
