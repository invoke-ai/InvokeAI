import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { receivedUploadImagesPage } from 'services/thunks/gallery';
import { serializeError } from 'serialize-error';

const moduleLog = log.child({ namespace: 'gallery' });

export const addReceivedUploadImagesPageFulfilledListener = () => {
  startAppListening({
    actionCreator: receivedUploadImagesPage.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const page = action.payload;
      moduleLog.debug(
        { data: { page } },
        `Received ${page.items.length} uploads`
      );
    },
  });
};

export const addReceivedUploadImagesPageRejectedListener = () => {
  startAppListening({
    actionCreator: receivedUploadImagesPage.rejected,
    effect: (action, { getState, dispatch }) => {
      if (action.payload) {
        moduleLog.debug(
          { data: { error: serializeError(action.payload.error) } },
          'Problem receiving uploads'
        );
      }
    },
  });
};
