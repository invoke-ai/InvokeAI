import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { receivedGalleryImages } from 'services/thunks/gallery';
import { serializeError } from 'serialize-error';

const moduleLog = log.child({ namespace: 'gallery' });

export const addReceivedGalleryImagesFulfilledListener = () => {
  startAppListening({
    actionCreator: receivedGalleryImages.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const page = action.payload;
      moduleLog.debug(
        { data: { page } },
        `Received ${page.items.length} gallery images`
      );
    },
  });
};

export const addReceivedGalleryImagesRejectedListener = () => {
  startAppListening({
    actionCreator: receivedGalleryImages.rejected,
    effect: (action, { getState, dispatch }) => {
      if (action.payload) {
        moduleLog.debug(
          { data: { error: serializeError(action.payload.error) } },
          'Problem receiving gallery images'
        );
      }
    },
  });
};
