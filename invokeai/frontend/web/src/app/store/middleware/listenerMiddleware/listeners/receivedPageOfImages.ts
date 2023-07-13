import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { serializeError } from 'serialize-error';
import { receivedPageOfImages } from 'services/api/thunks/image';
import { imagesApi } from 'services/api/endpoints/images';

const moduleLog = log.child({ namespace: 'gallery' });

export const addReceivedPageOfImagesFulfilledListener = () => {
  startAppListening({
    actionCreator: receivedPageOfImages.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const { items } = action.payload;
      moduleLog.debug(
        { data: { payload: action.payload } },
        `Received ${items.length} images`
      );

      items.forEach((image) => {
        dispatch(
          imagesApi.util.upsertQueryData('getImageDTO', image.image_name, image)
        );
      });
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
