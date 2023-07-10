import { log } from 'app/logging/useLogger';
import { serializeError } from 'serialize-error';
import { imagesApi } from 'services/api/endpoints/images';
import { receivedListOfImages } from 'services/api/thunks/image';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'gallery' });

export const addReceivedListOfImagesListener = () => {
  startAppListening({
    actionCreator: receivedListOfImages.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const { image_dtos } = action.payload;
      moduleLog.debug(
        { data: { payload: action.payload } },
        `Received ${image_dtos.length} images`
      );

      image_dtos.forEach((image) => {
        dispatch(
          imagesApi.util.upsertQueryData('getImageDTO', image.image_name, image)
        );
      });
    },
  });

  startAppListening({
    actionCreator: receivedListOfImages.rejected,
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
