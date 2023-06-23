import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { imageUrlsReceived } from 'services/api/thunks/image';
import { imageUpdatedOne } from 'features/gallery/store/imagesSlice';

const moduleLog = log.child({ namespace: 'image' });

export const addImageUrlsReceivedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageUrlsReceived.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const image = action.payload;
      moduleLog.debug({ data: { image } }, 'Image URLs received');

      const { image_name, image_url, thumbnail_url } = image;

      dispatch(
        imageUpdatedOne({
          id: image_name,
          changes: { image_url, thumbnail_url },
        })
      );
    },
  });
};

export const addImageUrlsReceivedRejectedListener = () => {
  startAppListening({
    actionCreator: imageUrlsReceived.rejected,
    effect: (action, { getState, dispatch }) => {
      moduleLog.debug(
        { data: { image: action.meta.arg } },
        'Problem getting image URLs'
      );
    },
  });
};
