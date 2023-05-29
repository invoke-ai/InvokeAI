import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { imageUrlsReceived } from 'services/thunks/image';
import { imageUpdatedOne } from 'features/gallery/store/imagesSlice';

const moduleLog = log.child({ namespace: 'image' });

export const addImageUrlsReceivedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageUrlsReceived.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const image = action.payload;
      moduleLog.debug({ data: { image } }, 'Image URLs received');
      dispatch(imageUpdatedOne(image));
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
