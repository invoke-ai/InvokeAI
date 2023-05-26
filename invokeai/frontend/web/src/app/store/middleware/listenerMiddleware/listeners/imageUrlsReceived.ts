import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { imageUrlsReceived } from 'services/thunks/image';
import { resultsAdapter } from 'features/gallery/store/resultsSlice';
import { uploadsAdapter } from 'features/gallery/store/uploadsSlice';

const moduleLog = log.child({ namespace: 'image' });

export const addImageUrlsReceivedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageUrlsReceived.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const image = action.payload;
      moduleLog.debug({ data: { image } }, 'Image URLs received');

      const { image_type, image_name, image_url, thumbnail_url } = image;

      if (image_type === 'results') {
        resultsAdapter.updateOne(getState().results, {
          id: image_name,
          changes: {
            image_url,
            thumbnail_url,
          },
        });
      }

      if (image_type === 'uploads') {
        uploadsAdapter.updateOne(getState().uploads, {
          id: image_name,
          changes: {
            image_url,
            thumbnail_url,
          },
        });
      }
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
