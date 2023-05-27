import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { imageMetadataReceived } from 'services/thunks/image';
import { resultUpserted } from 'features/gallery/store/resultsSlice';
import { uploadUpserted } from 'features/gallery/store/uploadsSlice';
import { imageSelected } from 'features/gallery/store/gallerySlice';

const moduleLog = log.child({ namespace: 'image' });

export const addImageMetadataReceivedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageMetadataReceived.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const imageDTO = action.payload;
      moduleLog.debug({ data: { imageDTO } }, 'Image metadata received');

      if (imageDTO.image_origin === 'internal') {
        dispatch(resultUpserted(imageDTO));
      }

      if (imageDTO.image_origin === 'external') {
        dispatch(uploadUpserted(imageDTO));
      }
    },
  });
};

export const addImageMetadataReceivedRejectedListener = () => {
  startAppListening({
    actionCreator: imageMetadataReceived.rejected,
    effect: (action, { getState, dispatch }) => {
      moduleLog.debug(
        { data: { image: action.meta.arg } },
        'Problem receiving image metadata'
      );
    },
  });
};
