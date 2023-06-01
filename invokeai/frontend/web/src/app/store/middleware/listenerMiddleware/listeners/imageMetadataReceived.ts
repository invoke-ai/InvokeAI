import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { imageMetadataReceived } from 'services/thunks/image';
import { imageUpserted } from 'features/gallery/store/imagesSlice';

const moduleLog = log.child({ namespace: 'image' });

export const addImageMetadataReceivedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageMetadataReceived.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const image = action.payload;
      if (image.is_intermediate) {
        // No further actions needed for intermediate images
        return;
      }
      moduleLog.debug({ data: { image } }, 'Image metadata received');
      dispatch(imageUpserted(image));
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
