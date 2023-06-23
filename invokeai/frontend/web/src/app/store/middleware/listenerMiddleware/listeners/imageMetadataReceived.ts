import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { imageMetadataReceived, imageUpdated } from 'services/api/thunks/image';
import { imageUpserted } from 'features/gallery/store/imagesSlice';

const moduleLog = log.child({ namespace: 'image' });

export const addImageMetadataReceivedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageMetadataReceived.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const image = action.payload;

      const state = getState();

      if (
        image.session_id === state.canvas.layerState.stagingArea.sessionId &&
        state.canvas.shouldAutoSave
      ) {
        dispatch(
          imageUpdated({
            imageName: image.image_name,
            requestBody: { is_intermediate: false },
          })
        );
      } else if (image.is_intermediate) {
        // No further actions needed for intermediate images
        moduleLog.trace(
          { data: { image } },
          'Image metadata received (intermediate), skipping'
        );
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
