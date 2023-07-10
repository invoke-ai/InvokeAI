import { log } from 'app/logging/useLogger';
import { imageUpserted } from 'features/gallery/store/gallerySlice';
import { imageDTOReceived, imageUpdated } from 'services/api/thunks/image';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'image' });

export const addImageDTOReceivedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageDTOReceived.fulfilled,
    effect: (action, { getState, dispatch }) => {
      const image = action.payload;

      const state = getState();

      if (
        image.session_id === state.canvas.layerState.stagingArea.sessionId &&
        state.canvas.shouldAutoSave
      ) {
        dispatch(
          imageUpdated({
            image_name: image.image_name,
            is_intermediate: image.is_intermediate,
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

export const addImageDTOReceivedRejectedListener = () => {
  startAppListening({
    actionCreator: imageDTOReceived.rejected,
    effect: (action, { getState, dispatch }) => {
      moduleLog.debug(
        { data: { image: action.meta.arg } },
        'Problem receiving image metadata'
      );
    },
  });
};
