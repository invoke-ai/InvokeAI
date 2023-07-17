import { log } from 'app/logging/useLogger';
import { resetCanvas } from 'features/canvas/store/canvasSlice';
import { controlNetReset } from 'features/controlNet/store/controlNetSlice';
import { selectNextImageToSelect } from 'features/gallery/store/gallerySelectors';
import {
  imageRemoved,
  imageSelected,
} from 'features/gallery/store/gallerySlice';
import {
  imageDeletionConfirmed,
  isModalOpenChanged,
} from 'features/imageDeletion/store/imageDeletionSlice';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { api } from 'services/api';
import { imageDeleted } from 'services/api/thunks/image';
import { startAppListening } from '..';
import { imagesApi } from 'services/api/endpoints/images';

const moduleLog = log.child({ namespace: 'image' });

/**
 * Called when the user requests an image deletion
 */
export const addRequestedImageDeletionListener = () => {
  startAppListening({
    actionCreator: imageDeletionConfirmed,
    effect: async (action, { dispatch, getState, condition }) => {
      const { imageDTO, imageUsage } = action.payload;

      dispatch(isModalOpenChanged(false));

      const { image_name } = imageDTO;

      const state = getState();
      const lastSelectedImage =
        state.gallery.selection[state.gallery.selection.length - 1];

      if (lastSelectedImage === image_name) {
        const newSelectedImageId = selectNextImageToSelect(state, image_name);

        if (newSelectedImageId) {
          dispatch(imageSelected(newSelectedImageId));
        } else {
          dispatch(imageSelected(null));
        }
      }

      // We need to reset the features where the image is in use - none of these work if their image(s) don't exist

      if (imageUsage.isCanvasImage) {
        dispatch(resetCanvas());
      }

      if (imageUsage.isControlNetImage) {
        dispatch(controlNetReset());
      }

      if (imageUsage.isInitialImage) {
        dispatch(clearInitialImage());
      }

      if (imageUsage.isNodesImage) {
        dispatch(nodeEditorReset());
      }

      // Preemptively remove from gallery
      dispatch(imageRemoved(image_name));

      // Delete from server
      // const { requestId } = dispatch(imageDeleted({ image_name }));
      const { requestId } = dispatch(
        imagesApi.endpoints.deleteImage.initiate(imageDTO)
      );

      // Wait for successful deletion, then trigger boards to re-fetch
      const wasImageDeleted = await condition(
        (action) =>
          imagesApi.endpoints.deleteImage.matchFulfilled(action) &&
          action.meta.requestId === requestId,
        30000
      );

      if (wasImageDeleted) {
        dispatch(
          api.util.invalidateTags([{ type: 'Board', id: imageDTO.board_id }])
        );
      }
    },
  });
};

/**
 * Called when the actual delete request is sent to the server
 */
export const addImageDeletedPendingListener = () => {
  startAppListening({
    actionCreator: imageDeleted.pending,
    effect: (action, { dispatch, getState }) => {
      //
    },
  });
};

/**
 * Called on successful delete
 */
export const addImageDeletedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageDeleted.fulfilled,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug({ data: { image: action.meta.arg } }, 'Image deleted');
    },
  });
};

/**
 * Called on failed delete
 */
export const addImageDeletedRejectedListener = () => {
  startAppListening({
    actionCreator: imageDeleted.rejected,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(
        { data: { image: action.meta.arg } },
        'Unable to delete image'
      );
    },
  });
};
