import { requestedImageDeletion } from 'features/gallery/store/actions';
import { startAppListening } from '..';
import { imageDeleted } from 'services/api/thunks/image';
import { log } from 'app/logging/useLogger';
import { clamp } from 'lodash-es';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import {
  imageRemoved,
  selectImagesIds,
} from 'features/gallery/store/imagesSlice';
import { resetCanvas } from 'features/canvas/store/canvasSlice';
import { controlNetReset } from 'features/controlNet/store/controlNetSlice';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { api } from 'services/api';

const moduleLog = log.child({ namespace: 'image' });

/**
 * Called when the user requests an image deletion
 */
export const addRequestedImageDeletionListener = () => {
  startAppListening({
    actionCreator: requestedImageDeletion,
    effect: async (action, { dispatch, getState, condition }) => {
      const { image, imageUsage } = action.payload;

      const { image_name } = image;

      const state = getState();
      const selectedImage = state.gallery.selectedImage;

      if (selectedImage === image_name) {
        const ids = selectImagesIds(state);

        const deletedImageIndex = ids.findIndex(
          (result) => result.toString() === image_name
        );

        const filteredIds = ids.filter((id) => id.toString() !== image_name);

        const newSelectedImageIndex = clamp(
          deletedImageIndex,
          0,
          filteredIds.length - 1
        );

        const newSelectedImageId = filteredIds[newSelectedImageIndex];

        if (newSelectedImageId) {
          dispatch(imageSelected(newSelectedImageId as string));
        } else {
          dispatch(imageSelected());
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
      const { requestId } = dispatch(imageDeleted({ image_name }));

      // Wait for successful deletion, then trigger boards to re-fetch
      const wasImageDeleted = await condition(
        (action): action is ReturnType<typeof imageDeleted.fulfilled> =>
          imageDeleted.fulfilled.match(action) &&
          action.meta.requestId === requestId,
        30000
      );

      if (wasImageDeleted) {
        dispatch(
          api.util.invalidateTags([{ type: 'Board', id: image.board_id }])
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
