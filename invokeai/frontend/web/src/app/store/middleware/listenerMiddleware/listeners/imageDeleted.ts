import { log } from 'app/logging/useLogger';
import { resetCanvas } from 'features/canvas/store/canvasSlice';
import { controlNetReset } from 'features/controlNet/store/controlNetSlice';
import { selectListImagesBaseQueryArgs } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import {
  imageDeletionConfirmed,
  isModalOpenChanged,
} from 'features/imageDeletion/store/imageDeletionSlice';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { clamp } from 'lodash-es';
import { api } from 'services/api';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';

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
        const baseQueryArgs = selectListImagesBaseQueryArgs(state);
        const { data } =
          imagesApi.endpoints.listImages.select(baseQueryArgs)(state);

        const ids = data?.ids ?? [];

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

      // Delete from server
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
    matcher: imagesApi.endpoints.deleteImage.matchPending,
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
    matcher: imagesApi.endpoints.deleteImage.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(
        { data: { image: action.meta.arg.originalArgs } },
        'Image deleted'
      );
    },
  });
};

/**
 * Called on failed delete
 */
export const addImageDeletedRejectedListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.deleteImage.matchRejected,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(
        { data: { image: action.meta.arg.originalArgs } },
        'Unable to delete image'
      );
    },
  });
};
