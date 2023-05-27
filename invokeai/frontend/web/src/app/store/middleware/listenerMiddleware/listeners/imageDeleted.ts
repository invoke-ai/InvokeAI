import { requestedImageDeletion } from 'features/gallery/store/actions';
import { startAppListening } from '..';
import { imageDeleted } from 'services/thunks/image';
import { log } from 'app/logging/useLogger';
import { clamp } from 'lodash-es';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import {
  uploadRemoved,
  uploadsAdapter,
} from 'features/gallery/store/uploadsSlice';
import {
  resultRemoved,
  resultsAdapter,
} from 'features/gallery/store/resultsSlice';
import { isUploadsImageDTO } from 'services/types/guards';

const moduleLog = log.child({ namespace: 'addRequestedImageDeletionListener' });

/**
 * Called when the user requests an image deletion
 */
export const addRequestedImageDeletionListener = () => {
  startAppListening({
    actionCreator: requestedImageDeletion,
    effect: (action, { dispatch, getState }) => {
      const image = action.payload;
      if (!image) {
        moduleLog.warn('No image provided');
        return;
      }

      const { image_name, image_origin } = image;

      const state = getState();
      const selectedImage = state.gallery.selectedImage;
      const isUserImage = isUploadsImageDTO(selectedImage);
      if (selectedImage && selectedImage.image_name === image_name) {
        const allIds = isUserImage ? state.uploads.ids : state.results.ids;

        const allEntities = isUserImage
          ? state.uploads.entities
          : state.results.entities;

        const deletedImageIndex = allIds.findIndex(
          (result) => result.toString() === image_name
        );

        const filteredIds = allIds.filter((id) => id.toString() !== image_name);

        const newSelectedImageIndex = clamp(
          deletedImageIndex,
          0,
          filteredIds.length - 1
        );

        const newSelectedImageId = filteredIds[newSelectedImageIndex];

        const newSelectedImage = allEntities[newSelectedImageId];

        if (newSelectedImageId) {
          dispatch(imageSelected(newSelectedImage));
        } else {
          dispatch(imageSelected());
        }
      }

      if (isUserImage) {
        dispatch(uploadRemoved(image_name));
      } else {
        dispatch(resultRemoved(image_name));
      }

      dispatch(
        imageDeleted({ imageName: image_name, imageOrigin: image_origin })
      );
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
      const { imageName, imageOrigin } = action.meta.arg;
      // Preemptively remove the image from the gallery
      if (imageOrigin === 'external') {
        uploadsAdapter.removeOne(getState().uploads, imageName);
      }
      if (imageOrigin === 'internal') {
        resultsAdapter.removeOne(getState().results, imageName);
      }
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
