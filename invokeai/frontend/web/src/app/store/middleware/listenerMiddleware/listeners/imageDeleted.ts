import { requestedImageDeletion } from 'features/gallery/store/actions';
import { startAppListening } from '..';
import { imageDeleted } from 'services/thunks/image';
import { log } from 'app/logging/useLogger';
import { clamp } from 'lodash-es';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import {
  imageRemoved,
  imagesAdapter,
  selectImagesEntities,
  selectImagesIds,
} from 'features/gallery/store/imagesSlice';

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

      if (selectedImage && selectedImage.image_name === image_name) {
        const ids = selectImagesIds(state);
        const entities = selectImagesEntities(state);

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

        const newSelectedImage = entities[newSelectedImageId];

        if (newSelectedImageId) {
          dispatch(imageSelected(newSelectedImage));
        } else {
          dispatch(imageSelected());
        }
      }

      dispatch(imageRemoved(image_name));

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
      imagesAdapter.removeOne(getState().images, imageName);
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
