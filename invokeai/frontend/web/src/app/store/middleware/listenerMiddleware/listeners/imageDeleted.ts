import { requestedImageDeletion } from 'features/gallery/store/actions';
import { startAppListening } from '..';
import { imageDeleted } from 'services/thunks/image';
import { log } from 'app/logging/useLogger';
import { clamp } from 'lodash-es';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { uploadsAdapter } from 'features/gallery/store/uploadsSlice';
import { resultsAdapter } from 'features/gallery/store/resultsSlice';

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

      const { image_name, image_type } = image;

      const selectedImageName = getState().gallery.selectedImage?.image_name;

      if (selectedImageName === image_name) {
        const allIds = getState()[image_type].ids;
        const allEntities = getState()[image_type].entities;

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

      dispatch(imageDeleted({ imageName: image_name, imageType: image_type }));
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
      const { imageName, imageType } = action.meta.arg;
      // Preemptively remove the image from the gallery
      if (imageType === 'uploads') {
        uploadsAdapter.removeOne(getState().uploads, imageName);
      }
      if (imageType === 'results') {
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
