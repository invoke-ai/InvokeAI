import { requestedImageDeletion } from 'features/gallery/store/actions';
import { startAppListening } from '..';
import { imageDeleted } from 'services/thunks/image';
import { log } from 'app/logging/useLogger';
import { clamp } from 'lodash-es';
import { imageSelected } from 'features/gallery/store/gallerySlice';

const moduleLog = log.child({ namespace: 'addRequestedImageDeletionListener' });

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

      if (image_type !== 'uploads' && image_type !== 'results') {
        moduleLog.warn({ data: image }, `Invalid image type ${image_type}`);
        return;
      }

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
