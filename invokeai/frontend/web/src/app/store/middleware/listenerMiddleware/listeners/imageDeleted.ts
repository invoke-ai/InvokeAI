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

      const { name, type } = image;

      if (type !== 'uploads' && type !== 'results') {
        moduleLog.warn({ data: image }, `Invalid image type ${type}`);
        return;
      }

      const selectedImageName = getState().gallery.selectedImage?.name;

      if (selectedImageName === name) {
        const allIds = getState()[type].ids;
        const allEntities = getState()[type].entities;

        const deletedImageIndex = allIds.findIndex(
          (result) => result.toString() === name
        );

        const filteredIds = allIds.filter((id) => id.toString() !== name);

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

      dispatch(imageDeleted({ imageName: name, imageType: type }));
    },
  });
};
