import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';
import { selectionChanged } from '../../../../../features/gallery/store/gallerySlice';
import { ImageDTO } from '../../../../../services/api/types';

export const addImagesStarredListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.starImages.matchFulfilled,
    effect: async (action, { dispatch, getState }) => {
      const { updated_image_names: starredImages } = action.payload;

      const state = getState();

      const { selection } = state.gallery;
      const updatedSelection: ImageDTO[] = [];

      selection.forEach((selectedImageDTO) => {
        if (starredImages.includes(selectedImageDTO.image_name)) {
          updatedSelection.push({
            ...selectedImageDTO,
            starred: true,
          });
        } else {
          updatedSelection.push(selectedImageDTO);
        }
      });
      dispatch(selectionChanged(updatedSelection));
    },
  });
};
