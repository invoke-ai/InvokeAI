import { log } from 'app/logging/useLogger';
import {
  imageCategoriesChanged,
  selectFilteredImages,
} from 'features/gallery/store/gallerySlice';
import { receivedPageOfImages } from 'services/api/thunks/image';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'gallery' });

export const addImageCategoriesChangedListener = () => {
  startAppListening({
    actionCreator: imageCategoriesChanged,
    effect: (action, { getState, dispatch }) => {
      const state = getState();
      const filteredImagesCount = selectFilteredImages(state).length;

      if (!filteredImagesCount) {
        dispatch(
          receivedPageOfImages({
            categories: action.payload,
            board_id: state.gallery.selectedBoardId,
            is_intermediate: false,
          })
        );
      }
    },
  });
};
