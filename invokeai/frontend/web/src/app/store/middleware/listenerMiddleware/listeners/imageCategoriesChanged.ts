import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { receivedPageOfImages } from 'services/thunks/image';
import {
  imageCategoriesChanged,
  selectFilteredImagesAsArray,
} from 'features/gallery/store/imagesSlice';

const moduleLog = log.child({ namespace: 'gallery' });

export const addImageCategoriesChangedListener = () => {
  startAppListening({
    actionCreator: imageCategoriesChanged,
    effect: (action, { getState, dispatch }) => {
      const filteredImagesCount = selectFilteredImagesAsArray(
        getState()
      ).length;

      if (!filteredImagesCount) {
        dispatch(receivedPageOfImages());
      }
    },
  });
};
