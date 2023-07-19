import { log } from 'app/logging/useLogger';
import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
  boardIdSelected,
  imageSelected,
} from 'features/gallery/store/gallerySlice';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'boards' });

export const addBoardIdSelectedListener = () => {
  startAppListening({
    actionCreator: boardIdSelected,
    effect: (action, { getState, dispatch }) => {
      const board_id = action.payload;

      // we need to check if we need to fetch more images

      const state = getState();
      // const allImages = selectImagesAll(state);

      const categories =
        state.gallery.galleryView === 'images'
          ? IMAGE_CATEGORIES
          : ASSETS_CATEGORIES;

      if (board_id === 'images') {
        // Selected all images
        const { data: allImagesData } = imagesApi.endpoints.listImages.select({
          categories,
        })(state);
        if (allImagesData?.ids.length) {
          dispatch(imageSelected((allImagesData.ids[0] as string) ?? null));
        }
        return;
      }

      if (board_id === 'batch') {
        // Selected the batch
        // TODO
        return;
      }

      const { data: boardImagesData } = imagesApi.endpoints.listImages.select({
        board_id,
        categories,
      })(state);
      if (boardImagesData?.ids.length) {
        dispatch(imageSelected((boardImagesData.ids[0] as string) ?? null));
      }
      return;
    },
  });
};
