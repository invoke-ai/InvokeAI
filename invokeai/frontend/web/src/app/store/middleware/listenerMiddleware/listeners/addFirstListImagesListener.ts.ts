import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { imagesApi } from 'services/api/endpoints/images';
import { getListImagesUrl } from 'services/api/util';

export const addFirstListImagesListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.listImages.matchFulfilled,
    effect: async (action, { dispatch, unsubscribe, cancelActiveListeners }) => {
      // Only run this listener on the first listImages request for no-board images
      if (action.meta.arg.queryCacheKey !== getListImagesUrl({ board_id: 'none', categories: IMAGE_CATEGORIES })) {
        return;
      }

      // this should only run once
      cancelActiveListeners();
      unsubscribe();

      const data = action.payload;

      if (data.items.length > 0) {
        dispatch(imageSelected(data.items[0] ?? null));
      }
    },
  });
};
