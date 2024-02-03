import { createAction } from '@reduxjs/toolkit';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { imagesApi } from 'services/api/endpoints/images';
import type { ImageCache } from 'services/api/types';
import { getListImagesUrl, imagesSelectors } from 'services/api/util';

import { startAppListening } from '..';

export const appStarted = createAction('app/appStarted');

export const addFirstListImagesListener = () => {
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

      // TODO: figure out how to type the predicate
      const data = action.payload as ImageCache;

      if (data.ids.length > 0) {
        // Select the first image
        const firstImage = imagesSelectors.selectAll(data)[0];
        dispatch(imageSelected(firstImage ?? null));
      }
    },
  });
};
