import { createAction } from '@reduxjs/toolkit';
import {
  IMAGE_CATEGORIES,
  imageSelected,
} from 'features/gallery/store/gallerySlice';
import {
  ImageCache,
  getListImagesUrl,
  imagesApi,
} from 'services/api/endpoints/images';
import { startAppListening } from '..';

export const appStarted = createAction('app/appStarted');

export const addFirstListImagesListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.listImages.matchFulfilled,
    effect: async (
      action,
      { getState, dispatch, unsubscribe, cancelActiveListeners }
    ) => {
      // Only run this listener on the first listImages request for `images` categories
      if (
        action.meta.arg.queryCacheKey !==
        getListImagesUrl({ categories: IMAGE_CATEGORIES })
      ) {
        return;
      }

      // this should only run once
      cancelActiveListeners();
      unsubscribe();

      // TODO: figure out how to type the predicate
      const data = action.payload as ImageCache;

      if (data.ids.length > 0) {
        // Select the first image
        dispatch(imageSelected(data.ids[0] as string));
      }
    },
  });
};
