import { createAction } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { imagesApi } from 'services/api/endpoints/images';

export const appStarted = createAction('app/appStarted');

export const addAppStartedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: appStarted,
    effect: async (action, { unsubscribe, cancelActiveListeners, take, getState, dispatch }) => {
      // this should only run once
      cancelActiveListeners();
      unsubscribe();

      // ensure an image is selected when we load the first board
      const firstImageLoad = await take(imagesApi.endpoints.getImageNames.matchFulfilled);
      if (firstImageLoad !== null) {
        const [{ payload }] = firstImageLoad;
        const selectedImage = selectLastSelectedImage(getState());
        if (selectedImage) {
          return;
        }
        dispatch(imageSelected(payload.image_names.at(0) ?? null));
      }
    },
  });
};
