import { createAction } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import { noop } from 'es-toolkit';
import { setInfillMethod } from 'features/controlLayers/store/paramsSlice';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { appInfoApi } from 'services/api/endpoints/appInfo';
import { imagesApi } from 'services/api/endpoints/images';

export const appStarted = createAction('app/appStarted');

export const addAppStartedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: appStarted,
    effect: (action, { unsubscribe, cancelActiveListeners, take, getState, dispatch }) => {
      // this should only run once
      cancelActiveListeners();
      unsubscribe();

      // ensure an image is selected when we load the first board
      take(imagesApi.endpoints.getImageNames.matchFulfilled).then((firstImageLoad) => {
        if (firstImageLoad === null) {
          // timeout or cancelled
          return;
        }
        const [{ payload }] = firstImageLoad;
        const selectedImage = selectLastSelectedItem(getState());
        if (selectedImage) {
          return;
        }
        if (payload.image_names[0]) {
          dispatch(imageSelected(payload.image_names[0]));
        }
      });

      dispatch(appInfoApi.endpoints.getPatchmatchStatus.initiate())
        .unwrap()
        .then((isPatchmatchAvailable) => {
          const infillMethod = getState().params.infillMethod;

          if (!isPatchmatchAvailable && infillMethod === 'patchmatch') {
            dispatch(setInfillMethod('lama'));
          }
        })
        .catch(noop);
    },
  });
};
