import { createAction } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import { noop } from 'es-toolkit';
import { setInfillMethod } from 'features/controlLayers/store/paramsSlice';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { appInfoApi } from 'services/api/endpoints/appInfo';
import { galleryApi } from 'services/api/endpoints/gallery';

export const appStarted = createAction('app/appStarted');

export const addAppStartedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: appStarted,
    effect: async (action, { unsubscribe, cancelActiveListeners, take, getState, dispatch }) => {
      // this should only run once
      cancelActiveListeners();
      unsubscribe();

      // Fire patchmatch check without blocking the image-selection logic below
      dispatch(appInfoApi.endpoints.getPatchmatchStatus.initiate())
        .unwrap()
        .then((isPatchmatchAvailable) => {
          const infillMethod = getState().params.infillMethod;

          if (!isPatchmatchAvailable && infillMethod === 'patchmatch') {
            dispatch(setInfillMethod('lama'));
          }
        })
        .catch(noop);

      // Ensure a gallery item is selected when we load the first board. The grid is fed by the
      // polymorphic `getGalleryItemNames` endpoint (image + video names interleaved by date),
      // so that's what we wait on — the older `getImageNames` is no longer dispatched and would
      // time out forever.
      //
      // The effect must be async and await take() so that RTK keeps the listener's AbortController
      // alive until the query resolves; a synchronous effect causes the controller to be aborted
      // immediately after the effect returns, before any network response arrives.
      const firstLoad = await take(galleryApi.endpoints.getGalleryItemNames.matchFulfilled, 5000);
      if (firstLoad === null) {
        // timeout or cancelled
        return;
      }
      const [{ payload }] = firstLoad;
      const selectedItem = selectLastSelectedItem(getState());
      if (selectedItem) {
        return;
      }
      const firstItem = payload.items[0];
      if (firstItem) {
        dispatch(imageSelected(firstItem.name));
      }
    },
  });
};
