import { createAction } from '@reduxjs/toolkit';
import { isInitializedChanged } from 'features/gallery/store/gallerySlice';
import { receivedPageOfImages } from 'services/api/thunks/image';
import { startAppListening } from '..';

export const appStarted = createAction('app/appStarted');

export const addAppStartedListener = () => {
  startAppListening({
    actionCreator: appStarted,
    effect: async (
      action,
      { getState, dispatch, unsubscribe, cancelActiveListeners }
    ) => {
      cancelActiveListeners();
      unsubscribe();
      // fill up the gallery tab with images
      await dispatch(
        receivedPageOfImages({
          categories: ['general'],
          is_intermediate: false,
          offset: 0,
          limit: 100,
        })
      );

      // fill up the assets tab with images
      dispatch(
        receivedPageOfImages({
          categories: ['control', 'mask', 'user', 'other'],
          is_intermediate: false,
          offset: 0,
          limit: 100,
        })
      );

      // tell the gallery it has made its initial fetches
      dispatch(isInitializedChanged(true));
    },
  });
};
