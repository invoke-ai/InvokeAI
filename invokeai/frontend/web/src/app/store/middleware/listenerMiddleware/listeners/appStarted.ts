import { createAction } from '@reduxjs/toolkit';
import { isLoadingChanged } from 'features/gallery/store/gallerySlice';
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

      dispatch(isLoadingChanged(false));
    },
  });
};
