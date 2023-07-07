import { createAction } from '@reduxjs/toolkit';
import { receivedPageOfImages } from 'services/api/thunks/image';
import { startAppListening } from '..';

export const appStarted = createAction('app/appStarted');

export const addAppStartedListener = () => {
  startAppListening({
    actionCreator: appStarted,
    effect: (action, { getState, dispatch }) => {
      // fill up the gallery tab with images
      dispatch(
        receivedPageOfImages({
          categories: ['general'],
          is_intermediate: false,
          limit: 100,
        })
      );

      // fill up the assets tab with images
      dispatch(
        receivedPageOfImages({
          categories: ['control', 'mask', 'user', 'other'],
          is_intermediate: false,
          limit: 100,
        })
      );
    },
  });
};
