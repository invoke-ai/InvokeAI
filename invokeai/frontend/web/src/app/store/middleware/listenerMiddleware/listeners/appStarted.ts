import { createAction } from '@reduxjs/toolkit';
import { startAppListening } from '..';

export const appStarted = createAction('app/appStarted');

export const addAppStartedListener = () => {
  startAppListening({
    actionCreator: appStarted,
    effect: async (action, { unsubscribe, cancelActiveListeners }) => {
      // this should only run once
      cancelActiveListeners();
      unsubscribe();
    },
  });
};
