import { createAction } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';

export const appStarted = createAction('app/appStarted');

export const addAppStartedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: appStarted,
    effect: async (action, { unsubscribe, cancelActiveListeners }) => {
      // this should only run once
      cancelActiveListeners();
      unsubscribe();
    },
  });
};
