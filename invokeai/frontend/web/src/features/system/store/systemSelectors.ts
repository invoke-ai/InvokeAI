import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';

export const systemSelector = (state: RootState) => state.system;

export const toastQueueSelector = (state: RootState) => state.system.toastQueue;

export const languageSelector = createSelector(
  systemSelector,
  (system) => system.language,
  defaultSelectorOptions
);

export const isProcessingSelector = (state: RootState) =>
  state.system.isProcessing;

export const selectIsBusy = createSelector(
  (state: RootState) => state,
  (state) => state.system.isProcessing || !state.system.isConnected
);
