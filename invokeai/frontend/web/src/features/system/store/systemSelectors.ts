import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';

export const systemSelector = (state: RootState) => state.system;

export const languageSelector = createSelector(
  stateSelector,
  ({ system }) => system.language,
  defaultSelectorOptions
);
