import { createSelector } from '@reduxjs/toolkit';
import { RootState } from '../../app/store';
import { tabMap } from '../tabs/InvokeTabs';
import { OptionsState } from './optionsSlice';

export const activeTabNameSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => tabMap[options.activeTab]
);
