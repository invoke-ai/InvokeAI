import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { RootState } from 'app/store';
import { tabMap } from 'features/tabs/components/InvokeTabs';
import { OptionsState } from './optionsSlice';

export const activeTabNameSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => tabMap[options.activeTab],
  {
    memoizeOptions: {
      equalityCheck: _.isEqual,
    },
  }
);

export const mayGenerateMultipleImagesSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    const { shouldRandomizeSeed, shouldGenerateVariations } = options;

    return shouldRandomizeSeed || shouldGenerateVariations;
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export const optionsSelector = (state: RootState): OptionsState =>
  state.options;
