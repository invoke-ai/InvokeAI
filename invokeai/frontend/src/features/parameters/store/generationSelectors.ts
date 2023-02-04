import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { RootState } from 'app/store';

export const generationSelector = (state: RootState) => state.generation;

export const mayGenerateMultipleImagesSelector = createSelector(
  generationSelector,
  ({ shouldRandomizeSeed, shouldGenerateVariations }) => {
    return shouldRandomizeSeed || shouldGenerateVariations;
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);
