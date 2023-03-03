import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { isEqual } from 'lodash';

export const generationSelector = (state: RootState) => state.generation;

export const mayGenerateMultipleImagesSelector = createSelector(
  generationSelector,
  ({ shouldRandomizeSeed, shouldGenerateVariations }) => {
    return shouldRandomizeSeed || shouldGenerateVariations;
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);
