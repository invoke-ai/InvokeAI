import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { selectResultsById } from 'features/gallery/store/resultsSlice';
import { selectUploadsById } from 'features/gallery/store/uploadsSlice';
import { isEqual } from 'lodash-es';

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

export const initialImageSelector = createSelector(
  [(state: RootState) => state, generationSelector],
  (state, generation) => {
    const { initialImage } = generation;

    if (initialImage?.type === 'results') {
      return selectResultsById(state, initialImage.name);
    }

    if (initialImage?.type === 'uploads') {
      return selectUploadsById(state, initialImage.name);
    }
  }
);
