import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { gallerySelector } from 'features/gallery/store/gallerySelectors';
import {
  selectResultsById,
  selectResultsEntities,
} from 'features/gallery/store/resultsSlice';
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
    const { initialImage: initialImageName } = generation;

    return (
      selectResultsById(state, initialImageName as string) ??
      selectUploadsById(state, initialImageName as string)
    );
  }
);
