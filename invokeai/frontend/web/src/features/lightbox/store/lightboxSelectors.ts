import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { isEqual } from 'lodash';

export const lightboxSelector = createSelector(
  (state: RootState) => state.lightbox,
  (lightbox) => lightbox,
  {
    memoizeOptions: {
      equalityCheck: isEqual,
    },
  }
);
