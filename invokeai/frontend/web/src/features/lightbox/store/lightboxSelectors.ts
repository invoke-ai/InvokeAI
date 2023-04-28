import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { isEqual } from 'lodash-es';

export const lightboxSelector = createSelector(
  (state: RootState) => state.lightbox,
  (lightbox) => lightbox,
  {
    memoizeOptions: {
      equalityCheck: isEqual,
    },
  }
);
