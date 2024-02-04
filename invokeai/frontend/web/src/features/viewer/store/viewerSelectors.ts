import { createSelector } from '@reduxjs/toolkit';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { selectViewerSlice } from 'features/viewer/store/viewerSlice';

export const selectIsDisabledToolbarImageButtons = createSelector(
  selectViewerSlice,
  selectLastSelectedImage,
  (viewer, lastSelectedImage) => {
    return viewer.viewerMode === 'progress' || !lastSelectedImage;
  }
);
