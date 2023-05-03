import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { configSelector } from 'features/system/store/configSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash-es';
import {
  selectResultsAll,
  selectResultsById,
  selectResultsEntities,
} from './resultsSlice';
import {
  selectUploadsAll,
  selectUploadsById,
  selectUploadsEntities,
} from './uploadsSlice';

export const gallerySelector = (state: RootState) => state.gallery;

export const imageGallerySelector = createSelector(
  [
    (state: RootState) => state,
    gallerySelector,
    uiSelector,
    lightboxSelector,
    activeTabNameSelector,
  ],
  (state, gallery, ui, lightbox, activeTabName) => {
    const {
      currentCategory,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      shouldAutoSwitchToNewImages,
      galleryWidth,
      shouldUseSingleGalleryColumn,
      selectedImage,
    } = gallery;

    const { shouldPinGallery } = ui;

    const { isLightboxOpen } = lightbox;

    return {
      shouldPinGallery,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      galleryGridTemplateColumns: shouldUseSingleGalleryColumn
        ? 'auto'
        : `repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, auto))`,
      shouldAutoSwitchToNewImages,
      currentCategory,
      images: state[currentCategory].entities,
      galleryWidth,
      shouldEnableResize:
        isLightboxOpen ||
        (activeTabName === 'unifiedCanvas' && shouldPinGallery)
          ? false
          : true,
      shouldUseSingleGalleryColumn,
      selectedImage,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const selectedImageSelector = createSelector(
  [(state: RootState) => state, gallerySelector],
  (state, gallery) => {
    const selectedImage = gallery.selectedImage;

    if (selectedImage?.type === 'results') {
      return selectResultsById(state, selectedImage.name);
    }

    if (selectedImage?.type === 'uploads') {
      return selectUploadsById(state, selectedImage.name);
    }
  }
);
