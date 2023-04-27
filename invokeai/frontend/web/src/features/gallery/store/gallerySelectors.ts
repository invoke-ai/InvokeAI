import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { configSelector } from 'features/system/store/configSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash';
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
  [gallerySelector, uiSelector, lightboxSelector, activeTabNameSelector],
  (gallery, ui, lightbox, activeTabName) => {
    const {
      categories,
      currentCategory,
      currentImageUuid,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      shouldAutoSwitchToNewImages,
      galleryWidth,
      shouldUseSingleGalleryColumn,
    } = gallery;

    const { shouldPinGallery } = ui;

    const { isLightboxOpen } = lightbox;

    return {
      currentImageUuid,
      shouldPinGallery,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      galleryGridTemplateColumns: shouldUseSingleGalleryColumn
        ? 'auto'
        : `repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, auto))`,
      shouldAutoSwitchToNewImages,
      currentCategory,
      images: categories[currentCategory].images,
      areMoreImagesAvailable:
        categories[currentCategory].areMoreImagesAvailable,
      galleryWidth,
      shouldEnableResize:
        isLightboxOpen ||
        (activeTabName === 'unifiedCanvas' && shouldPinGallery)
          ? false
          : true,
      shouldUseSingleGalleryColumn,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const selectedImageSelector = createSelector(
  [gallerySelector, selectResultsEntities, selectUploadsEntities],
  (gallery, allResults, allUploads) => {
    const selectedImageName = gallery.selectedImageName;

    if (selectedImageName in allResults) {
      return allResults[selectedImageName];
    }

    if (selectedImageName in allUploads) {
      return allUploads[selectedImageName];
    }
  }
);
