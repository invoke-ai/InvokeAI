import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash';

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

export const hoverableImageSelector = createSelector(
  [gallerySelector, systemSelector, lightboxSelector, activeTabNameSelector],
  (gallery, system, lightbox, activeTabName) => {
    return {
      mayDeleteImage: system.isConnected && !system.isProcessing,
      galleryImageObjectFit: gallery.galleryImageObjectFit,
      galleryImageMinimumWidth: gallery.galleryImageMinimumWidth,
      shouldUseSingleGalleryColumn: gallery.shouldUseSingleGalleryColumn,
      activeTabName,
      isLightboxOpen: lightbox.isLightboxOpen,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);
