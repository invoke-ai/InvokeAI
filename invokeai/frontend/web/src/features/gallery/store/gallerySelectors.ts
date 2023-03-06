import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash';

import { GalleryState } from './gallerySlice';

export const gallerySelector = (state: RootState) => state.gallery;

export const imageGallerySelector = createSelector(
  [gallerySelector, lightboxSelector, isStagingSelector, activeTabNameSelector],
  (gallery: GalleryState, lightbox, isStaging, activeTabName) => {
    const {
      categories,
      currentCategory,
      currentImageUuid,
      shouldPinGallery,
      shouldShowGallery,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      shouldHoldGalleryOpen,
      shouldAutoSwitchToNewImages,
      galleryWidth,
      shouldUseSingleGalleryColumn,
    } = gallery;

    const { isLightboxOpen } = lightbox;

    return {
      currentImageUuid,
      shouldPinGallery,
      shouldShowGallery,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      galleryGridTemplateColumns: shouldUseSingleGalleryColumn
        ? 'auto'
        : `repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, auto))`,
      activeTabName,
      shouldHoldGalleryOpen,
      shouldAutoSwitchToNewImages,
      images: categories[currentCategory].images,
      areMoreImagesAvailable:
        categories[currentCategory].areMoreImagesAvailable,
      currentCategory,
      galleryWidth,
      isLightboxOpen,
      isStaging,
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
  (gallery: GalleryState, system, lightbox, activeTabName) => {
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
