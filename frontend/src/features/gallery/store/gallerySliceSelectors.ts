import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import { OptionsState } from 'features/options/store/optionsSlice';
import { SystemState } from 'features/system/store/systemSlice';
import { GalleryState } from './gallerySlice';
import _ from 'lodash';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';

export const imageGallerySelector = createSelector(
  [
    (state: RootState) => state.gallery,
    (state: RootState) => state.options,
    isStagingSelector,
    activeTabNameSelector,
  ],
  (gallery: GalleryState, options: OptionsState, isStaging, activeTabName) => {
    const {
      categories,
      currentCategory,
      currentImageUuid,
      shouldPinGallery,
      shouldShowGallery,
      galleryScrollPosition,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      shouldHoldGalleryOpen,
      shouldAutoSwitchToNewImages,
      galleryWidth,
      shouldUseSingleGalleryColumn,
    } = gallery;

    const { isLightBoxOpen } = options;

    return {
      currentImageUuid,
      shouldPinGallery,
      shouldShowGallery,
      galleryScrollPosition,
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
      isLightBoxOpen,
      isStaging,
      shouldEnableResize:
        isLightBoxOpen ||
        (activeTabName === 'unifiedCanvas' && shouldPinGallery)
          ? false
          : true,
      shouldUseSingleGalleryColumn,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export const hoverableImageSelector = createSelector(
  [
    (state: RootState) => state.options,
    (state: RootState) => state.gallery,
    (state: RootState) => state.system,
    activeTabNameSelector,
  ],
  (
    options: OptionsState,
    gallery: GalleryState,
    system: SystemState,
    activeTabName
  ) => {
    return {
      mayDeleteImage: system.isConnected && !system.isProcessing,
      galleryImageObjectFit: gallery.galleryImageObjectFit,
      galleryImageMinimumWidth: gallery.galleryImageMinimumWidth,
      shouldUseSingleGalleryColumn: gallery.shouldUseSingleGalleryColumn,
      activeTabName,
      isLightBoxOpen: options.isLightBoxOpen,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export const gallerySelector = (state: RootState) => state.gallery;
