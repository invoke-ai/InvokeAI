import { createSelector } from '@reduxjs/toolkit';
import { RootState } from '../../app/store';
import { activeTabNameSelector } from '../options/optionsSelectors';
import { OptionsState } from '../options/optionsSlice';
import { SystemState } from '../system/systemSlice';
import { GalleryState } from './gallerySlice';
import _ from 'lodash';

export const imageGallerySelector = createSelector(
  [
    (state: RootState) => state.gallery,
    (state: RootState) => state.options,
    activeTabNameSelector,
  ],
  (gallery: GalleryState, options: OptionsState, activeTabName) => {
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
    } = gallery;

    return {
      currentImageUuid,
      shouldPinGallery,
      shouldShowGallery,
      galleryScrollPosition,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      galleryGridTemplateColumns: `repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, auto))`,
      activeTabName,
      shouldHoldGalleryOpen,
      shouldAutoSwitchToNewImages,
      images: categories[currentCategory].images,
      areMoreImagesAvailable:
        categories[currentCategory].areMoreImagesAvailable,
      currentCategory,
      galleryWidth,
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
      activeTabName,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);
