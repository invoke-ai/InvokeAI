import { createSelector } from '@reduxjs/toolkit';
import { RootState } from '../../app/store';
import { activeTabNameSelector } from '../options/optionsSelectors';
import { OptionsState } from '../options/optionsSlice';
import { GalleryState } from './gallerySlice';

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
  }
);

export const hoverableImageSelector = createSelector(
  [(state: RootState) => state.options, (state: RootState) => state.gallery, activeTabNameSelector],
  (options: OptionsState, gallery: GalleryState, activeTabName) => {
    return {
      galleryImageObjectFit: gallery.galleryImageObjectFit,
      galleryImageMinimumWidth: gallery.galleryImageMinimumWidth,
      activeTabName,
    };
  }
);
