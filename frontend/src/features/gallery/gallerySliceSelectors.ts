import { createSelector } from '@reduxjs/toolkit';
import { RootState } from '../../app/store';
import { OptionsState } from '../options/optionsSlice';
import { tabMap } from '../tabs/InvokeTabs';
import { GalleryState } from './gallerySlice';

export const imageGallerySelector = createSelector(
  [(state: RootState) => state.gallery, (state: RootState) => state.options],
  (gallery: GalleryState, options: OptionsState) => {
    const {
      images,
      currentImageUuid,
      areMoreImagesAvailable,
      shouldPinGallery,
      shouldShowGallery,
      galleryScrollPosition,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      shouldHoldGalleryOpen,
    } = gallery;

    const { activeTab } = options;

    return {
      images,
      currentImageUuid,
      areMoreImagesAvailable,
      shouldPinGallery,
      shouldShowGallery,
      galleryScrollPosition,
      galleryImageMinimumWidth,
      galleryImageObjectFit,
      galleryGridTemplateColumns: `repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, auto))`,
      activeTabName: tabMap[activeTab],
      shouldHoldGalleryOpen,
    };
  }
);

export const hoverableImageSelector = createSelector(
  [(state: RootState) => state.options, (state: RootState) => state.gallery],
  (options: OptionsState, gallery: GalleryState) => {
    return {
      galleryImageObjectFit: gallery.galleryImageObjectFit,
      galleryImageMinimumWidth: gallery.galleryImageMinimumWidth,
      activeTabName: tabMap[options.activeTab],
    };
  }
);
