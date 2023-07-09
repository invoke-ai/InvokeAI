import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { keyBy } from 'lodash-es';
import { imagesAdapter, initialGalleryState } from './gallerySlice';

export const gallerySelector = (state: RootState) => state.gallery;

export const selectFilteredImagesLocal = createSelector(
  (state: typeof initialGalleryState) => state,
  (galleryState) => {
    const allImages = imagesAdapter.getSelectors().selectAll(galleryState);
    const { categories, selectedBoardId } = galleryState;

    const filteredImages = allImages.filter((i) => {
      const isInCategory = categories.includes(i.image_category);
      const isInSelectedBoard = selectedBoardId
        ? i.board_id === selectedBoardId
        : true;
      return isInCategory && isInSelectedBoard;
    });

    return filteredImages;
  }
);

export const selectFilteredImages = createSelector(
  (state: RootState) => state,
  (state) => {
    return selectFilteredImagesLocal(state.gallery);
  },
  defaultSelectorOptions
);

export const selectFilteredImagesAsObject = createSelector(
  selectFilteredImages,
  (filteredImages) => keyBy(filteredImages, 'image_name')
);

export const selectFilteredImagesIds = createSelector(
  selectFilteredImages,
  (filteredImages) => filteredImages.map((i) => i.image_name)
);

export const selectLastSelectedImage = createSelector(
  (state: RootState) => state,
  (state) => state.gallery.selection[state.gallery.selection.length - 1],
  defaultSelectorOptions
);
