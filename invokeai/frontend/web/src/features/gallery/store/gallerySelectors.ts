import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { ListImagesArgs } from 'services/api/endpoints/images';
import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
  INITIAL_IMAGE_LIMIT,
} from './gallerySlice';

export const gallerySelector = (state: RootState) => state.gallery;

export const selectLastSelectedImage = createSelector(
  (state: RootState) => state,
  (state) => state.gallery.selection[state.gallery.selection.length - 1],
  defaultSelectorOptions
);

export const selectListImagesBaseQueryArgs = createSelector(
  [(state: RootState) => state],
  (state) => {
    const { selectedBoardId, galleryView } = state.gallery;

    const listImagesBaseQueryArgs: ListImagesArgs = {
      categories:
        galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES,
      board_id: selectedBoardId === 'all' ? undefined : selectedBoardId,
      offset: 0,
      limit: INITIAL_IMAGE_LIMIT,
      is_intermediate: false,
    };

    return listImagesBaseQueryArgs;
  },
  defaultSelectorOptions
);

export const selectListAllImagesBaseQueryArgs = createSelector(
  [(state: RootState) => state],
  (state) => {
    const { galleryView } = state.gallery;

    const listImagesBaseQueryArgs = {
      categories:
        galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES,
      offset: 0,
      limit: INITIAL_IMAGE_LIMIT,
      is_intermediate: false,
    };

    return listImagesBaseQueryArgs;
  },
  defaultSelectorOptions
);
