import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { ListImagesArgs } from 'services/api/endpoints/images';
import { INITIAL_IMAGE_LIMIT } from './gallerySlice';
import {
  getBoardIdQueryParamForBoard,
  getCategoriesQueryParamForBoard,
} from './util';

export const gallerySelector = (state: RootState) => state.gallery;

export const selectLastSelectedImage = createSelector(
  (state: RootState) => state,
  (state) => state.gallery.selection[state.gallery.selection.length - 1],
  defaultSelectorOptions
);

export const selectListImagesBaseQueryArgs = createSelector(
  [(state: RootState) => state],
  (state) => {
    const { selectedBoardId } = state.gallery;

    const categories = getCategoriesQueryParamForBoard(selectedBoardId);
    const board_id = getBoardIdQueryParamForBoard(selectedBoardId);

    const listImagesBaseQueryArgs: ListImagesArgs = {
      categories,
      board_id,
      offset: 0,
      limit: INITIAL_IMAGE_LIMIT,
      is_intermediate: false,
    };

    return listImagesBaseQueryArgs;
  },
  defaultSelectorOptions
);
