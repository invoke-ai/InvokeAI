import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import type { GetImageNamesArgs, GetVideoIdsArgs, ListBoardsArgs } from 'services/api/types';

export const selectFirstSelectedImage = createSelector(selectGallerySlice, (gallery) => gallery.selection.at(0));
export const selectLastSelectedImage = createSelector(selectGallerySlice, (gallery) => gallery.selection.at(-1));

export const selectListBoardsQueryArgs = createMemoizedSelector(
  selectGallerySlice,
  (gallery): ListBoardsArgs => ({
    order_by: gallery.boardsListOrderBy,
    direction: gallery.boardsListOrderDir,
    include_archived: gallery.shouldShowArchivedBoards ? true : undefined,
  })
);

export const selectAutoAddBoardId = createSelector(selectGallerySlice, (gallery) => gallery.autoAddBoardId);
export const selectAutoSwitch = createSelector(selectGallerySlice, (gallery) => gallery.shouldAutoSwitch);
export const selectSelectedBoardId = createSelector(selectGallerySlice, (gallery) => gallery.selectedBoardId);
export const selectGalleryView = createSelector(selectGallerySlice, (gallery) => gallery.galleryView);
const selectGalleryQueryCategories = createSelector(selectGalleryView, (galleryView) => {
  if (galleryView === 'images') {
    return IMAGE_CATEGORIES;
  }
  if (galleryView === 'videos') {
    return [];
  }
  return ASSETS_CATEGORIES;
});
const selectGallerySearchTerm = createSelector(selectGallerySlice, (gallery) => gallery.searchTerm);
const selectGalleryOrderDir = createSelector(selectGallerySlice, (gallery) => gallery.orderDir);
const selectGalleryStarredFirst = createSelector(selectGallerySlice, (gallery) => gallery.starredFirst);

export const selectGetImageNamesQueryArgs = createMemoizedSelector(
  [
    selectSelectedBoardId,
    selectGalleryQueryCategories,
    selectGallerySearchTerm,
    selectGalleryOrderDir,
    selectGalleryStarredFirst,
  ],
  (board_id, categories, search_term, order_dir, starred_first): GetImageNamesArgs => ({
    board_id,
    categories,
    search_term,
    order_dir,
    starred_first,
    is_intermediate: false,
  })
);

export const selectGetVideoIdsQueryArgs = createMemoizedSelector(
  [
    selectSelectedBoardId,
    selectGallerySearchTerm,
    selectGalleryOrderDir,
    selectGalleryStarredFirst,
  ],
  (board_id, search_term, order_dir, starred_first): GetVideoIdsArgs => ({
    board_id,
    search_term,
    order_dir,
    starred_first,
    is_intermediate: false,
  })
);

export const selectAutoAssignBoardOnClick = createSelector(
  selectGallerySlice,
  (gallery) => gallery.autoAssignBoardOnClick
);
export const selectBoardSearchText = createSelector(selectGallerySlice, (gallery) => gallery.boardSearchText);
export const selectSearchTerm = createSelector(selectGallerySlice, (gallery) => gallery.searchTerm);
export const selectBoardsListOrderBy = createSelector(selectGallerySlice, (gallery) => gallery.boardsListOrderBy);
export const selectBoardsListOrderDir = createSelector(selectGallerySlice, (gallery) => gallery.boardsListOrderDir);

export const selectSelectionCount = createSelector(selectGallerySlice, (gallery) => gallery.selection.length);
export const selectSelection = createSelector(selectGallerySlice, (gallery) => gallery.selection);
export const selectGalleryImageMinimumWidth = createSelector(
  selectGallerySlice,
  (gallery) => gallery.galleryImageMinimumWidth
);

export const selectComparisonMode = createSelector(selectGallerySlice, (gallery) => gallery.comparisonMode);
export const selectComparisonFit = createSelector(selectGallerySlice, (gallery) => gallery.comparisonFit);
export const selectImageToCompare = createSelector(selectGallerySlice, (gallery) => gallery.imageToCompare);
export const selectAlwaysShouldImageSizeBadge = createSelector(
  selectGallerySlice,
  (gallery) => gallery.alwaysShowImageSizeBadge
);
