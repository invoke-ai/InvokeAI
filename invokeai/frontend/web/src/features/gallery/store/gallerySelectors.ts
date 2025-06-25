import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import type { ListBoardsArgs, ListImagesArgs } from 'services/api/types';
import type { SetNonNullable } from 'type-fest';

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
export const selectGalleryQueryCategories = createSelector(selectGalleryView, (galleryView) =>
  galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES
);
export const selectGallerySearchTerm = createSelector(selectGallerySlice, (gallery) => gallery.searchTerm);
export const selectGalleryOrderDir = createSelector(selectGallerySlice, (gallery) => gallery.orderDir);
export const selectGalleryStarredFirst = createSelector(selectGallerySlice, (gallery) => gallery.starredFirst);

export const selectListImagesQueryArgs = createMemoizedSelector(
  [
    selectSelectedBoardId,
    selectGalleryQueryCategories,
    selectGallerySearchTerm,
    selectGalleryOrderDir,
    selectGalleryStarredFirst,
  ],
  (board_id, categories, search_term, order_dir, starred_first) =>
    ({
      board_id,
      categories,
      search_term,
      order_dir,
      starred_first,
      is_intermediate: false, // We don't show intermediate images in the gallery
      limit: 100, // Page size is _always_ 100
    }) satisfies SetNonNullable<ListImagesArgs, 'limit'>
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
export const selectHasImageToCompare = createSelector(selectGallerySlice, (gallery) => Boolean(gallery.imageToCompare));
export const selectAlwaysShouldImageSizeBadge = createSelector(
  selectGallerySlice,
  (gallery) => gallery.alwaysShowImageSizeBadge
);
