import { createSelector } from '@reduxjs/toolkit';
import type { SkipToken } from '@reduxjs/toolkit/query';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import type { ListBoardsArgs, ListImagesArgs } from 'services/api/types';

export const selectFirstSelectedImage = createSelector(selectGallerySlice, (gallery) => gallery.selection.at(0));
export const selectLastSelectedImage = createSelector(selectGallerySlice, (gallery) => gallery.selection.at(-1));
export const selectLastSelectedImageName = createSelector(
  selectGallerySlice,
  (gallery) => gallery.selection.at(-1)?.image_name
);

export const selectGalleryLimit = createSelector(selectGallerySlice, (gallery) => gallery.limit);
export const selectListImagesQueryArgs = createMemoizedSelector(
  selectGallerySlice,
  (gallery): ListImagesArgs | SkipToken =>
    gallery.limit
      ? {
          board_id: gallery.selectedBoardId,
          categories: gallery.galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES,
          offset: gallery.offset,
          limit: gallery.limit,
          is_intermediate: false,
          starred_first: gallery.starredFirst,
          order_dir: gallery.orderDir,
          search_term: gallery.searchTerm,
        }
      : skipToken
);

export const selectListBoardsQueryArgs = createMemoizedSelector(
  selectGallerySlice,
  (gallery): ListBoardsArgs => ({
    order_by: gallery.boardsListOrderBy,
    direction: gallery.boardsListOrderDir,
    include_archived: gallery.shouldShowArchivedBoards ? true : undefined,
  })
);

export const selectAutoAddBoardId = createSelector(selectGallerySlice, (gallery) => gallery.autoAddBoardId);
export const selectSelectedBoardId = createSelector(selectGallerySlice, (gallery) => gallery.selectedBoardId);
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
