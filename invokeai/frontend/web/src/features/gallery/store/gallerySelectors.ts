import type { SkipToken } from '@reduxjs/toolkit/query';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import type { ListBoardsArgs, ListImagesArgs } from 'services/api/types';

export const selectLastSelectedImage = createMemoizedSelector(
  selectGallerySlice,
  (gallery) => gallery.selection[gallery.selection.length - 1]
);

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
        }
      : skipToken
);

export const selectListBoardsQueryArgs = createMemoizedSelector(
  selectGallerySlice,
  (gallery): ListBoardsArgs => ({
    archived: gallery.shouldShowArchivedBoards ? true : undefined,
  })
);
