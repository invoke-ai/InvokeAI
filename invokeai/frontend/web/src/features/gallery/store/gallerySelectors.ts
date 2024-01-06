import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import type { ListImagesArgs } from 'services/api/types';

import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
  INITIAL_IMAGE_LIMIT,
} from './types';

export const selectLastSelectedImage = createMemoizedSelector(
  selectGallerySlice,
  (gallery) => gallery.selection[gallery.selection.length - 1]
);

export const selectListImagesBaseQueryArgs = createMemoizedSelector(
  selectGallerySlice,
  (gallery) => {
    const { selectedBoardId, galleryView } = gallery;
    const categories =
      galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES;

    const listImagesBaseQueryArgs: ListImagesArgs = {
      board_id: selectedBoardId,
      categories,
      offset: 0,
      limit: INITIAL_IMAGE_LIMIT,
      is_intermediate: false,
    };

    return listImagesBaseQueryArgs;
  }
);
