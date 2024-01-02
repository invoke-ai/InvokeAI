import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { RootState } from 'app/store/store';
import type { ListImagesArgs } from 'services/api/types';

import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
  INITIAL_IMAGE_LIMIT,
} from './types';

export const gallerySelector = (state: RootState) => state.gallery;

export const selectLastSelectedImage = createMemoizedSelector(
  (state: RootState) => state,
  (state) => state.gallery.selection[state.gallery.selection.length - 1]
);

export const selectListImagesBaseQueryArgs = createMemoizedSelector(
  [(state: RootState) => state],
  (state) => {
    const { selectedBoardId, galleryView } = state.gallery;
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
