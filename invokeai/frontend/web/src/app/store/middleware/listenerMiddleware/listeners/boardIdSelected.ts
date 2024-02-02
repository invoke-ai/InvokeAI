import { isAnyOf } from '@reduxjs/toolkit';
import { boardIdSelected, galleryViewChanged, imageSelected } from 'features/gallery/store/gallerySlice';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { imagesApi } from 'services/api/endpoints/images';
import { imagesSelectors } from 'services/api/util';

import { startAppListening } from '..';

export const addBoardIdSelectedListener = () => {
  startAppListening({
    matcher: isAnyOf(boardIdSelected, galleryViewChanged),
    effect: async (action, { getState, dispatch, condition, cancelActiveListeners }) => {
      // Cancel any in-progress instances of this listener, we don't want to select an image from a previous board
      cancelActiveListeners();

      const state = getState();

      const board_id = boardIdSelected.match(action) ? action.payload.boardId : state.gallery.selectedBoardId;

      const galleryView = galleryViewChanged.match(action) ? action.payload : state.gallery.galleryView;

      // when a board is selected, we need to wait until the board has loaded *some* images, then select the first one
      const categories = galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES;

      const queryArgs = { board_id: board_id ?? 'none', categories };

      // wait until the board has some images - maybe it already has some from a previous fetch
      // must use getState() to ensure we do not have stale state
      const isSuccess = await condition(
        () => imagesApi.endpoints.listImages.select(queryArgs)(getState()).isSuccess,
        5000
      );

      if (isSuccess) {
        // the board was just changed - we can select the first image
        const { data: boardImagesData } = imagesApi.endpoints.listImages.select(queryArgs)(getState());

        if (boardImagesData && boardIdSelected.match(action) && action.payload.selectedImageName) {
          const selectedImage = imagesSelectors.selectById(boardImagesData, action.payload.selectedImageName);
          dispatch(imageSelected(selectedImage || null));
        } else if (boardImagesData) {
          const firstImage = imagesSelectors.selectAll(boardImagesData)[0];
          dispatch(imageSelected(firstImage || null));
        } else {
          // board has no images - deselect
          dispatch(imageSelected(null));
        }
      } else {
        // fallback - deselect
        dispatch(imageSelected(null));
      }
    },
  });
};
