import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged, imageSelected } from 'features/gallery/store/gallerySlice';
import { imagesApi } from 'services/api/endpoints/images';

export const addBoardIdSelectedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: isAnyOf(boardIdSelected, galleryViewChanged),
    effect: async (action, { getState, dispatch, condition, cancelActiveListeners }) => {
      // Cancel any in-progress instances of this listener, we don't want to select an image from a previous board
      cancelActiveListeners();

      const state = getState();

      const queryArgs = selectListImagesQueryArgs(state);

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
          const selectedImage = boardImagesData.items.find(
            (item) => item.image_name === action.payload.selectedImageName
          );
          dispatch(imageSelected(selectedImage || null));
        } else if (boardImagesData) {
          dispatch(imageSelected(boardImagesData.items[0] || null));
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
