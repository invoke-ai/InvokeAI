import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import { selectGetImageNamesQueryArgs, selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged, imageSelected } from 'features/gallery/store/gallerySlice';
import { imagesApi } from 'services/api/endpoints/images';

export const addBoardIdSelectedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: isAnyOf(boardIdSelected, galleryViewChanged),
    effect: async (action, { getState, dispatch, condition, cancelActiveListeners }) => {
      // Cancel any in-progress instances of this listener, we don't want to select an image from a previous board
      cancelActiveListeners();

      if (boardIdSelected.match(action) && action.payload.selectedImageName) {
        // This action already has a selected image name, we trust it is valid
        return;
      }

      const state = getState();

      const board_id = selectSelectedBoardId(state);

      const queryArgs = { ...selectGetImageNamesQueryArgs(state), board_id };

      // wait until the board has some images - maybe it already has some from a previous fetch
      // must use getState() to ensure we do not have stale state
      const isSuccess = await condition(
        () => imagesApi.endpoints.getImageNames.select(queryArgs)(getState()).isSuccess,
        5000
      );

      if (!isSuccess) {
        dispatch(imageSelected(null));
        return;
      }

      // the board was just changed - we can select the first image
      const imageNames = imagesApi.endpoints.getImageNames.select(queryArgs)(getState()).data?.image_names;

      const imageToSelect = imageNames?.at(0) ?? null;

      dispatch(imageSelected(imageToSelect));
    },
  });
};
