import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import { selectGetImageNamesQueryArgs, selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged, imageSelected } from 'features/gallery/store/gallerySlice';
import { galleryApi } from 'services/api/endpoints/gallery';

export const addBoardIdSelectedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: isAnyOf(boardIdSelected, galleryViewChanged),
    effect: async (action, { getState, dispatch, condition, cancelActiveListeners }) => {
      // Cancel any in-progress instances of this listener, we don't want to select an item from a previous board
      cancelActiveListeners();

      if (boardIdSelected.match(action) && action.payload.select) {
        // This action already has a resource selection - skip the below auto-selection logic
        return;
      }

      const state = getState();

      const board_id = selectSelectedBoardId(state);

      // The grid is now backed by the polymorphic getGalleryItemNames endpoint (the legacy
      // getImageNames query is no longer dispatched), so the auto-select probe must read its
      // cache or it will time out and clear the user's selection on every board switch.
      const queryArgs = { ...selectGetImageNamesQueryArgs(state), board_id };
      // wait until the board has some items - maybe it already has some from a previous fetch
      // must use getState() to ensure we do not have stale state
      const isSuccess = await condition(
        () => galleryApi.endpoints.getGalleryItemNames.select(queryArgs)(getState()).isSuccess,
        5000
      );

      if (!isSuccess) {
        dispatch(imageSelected(null));
        return;
      }

      // the board was just changed - we can select the first gallery item (image or video)
      const items = galleryApi.endpoints.getGalleryItemNames.select(queryArgs)(getState()).data?.items;

      const itemToSelect = items && items.length > 0 ? (items[0]?.name ?? null) : null;

      dispatch(imageSelected(itemToSelect));
    },
  });
};
