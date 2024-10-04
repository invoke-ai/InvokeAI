import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, offsetChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { imagesApi } from 'services/api/endpoints/images';

export const addGalleryOffsetChangedListener = (startAppListening: AppStartListening) => {
  /**
   * When the user changes pages in the gallery, we need to wait until the next page of images is loaded, then maybe
   * update the selection.
   *
   * There are a three scenarios:
   *
   * 1. The page is changed by clicking the pagination buttons. No changes to selection are needed.
   *
   * 2. The page is changed by using the arrow keys (without alt).
   * - When going backwards, select the last image.
   * - When going forwards, select the first image.
   *
   * 3. The page is changed by using the arrows keys with alt. This means the user is changing the comparison image.
   * - When going backwards, select the last image _as the comparison image_.
   * - When going forwards, select the first image _as the comparison image_.
   */
  startAppListening({
    actionCreator: offsetChanged,
    effect: async (action, { dispatch, getState, getOriginalState, take, cancelActiveListeners }) => {
      // Cancel any active listeners to prevent the selection from changing without user input
      cancelActiveListeners();

      const { withHotkey } = action.payload;

      if (!withHotkey) {
        // User changed pages by clicking the pagination buttons - no changes to selection
        return;
      }

      const originalState = getOriginalState();
      const prevOffset = originalState.gallery.offset;
      const offset = getState().gallery.offset;

      if (offset === prevOffset) {
        // The page didn't change - bail
        return;
      }

      /**
       * We need to wait until the next page of images is loaded before updating the selection, so we use the correct
       * page of images.
       *
       * The simplest way to do it would be to use `take` to wait for the next fulfilled action, but RTK-Q doesn't
       * dispatch an action on cache hits. This means the `take` will only return if the cache is empty. If the user
       * changes to a cached page - a common situation - the `take` will never resolve.
       *
       * So we need to take a two-step approach. First, check if we have data in the cache for the page of images. If
       * we have data cached, use it to update the selection. If we don't have data cached, wait for the next fulfilled
       * action, which updates the cache, then use the cache to update the selection.
       */

      // Check if we have data in the cache for the page of images
      const queryArgs = selectListImagesQueryArgs(getState());
      let { data } = imagesApi.endpoints.listImages.select(queryArgs)(getState());

      // No data yet - wait for the network request to complete
      if (!data) {
        const takeResult = await take(imagesApi.endpoints.listImages.matchFulfilled, 5000);
        if (!takeResult) {
          // The request didn't complete in time - bail
          return;
        }
        data = takeResult[0].payload;
      }

      // We awaited a network request - state could have changed, get fresh state
      const state = getState();
      const { selection, imageToCompare } = state.gallery;
      const imageDTOs = data?.items;

      if (!imageDTOs) {
        // The page didn't load - bail
        return;
      }

      if (withHotkey === 'arrow') {
        // User changed pages by using the arrow keys - selection changes to first or last image depending
        if (offset < prevOffset) {
          // We've gone backwards
          const lastImage = imageDTOs[imageDTOs.length - 1];
          if (!selection.some((selectedImage) => selectedImage.image_name === lastImage?.image_name)) {
            dispatch(selectionChanged(lastImage ? [lastImage] : []));
          }
        } else {
          // We've gone forwards
          const firstImage = imageDTOs[0];
          if (!selection.some((selectedImage) => selectedImage.image_name === firstImage?.image_name)) {
            dispatch(selectionChanged(firstImage ? [firstImage] : []));
          }
        }
        return;
      }

      if (withHotkey === 'alt+arrow') {
        // User changed pages by using the arrow keys with alt - comparison image changes to first or last depending
        if (offset < prevOffset) {
          // We've gone backwards
          const lastImage = imageDTOs[imageDTOs.length - 1];
          if (lastImage && imageToCompare?.image_name !== lastImage.image_name) {
            dispatch(imageToCompareChanged(lastImage));
          }
        } else {
          // We've gone forwards
          const firstImage = imageDTOs[0];
          if (firstImage && imageToCompare?.image_name !== firstImage.image_name) {
            dispatch(imageToCompareChanged(firstImage));
          }
        }
        return;
      }
    },
  });
};
