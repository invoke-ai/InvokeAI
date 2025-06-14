import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { imagesApi } from 'services/api/endpoints/images';

export const addEnsureImageIsSelectedListener = (startAppListening: AppStartListening) => {
  // When we list images, if no images is selected, select the first one.
  startAppListening({
    matcher: imagesApi.endpoints.listImages.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      const selection = getState().gallery.selection;
      if (selection.length === 0) {
        dispatch(imageSelected(action.payload.items[0] ?? null));
      }
    },
  });
};
