import { createAction } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { selectionChanged } from 'features/gallery/store/gallerySlice';
import { imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { imagesSelectors } from 'services/api/util';

export const galleryImageClicked = createAction<{
  imageDTO: ImageDTO;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
}>('gallery/imageClicked');

/**
 * This listener handles the logic for selecting images in the gallery.
 *
 * Previously, this logic was in a `useCallback` with the whole gallery selection as a dependency. Every time
 * the selection changed, the callback got recreated and all images rerendered. This could easily block for
 * hundreds of ms, more for lower end devices.
 *
 * Moving this logic into a listener means we don't need to recalculate anything dynamically and the gallery
 * is much more responsive.
 */

export const addGalleryImageClickedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: galleryImageClicked,
    effect: async (action, { dispatch, getState }) => {
      const { imageDTO, shiftKey, ctrlKey, metaKey } = action.payload;
      const state = getState();
      const queryArgs = selectListImagesQueryArgs(state);
      const { data: listImagesData } = imagesApi.endpoints.listImages.select(queryArgs)(state);

      if (!listImagesData) {
        // Should never happen if we have clicked a gallery image
        return;
      }

      const imageDTOs = imagesSelectors.selectAll(listImagesData);
      const selection = state.gallery.selection;

      if (shiftKey) {
        const rangeEndImageName = imageDTO.image_name;
        const lastSelectedImage = selection[selection.length - 1]?.image_name;
        const lastClickedIndex = imageDTOs.findIndex((n) => n.image_name === lastSelectedImage);
        const currentClickedIndex = imageDTOs.findIndex((n) => n.image_name === rangeEndImageName);
        if (lastClickedIndex > -1 && currentClickedIndex > -1) {
          // We have a valid range!
          const start = Math.min(lastClickedIndex, currentClickedIndex);
          const end = Math.max(lastClickedIndex, currentClickedIndex);
          const imagesToSelect = imageDTOs.slice(start, end + 1);
          dispatch(selectionChanged(selection.concat(imagesToSelect)));
        }
      } else if (ctrlKey || metaKey) {
        if (selection.some((i) => i.image_name === imageDTO.image_name) && selection.length > 1) {
          dispatch(selectionChanged(selection.filter((n) => n.image_name !== imageDTO.image_name)));
        } else {
          dispatch(selectionChanged(selection.concat(imageDTO)));
        }
      } else {
        dispatch(selectionChanged([imageDTO]));
      }
    },
  });
};
