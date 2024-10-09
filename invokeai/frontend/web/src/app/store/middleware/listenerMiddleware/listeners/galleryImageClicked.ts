import { createAction } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

export const galleryImageClicked = createAction<{
  imageDTO: ImageDTO;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
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
    effect: (action, { dispatch, getState }) => {
      const { imageDTO, shiftKey, ctrlKey, metaKey, altKey } = action.payload;
      const state = getState();
      const queryArgs = selectListImagesQueryArgs(state);
      const queryResult = imagesApi.endpoints.listImages.select(queryArgs)(state);

      if (!queryResult.data) {
        // Should never happen if we have clicked a gallery image
        return;
      }

      const imageDTOs = queryResult.data.items;
      const selection = state.gallery.selection;

      if (altKey) {
        if (state.gallery.imageToCompare?.image_name === imageDTO.image_name) {
          dispatch(imageToCompareChanged(null));
        } else {
          dispatch(imageToCompareChanged(imageDTO));
        }
      } else if (shiftKey) {
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
