import { createAction } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import type { RootState } from 'app/store/store';
import { selectImageCollectionQueryArgs } from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { uniq } from 'lodash-es';
import { imagesApi } from 'services/api/endpoints/images';
import type { ImageCategory, ImageDTO, SQLiteDirection } from 'services/api/types';

// Type for image collection query arguments
type ImageCollectionQueryArgs = {
  board_id?: string;
  categories?: ImageCategory[];
  search_term?: string;
  order_dir?: SQLiteDirection;
  is_intermediate: boolean;
};

/**
 * Helper function to get all cached image data from collection queries
 * Returns a combined array of starred images followed by unstarred images
 */
const getCachedImageList = (state: RootState, queryArgs: ImageCollectionQueryArgs): ImageDTO[] => {
  const countsQueryResult = imagesApi.endpoints.getImageCollectionCounts.select(queryArgs)(state);

  if (!countsQueryResult.data) {
    return [];
  }

  const starredCount = countsQueryResult.data.starred_count ?? 0;
  const totalCount = countsQueryResult.data.total_count ?? 0;
  const unstarredCount = totalCount - starredCount;

  const imageDTOs: ImageDTO[] = [];

  // Add starred images first (in order)
  if (starredCount > 0) {
    for (let offset = 0; offset < starredCount; offset += 50) {
      const queryResult = imagesApi.endpoints.getImageCollection.select({
        collection: 'starred',
        offset,
        limit: 50,
        ...queryArgs,
      })(state);

      if (queryResult.data?.items) {
        imageDTOs.push(...queryResult.data.items);
      }
    }
  }

  // Add unstarred images (in order)
  if (unstarredCount > 0) {
    for (let offset = 0; offset < unstarredCount; offset += 50) {
      const queryResult = imagesApi.endpoints.getImageCollection.select({
        collection: 'unstarred',
        offset,
        limit: 50,
        ...queryArgs,
      })(state);

      if (queryResult.data?.items) {
        imageDTOs.push(...queryResult.data.items);
      }
    }
  }

  return imageDTOs;
};

export const galleryImageClicked = createAction<{
  imageName: string;
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
      const { imageName, shiftKey, ctrlKey, metaKey, altKey } = action.payload;
      const state = getState();
      const queryArgs = selectImageCollectionQueryArgs(state);

      // Get all cached image data
      const imageDTOs = getCachedImageList(state, queryArgs);

      // If we don't have the image data cached, we can't perform selection operations
      // This can happen if the user clicks on an image before all data is loaded
      if (imageDTOs.length === 0) {
        // For basic click without modifiers, we can still set selection
        if (!shiftKey && !ctrlKey && !metaKey && !altKey) {
          dispatch(selectionChanged([imageName]));
        }
        return;
      }

      const selection = state.gallery.selection;

      if (altKey) {
        if (state.gallery.imageToCompare === imageName) {
          dispatch(imageToCompareChanged(null));
        } else {
          dispatch(imageToCompareChanged(imageName));
        }
      } else if (shiftKey) {
        const rangeEndImageName = imageName;
        const lastSelectedImage = selection.at(-1);
        const lastClickedIndex = imageDTOs.findIndex((n) => n.image_name === lastSelectedImage);
        const currentClickedIndex = imageDTOs.findIndex((n) => n.image_name === rangeEndImageName);
        if (lastClickedIndex > -1 && currentClickedIndex > -1) {
          // We have a valid range!
          const start = Math.min(lastClickedIndex, currentClickedIndex);
          const end = Math.max(lastClickedIndex, currentClickedIndex);
          const imagesToSelect = imageDTOs.slice(start, end + 1).map(({ image_name }) => image_name);
          dispatch(selectionChanged(uniq(selection.concat(imagesToSelect))));
        }
      } else if (ctrlKey || metaKey) {
        if (selection.some((n) => n === imageName) && selection.length > 1) {
          dispatch(selectionChanged(uniq(selection.filter((n) => n !== imageName))));
        } else {
          dispatch(selectionChanged(uniq(selection.concat(imageName))));
        }
      } else {
        dispatch(selectionChanged([imageName]));
      }
    },
  });
};
