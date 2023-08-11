import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
} from 'features/gallery/store/types';
import { ImageCache, ImageDTO, ListImagesArgs } from './types';
import { createEntityAdapter } from '@reduxjs/toolkit';
import { dateComparator } from 'common/util/dateComparator';
import queryString from 'query-string';

export const getIsImageInDateRange = (
  data: ImageCache | undefined,
  imageDTO: ImageDTO
) => {
  if (!data) {
    return false;
  }

  const totalCachedImageDtos = imagesSelectors.selectAll(data);

  if (totalCachedImageDtos.length <= 1) {
    return true;
  }

  const cacheImageDTOSForPinnedState = totalCachedImageDtos.filter((image) => image.pinned === imageDTO.pinned);

  if (cacheImageDTOSForPinnedState.length > 1) {
    // Images are sorted by `pinned` DESC and then `created_at` DESC
    // check if the image is newer than the oldest image in the cache for either the pinned group or unpinned group
    const createdDate = new Date(imageDTO.created_at);
    const oldestImage = cacheImageDTOSForPinnedState[cacheImageDTOSForPinnedState.length - 1];
    if (!oldestImage) {
      // satisfy TS gods, we already confirmed the array has more than one image
      return false;
    }
    const oldestDate = new Date(oldestImage.created_at);
    return createdDate >= oldestDate;
  }
  return false;
};

export const getCategories = (imageDTO: ImageDTO) => {
  if (IMAGE_CATEGORIES.includes(imageDTO.image_category)) {
    return IMAGE_CATEGORIES;
  }
  return ASSETS_CATEGORIES;
};

// The adapter is not actually the data store - it just provides helper functions to interact
// with some other store of data. We will use the RTK Query cache as that store.
export const imagesAdapter = createEntityAdapter<ImageDTO>({
  selectId: (image) => image.image_name,
  sortComparer: (a, b) => {
    // Compare pinned images first
    if (a.pinned && !b.pinned) {
      return -1;
    }
    if (!a.pinned && b.pinned) {
      return 1;
    }
    return dateComparator(b.created_at, a.created_at)
  },
});

// Create selectors for the adapter.
export const imagesSelectors = imagesAdapter.getSelectors();

// Helper to create the url for the listImages endpoint. Also we use it to create the cache key.
export const getListImagesUrl = (queryArgs: ListImagesArgs) =>
  `images/?${queryString.stringify(queryArgs, { arrayFormat: 'none' })}`;
