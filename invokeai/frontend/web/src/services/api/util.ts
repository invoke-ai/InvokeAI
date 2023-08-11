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

  const cachedPinnedImages = [];
  const cachedUnpinnedImages = [];

  for (let index = 0; index < totalCachedImageDtos.length; index++) {
    const image = totalCachedImageDtos[index];
    if (image?.pinned) cachedPinnedImages.push(image)
    if (!image?.pinned) cachedUnpinnedImages.push(image)
  }

  const lastPinnedImage = cachedPinnedImages[cachedPinnedImages.length - 1];
  const lastUnpinnedImage = cachedUnpinnedImages[cachedUnpinnedImages.length - 1];

  if (!lastPinnedImage || !lastUnpinnedImage) {
    // satisfy TS gods, we already confirmed the array has more than one image
    return false;
  }

  if (imageDTO.pinned) {
    // if pinning or already pinned, want to look in list of pinned images 
    const createdDate = new Date(imageDTO.created_at);
    const oldestDate = new Date(lastPinnedImage.created_at);
    return createdDate >= oldestDate;
  } else {
    // if unpinning or already unpinned, want to look in list of unpinned images 
    const createdDate = new Date(imageDTO.created_at);
    const oldestDate = new Date(lastUnpinnedImage.created_at);
    return createdDate >= oldestDate;
  }

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
