import { createEntityAdapter } from '@reduxjs/toolkit';
import { getSelectorsOptions } from 'app/store/createMemoizedSelector';
import { dateComparator } from 'common/util/dateComparator';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import queryString from 'query-string';

import type { ImageCache, ImageDTO, ListImagesArgs } from './types';

export const getIsImageInDateRange = (data: ImageCache | undefined, imageDTO: ImageDTO) => {
  if (!data) {
    return false;
  }

  const totalCachedImageDtos = imagesSelectors.selectAll(data);

  if (totalCachedImageDtos.length <= 1) {
    return true;
  }

  const cachedStarredImages = [];
  const cachedUnstarredImages = [];

  for (let index = 0; index < totalCachedImageDtos.length; index++) {
    const image = totalCachedImageDtos[index];
    if (image?.starred) {
      cachedStarredImages.push(image);
    }
    if (!image?.starred) {
      cachedUnstarredImages.push(image);
    }
  }

  if (imageDTO.starred) {
    const lastStarredImage = cachedStarredImages[cachedStarredImages.length - 1];
    // if starring or already starred, want to look in list of starred images
    if (!lastStarredImage) {
      return true;
    } // no starred images showing, so always show this one
    const createdDate = new Date(imageDTO.created_at);
    const oldestDate = new Date(lastStarredImage.created_at);
    return createdDate >= oldestDate;
  } else {
    const lastUnstarredImage = cachedUnstarredImages[cachedUnstarredImages.length - 1];
    // if unstarring or already unstarred, want to look in list of unstarred images
    if (!lastUnstarredImage) {
      return false;
    } // no unstarred images showing, so don't show this one
    const createdDate = new Date(imageDTO.created_at);
    const oldestDate = new Date(lastUnstarredImage.created_at);
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
export const imagesAdapter = createEntityAdapter<ImageDTO, string>({
  selectId: (image) => image.image_name,
  sortComparer: (a, b) => {
    // Compare starred images first
    if (a.starred && !b.starred) {
      return -1;
    }
    if (!a.starred && b.starred) {
      return 1;
    }
    return dateComparator(b.created_at, a.created_at);
  },
});

// Create selectors for the adapter.
export const imagesSelectors = imagesAdapter.getSelectors(undefined, getSelectorsOptions);

// Helper to create the url for the listImages endpoint. Also we use it to create the cache key.
export const getListImagesUrl = (queryArgs: ListImagesArgs) =>
  `images/?${queryString.stringify(queryArgs, { arrayFormat: 'none' })}`;
