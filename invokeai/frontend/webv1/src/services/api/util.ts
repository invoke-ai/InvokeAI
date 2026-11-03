import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import queryString from 'query-string';
import { buildV1Url } from 'services/api';

import type { ImageDTO, ListGalleryItemsArgs, ListImagesArgs, ListVideosArgs } from './types';

export const getCategories = (imageDTO: ImageDTO) => {
  if (IMAGE_CATEGORIES.includes(imageDTO.image_category)) {
    return IMAGE_CATEGORIES;
  }
  return ASSETS_CATEGORIES;
};

// Helper to create the url for the listImages endpoint. Also we use it to create the cache key.
export const getListImagesUrl = (queryArgs: ListImagesArgs) =>
  buildV1Url(`images/?${queryString.stringify(queryArgs, { arrayFormat: 'none' })}`);

// Helper to create the url for the listVideos endpoint. Also we use it to create the cache key.
export const getListVideosUrl = (queryArgs: ListVideosArgs) =>
  buildV1Url(`videos/?${queryString.stringify(queryArgs, { arrayFormat: 'none' })}`);

// Helper to create the url for the polymorphic listGalleryItems endpoint.
export const getListGalleryItemsUrl = (queryArgs: ListGalleryItemsArgs) =>
  buildV1Url(`gallery/items/?${queryString.stringify(queryArgs, { arrayFormat: 'none' })}`);
