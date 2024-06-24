import { dateComparator } from 'common/util/dateComparator';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import queryString from 'query-string';
import { buildV1Url } from 'services/api';

import type { ImageDTO, ListImagesArgs } from './types';

export const getCategories = (imageDTO: ImageDTO) => {
  if (IMAGE_CATEGORIES.includes(imageDTO.image_category)) {
    return IMAGE_CATEGORIES;
  }
  return ASSETS_CATEGORIES;
};

export const imageListDefaultSort = () => {
  return (a: ImageDTO, b: ImageDTO) => {
    if (a.starred && !b.starred) {
      return -1;
    }
    if (!a.starred && b.starred) {
      return 1;
    }
    return dateComparator(b.created_at, a.created_at);
  };
};

// Helper to create the url for the listImages endpoint. Also we use it to create the cache key.
export const getListImagesUrl = (queryArgs: ListImagesArgs) =>
  buildV1Url(`images/?${queryString.stringify(queryArgs, { arrayFormat: 'none' })}`);
