import { ImageDTO } from '../types';
import { ImageCache, imagesSelectors } from './images';

export const getIsImageInDateRange = (
  data: ImageCache | undefined,
  imageDTO: ImageDTO
) => {
  if (!data) {
    return false;
  }
  const cacheImageDTOS = imagesSelectors.selectAll(data);

  if (cacheImageDTOS.length > 1) {
    // Images are sorted by `created_at` DESC
    // check if the image is newer than the oldest image in the cache
    const createdDate = new Date(imageDTO.created_at);
    const oldestDate = new Date(
      cacheImageDTOS[cacheImageDTOS.length - 1].created_at
    );
    return createdDate >= oldestDate;
  } else if ([0, 1].includes(cacheImageDTOS.length)) {
    // if there are only 1 or 0 images in the cache, we consider the image to be in the date range
    return true;
  }
  return false;
};

/**
 * Determines the action we should take when an image may need to be added or updated in a cache.
 */
export const getCacheAction = (
  data: ImageCache | undefined,
  imageDTO: ImageDTO
): 'add' | 'update' | 'none' => {
  const isInDateRange = getIsImageInDateRange(data, imageDTO);
  const isCacheFullyPopulated = data && data.total === data.ids.length;
  const shouldUpdateCache =
    Boolean(isInDateRange) || Boolean(isCacheFullyPopulated);

  const isImageInCache = data && data.ids.includes(imageDTO.image_name);

  if (shouldUpdateCache && isImageInCache) {
    return 'update';
  }

  if (shouldUpdateCache && !isImageInCache) {
    return 'add';
  }

  return 'none';
};
