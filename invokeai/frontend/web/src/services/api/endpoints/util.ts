import { EntityState } from '@reduxjs/toolkit';
import { ImageDTO } from '../types';
import { imagesSelectors } from './images';

export const getIsImageInDateRange = (
  data: (EntityState<ImageDTO> & { total: number }) | undefined,
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
