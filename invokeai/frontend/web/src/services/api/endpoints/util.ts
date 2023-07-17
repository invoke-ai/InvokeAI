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
    const start = new Date(
      cacheImageDTOS[cacheImageDTOS.length - 1].created_at
    );
    const end = new Date(cacheImageDTOS[0].created_at);
    if (
      new Date(imageDTO.created_at) >= start &&
      new Date(imageDTO.created_at) <= end
    ) {
      return true;
    }
  } else if ([0, 1].includes(cacheImageDTOS.length)) {
    return true;
  }
  return false;
};
