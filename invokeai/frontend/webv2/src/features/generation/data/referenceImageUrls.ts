import type { CroppableImageWithDims } from '@features/generation/core/types';

import { galleryImageUrls } from '@features/gallery/utility';
import { getEffectiveReferenceImage } from '@features/generation/core/referenceImage';

export const getReferenceImageUrls = (image: CroppableImageWithDims) => {
  const { image_name } = getEffectiveReferenceImage(image);
  return { imageUrl: galleryImageUrls.full(image_name), thumbnailUrl: galleryImageUrls.thumbnail(image_name) };
};
