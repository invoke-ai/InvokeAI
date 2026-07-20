export {
  getGalleryBoardDropData,
  getGalleryBoardDropId,
  getGalleryImageDragData,
  getGalleryImageDragId,
  getGalleryImageNamesOutsideBoard,
  isGalleryBoardDropData,
  isGalleryImageDragData,
  type GalleryBoardDropData,
  type GalleryImageDragData,
  type GalleryImageDragImage,
} from './ui/galleryDnd';

import { getGalleryImageFullUrl, getGalleryImageThumbnailUrl } from './data/imageUrls';

export const galleryImageUrls = {
  full: getGalleryImageFullUrl,
  thumbnail: getGalleryImageThumbnailUrl,
} as const;
