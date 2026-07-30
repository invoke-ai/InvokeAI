export {
  getGalleryBoardDropData,
  getGalleryBoardDropId,
  getGalleryItemDragData,
  getGalleryItemDragId,
  getGalleryItemRefsOutsideBoard,
  isGalleryBoardDropData,
  isGalleryImageDragData,
  isGalleryItemDragData,
  resolveGalleryBoardDrop,
  type GalleryBoardDropData,
  type GalleryBoardDropResolution,
  type GalleryImageDragData,
  type GalleryImageDragItem,
  type GalleryItemDragData,
} from './ui/galleryDnd';

import { getGalleryImageFullUrl, getGalleryImageThumbnailUrl } from './data/imageUrls';

export const galleryImageUrls = {
  full: getGalleryImageFullUrl,
  thumbnail: getGalleryImageThumbnailUrl,
} as const;
