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
  useGalleryImageDroppable,
  type GalleryBoardDropData,
  type GalleryBoardDropResolution,
  type GalleryImageDragData,
  type GalleryImageDragItem,
  type GalleryItemDragData,
  type GalleryItemDragId,
  type GalleryItemDragSource,
} from './ui/galleryDnd';

import { getGalleryImageFullUrl, getGalleryImageThumbnailUrl } from './data/imageUrls';
import { getGalleryVideoFullUrl, getGalleryVideoThumbnailUrl } from './data/videoUrls';

export const galleryImageUrls = {
  full: getGalleryImageFullUrl,
  thumbnail: getGalleryImageThumbnailUrl,
} as const;

export const galleryVideoUrls = {
  full: getGalleryVideoFullUrl,
  thumbnail: getGalleryVideoThumbnailUrl,
} as const;
