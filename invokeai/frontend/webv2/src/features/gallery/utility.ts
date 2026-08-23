export {
  getGalleryBoardDropData,
  getGalleryBoardDropId,
  getGalleryItemDragData,
  getGalleryItemDragId,
  getGalleryItemRefsOutsideBoard,
  isGalleryBoardDropData,
  isGalleryImageDragData,
  isGalleryItemDragData,
  isSingleGalleryImageDragData,
  isSingleGalleryVideoDragData,
  resolveGalleryBoardDrop,
  useGalleryImageDroppable,
  useGalleryItemDroppable,
  type GalleryBoardDropData,
  type GalleryBoardDropResolution,
  type GalleryImageDragData,
  type GalleryImageDragItem,
  type GalleryItemDragData,
  type GalleryItemDragId,
  type GalleryItemDragSource,
} from './ui/galleryDnd';

export { GalleryDragCursor } from './ui/GalleryDragCursor';

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
