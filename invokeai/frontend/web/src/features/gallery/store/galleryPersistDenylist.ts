import { initialGalleryState } from './gallerySlice';

/**
 * Gallery slice persist denylist
 */
export const galleryPersistDenylist: (keyof typeof initialGalleryState)[] = [
  'selection',
  'selectedBoardId',
  'galleryView',
];
