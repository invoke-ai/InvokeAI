import { initialGalleryState } from './gallerySlice';

/**
 * Gallery slice persist denylist
 */
export const galleryPersistDenylist: (keyof typeof initialGalleryState)[] = [
  'selection',
  'entities',
  'ids',
  'isLoading',
  'limit',
  'offset',
  'selectedBoardId',
  'categories',
  'galleryView',
  'total',
  'isInitialized',
];
