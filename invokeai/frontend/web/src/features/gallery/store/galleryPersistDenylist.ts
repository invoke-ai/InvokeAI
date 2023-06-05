import { GalleryState } from './gallerySlice';

/**
 * Gallery slice persist denylist
 */
export const galleryPersistDenylist: (keyof GalleryState)[] = [
  'shouldAutoSwitchToNewImages',
];
