import { GalleryState } from './gallerySlice';

/**
 * Gallery slice persist blacklist
 */
const itemsToBlacklist: (keyof GalleryState)[] = [
  'categories',
  'currentCategory',
  'currentImage',
  'currentImageUuid',
  'shouldAutoSwitchToNewImages',
  'intermediateImage',
];

export const galleryBlacklist = itemsToBlacklist.map(
  (blacklistItem) => `gallery.${blacklistItem}`
);
