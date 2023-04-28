import { GalleryState } from './gallerySlice';

/**
 * Gallery slice persist denylist
 */
const itemsToDenylist: (keyof GalleryState)[] = [
  'categories',
  'currentCategory',
  'currentImage',
  'currentImageUuid',
  'shouldAutoSwitchToNewImages',
  'intermediateImage',
];

export const galleryDenylist = itemsToDenylist.map(
  (denylistItem) => `gallery.${denylistItem}`
);
