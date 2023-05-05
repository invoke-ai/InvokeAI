import { GalleryState } from './gallerySlice';

/**
 * Gallery slice persist denylist
 */
const itemsToDenylist: (keyof GalleryState)[] = [
  'currentCategory',
  'shouldAutoSwitchToNewImages',
];

export const galleryPersistDenylist: (keyof GalleryState)[] = [
  'currentCategory',
  'shouldAutoSwitchToNewImages',
];

export const galleryDenylist = itemsToDenylist.map(
  (denylistItem) => `gallery.${denylistItem}`
);
