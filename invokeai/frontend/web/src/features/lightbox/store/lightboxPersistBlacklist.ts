import { LightboxState } from './lightboxSlice';

/**
 * Lightbox slice persist blacklist
 */
const itemsToBlacklist: (keyof LightboxState)[] = ['isLightboxOpen'];

export const lightboxBlacklist = itemsToBlacklist.map(
  (blacklistItem) => `lightbox.${blacklistItem}`
);
