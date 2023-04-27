import { LightboxState } from './lightboxSlice';

/**
 * Lightbox slice persist denylist
 */
const itemsToDenylist: (keyof LightboxState)[] = ['isLightboxOpen'];

export const lightboxDenylist = itemsToDenylist.map(
  (denylistItem) => `lightbox.${denylistItem}`
);
